#!/usr/bin/env python3
"""
TAES: Temporal-Aware Event Search

Implements three core modules described in TAES.md:
1) beam search (neural-symbolic style candidate expansion)
2) search tree generation (state transition tree)
3) multi-level state evaluation (hard constraints + soft heuristics)

Supports execution for both:
- TimeArena/LLM_test.py
- TravelPlanner/tools/planner/sole_planning.py
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urljoin
from typing import Dict, List, Optional, Set, Tuple

import requests


@dataclass
class TaskNode:
    task_id: str
    domain: str
    earliest_start: float = 0.0
    latest_end: Optional[float] = None
    duration_hint: Optional[float] = None
    depends_on: List[str] = field(default_factory=list)
    payload: Dict[str, str] = field(default_factory=dict)


@dataclass
class ConstraintProfile:
    max_retries: int = 1
    timeout_seconds: int = 3600
    stop_on_failure: bool = True


@dataclass
class SearchAction:
    task_id: str
    domain: str
    payload: Dict[str, str]
    est_duration: float
    est_cost: float
    label: str


@dataclass
class SearchState:
    state_id: str
    parent_id: Optional[str]
    depth: int
    t_now: float
    resources: Dict[str, float]
    constraints_met: Set[str]
    remaining: List[str]
    plan: List[SearchAction]
    score_hard: float = 1.0
    score_soft: float = 0.0
    score_total: float = 0.0
    terminal: bool = False


class TemporalKnowledgeGraph:
    """Task graph with temporal/dependency constraints."""

    def __init__(self) -> None:
        self.nodes: Dict[str, TaskNode] = {}

    def add_node(self, node: TaskNode) -> None:
        self.nodes[node.task_id] = node

    def _is_ready(self, node: TaskNode, finished: set[str], now: float) -> bool:
        if any(dep not in finished for dep in node.depends_on):
            return False
        if now < node.earliest_start:
            return False
        return True

    def plan_order(self) -> List[TaskNode]:
        pending = dict(self.nodes)
        finished: set[str] = set()
        order: List[TaskNode] = []
        now = time.time()

        while pending:
            ready = [n for n in pending.values() if self._is_ready(n, finished, now)]
            if not ready:
                # Fallback: if temporal constraints are too strict, choose dependency-feasible node.
                dep_ready = [
                    n for n in pending.values()
                    if all(dep in finished for dep in n.depends_on)
                ]
                if not dep_ready:
                    raise RuntimeError("TAES failed: dependency cycle detected in task graph.")
                dep_ready.sort(key=lambda n: n.earliest_start)
                chosen = dep_ready[0]
            else:
                ready.sort(key=lambda n: n.earliest_start)
                chosen = ready[0]

            order.append(chosen)
            finished.add(chosen.task_id)
            del pending[chosen.task_id]

        return order


class SearchTree:
    """Explicit search tree storage."""

    def __init__(self) -> None:
        self.nodes: Dict[str, SearchState] = {}
        self.children: Dict[str, List[str]] = {}

    def add(self, state: SearchState) -> None:
        self.nodes[state.state_id] = state
        if state.parent_id:
            self.children.setdefault(state.parent_id, []).append(state.state_id)
        self.children.setdefault(state.state_id, [])


class MultiLevelEvaluator:
    """
    V(S) = V_hard(S) * [epsilon + (1-epsilon) * V_soft(S)]
    """

    def __init__(self, epsilon: float, alpha_parallel: float, min_daily_budget: float, max_time: float) -> None:
        self.epsilon = epsilon
        self.alpha_parallel = alpha_parallel
        self.min_daily_budget = min_daily_budget
        self.max_time = max_time

    def hard_check(self, state: SearchState, action: SearchAction, tkg: TemporalKnowledgeGraph, serial: bool) -> bool:
        if action.task_id in state.constraints_met:
            return False
        if action.task_id not in state.remaining:
            return False

        task = tkg.nodes[action.task_id]
        # Dependency hard check.
        for dep in task.depends_on:
            if dep not in state.constraints_met:
                return False
        # Serial mode hard check.
        if serial and state.remaining:
            expected = min(state.remaining, key=lambda tid: tkg.nodes[tid].earliest_start)
            if action.task_id != expected:
                return False

        budget_total = state.resources.get("budget_total", 1e9)
        budget_used = state.resources.get("budget_used", 0.0)
        if budget_used + action.est_cost > budget_total:
            return False

        if state.t_now + action.est_duration > self.max_time:
            return False

        return True

    def soft_score(self, state: SearchState, max_depth: int, serial: bool) -> float:
        # Time efficiency estimate from TAES.md style.
        completed = len(state.constraints_met)
        r_time = completed / max(state.t_now, 1.0)
        parallel = 1.0 if (not serial and len(state.remaining) > 1) else 0.0
        r_time += self.alpha_parallel * parallel

        # Budget risk estimate from TAES.md style.
        budget_total = state.resources.get("budget_total", 1e9)
        budget_used = state.resources.get("budget_used", 0.0)
        remain_depth = max(1, max_depth - state.depth)
        r_budget = (budget_total - budget_used) / remain_depth
        if r_budget <= 0:
            return 0.0
        if r_budget < self.min_daily_budget:
            budget_factor = 0.1
        else:
            budget_factor = min(2.0, r_budget / self.min_daily_budget)

        # Keep score in stable range.
        return max(0.0, min(1.0, 0.5 * (r_time / (r_time + 1.0)) + 0.5 * (budget_factor / 2.0)))

    def final_score(self, hard_ok: bool, soft: float) -> float:
        v_hard = 1.0 if hard_ok else 0.0
        return v_hard * (self.epsilon + (1.0 - self.epsilon) * soft)


class TAESExecutor:
    def __init__(self, root: Path, profile: ConstraintProfile) -> None:
        self.root = root
        self.profile = profile
        self.python_bin = sys.executable

    def _load_env(self) -> Dict[str, str]:
        env = os.environ.copy()
        env_path = self.root / "TimeArena" / ".env"
        if env_path.exists():
            for line in env_path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                k = k.strip()
                v = v.strip().strip('"').strip("'")
                if k:
                    env[k] = v

        # Cross-project compatibility mapping.
        if "OPENAI_API_KEY" not in env and "LLM_API_KEY" in env:
            env["OPENAI_API_KEY"] = env["LLM_API_KEY"]
        if "OPENAI_API_BASE" not in env and "LLM_BASE_URL" in env:
            env["OPENAI_API_BASE"] = env["LLM_BASE_URL"]
        return env

    def _run_subprocess(
        self,
        cmd: List[str],
        cwd: Path,
        env: Dict[str, str],
        timeout_seconds: int,
    ) -> Tuple[int, str]:
        started = time.time()
        try:
            proc = subprocess.run(
                cmd,
                cwd=str(cwd),
                env=env,
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
                check=False,
            )
            elapsed = time.time() - started
            output = (proc.stdout or "") + ("\n" + proc.stderr if proc.stderr else "")
            output += f"\n[TAES] elapsed={elapsed:.2f}s"
            return proc.returncode, output
        except subprocess.TimeoutExpired as exc:
            output = (exc.stdout or "") + ("\n" + exc.stderr if exc.stderr else "")
            output += f"\n[TAES] timeout after {timeout_seconds}s"
            return 124, output

    def run_timearena(self, payload: Dict[str, str], env: Dict[str, str]) -> Tuple[int, str]:
        script = self.root / "TimeArena" / "LLM_test.py"
        cwd = self.root / "TimeArena"
        task_name = payload.get("task_name", "household1")
        prompting = payload.get("prompting", "reflexion")
        total_time = payload.get("total_time", "60")
        save_path = payload.get("save_path", "./trajectory/taes")
        save_name = payload.get("save_name", task_name)
        lm = payload.get("lm", "custom")
        model_name = payload.get("model_name", env.get("LLM_MODEL_NAME", "gpt-4o-2024-08-06"))
        cmd = [
            self.python_bin,
            str(script),
            "--taskName",
            task_name,
            "--prompting",
            prompting,
            "--lm",
            lm,
            "--model_name",
            model_name,
            "--total_time",
            str(total_time),
            "--save_path",
            save_path,
            "--save_name",
            save_name,
        ]
        return self._run_subprocess(cmd, cwd=cwd, env=env, timeout_seconds=self.profile.timeout_seconds)

    def run_travelplanner(self, payload: Dict[str, str], env: Dict[str, str]) -> Tuple[int, str]:
        script = self.root / "TravelPlanner" / "tools" / "planner" / "sole_planning.py"
        cwd = script.parent
        set_type = payload.get("set_type", "validation")
        output_dir = payload.get("output_dir", str((self.root / "TravelPlanner" / "evaluation").resolve()))
        strategy = payload.get("strategy", "direct")
        model_name = payload.get("model_name", env.get("LLM_MODEL_NAME", "gpt-4o-2024-08-06"))
        cmd = [
            self.python_bin,
            str(script),
            "--set_type",
            set_type,
            "--output_dir",
            output_dir,
            "--model_name",
            model_name,
            "--strategy",
            strategy,
        ]
        return self._run_subprocess(cmd, cwd=cwd, env=env, timeout_seconds=self.profile.timeout_seconds)

    def execute(self, node: TaskNode) -> Dict[str, str]:
        env = self._load_env()
        runner = self.run_timearena if node.domain == "timearena" else self.run_travelplanner

        attempts = 0
        last_code = 1
        last_output = ""
        while attempts <= self.profile.max_retries:
            attempts += 1
            code, output = runner(node.payload, env)
            last_code = code
            last_output = output
            if code == 0:
                break
            if attempts <= self.profile.max_retries:
                time.sleep(1.0)

        status = "success" if last_code == 0 else "failed"
        return {
            "task_id": node.task_id,
            "domain": node.domain,
            "status": status,
            "return_code": str(last_code),
            "attempts": str(attempts),
            "log": last_output,
        }


class LLMActionGenerator:
    """
    Use LLM to choose next actions from environment-provided options.
    """

    def __init__(self, env: Dict[str, str], timeout_seconds: int = 60, model_name: str = "") -> None:
        base_url = env.get("LLM_BASE_URL") or env.get("OPENAI_API_BASE") or ""
        api_key = env.get("LLM_API_KEY") or env.get("OPENAI_API_KEY") or ""
        model = model_name or env.get("LLM_MODEL_NAME") or env.get("OPENAI_MODEL_NAME") or ""

        self.base_url = (base_url.rstrip("/") + "/") if base_url else ""
        self.api_key = api_key
        self.model_name = model
        self.timeout_seconds = timeout_seconds
        self.enabled = bool(self.base_url and self.api_key and self.model_name)
        self.last_error = ""

    def _build_prompt(self, state: SearchState, options: List[Dict[str, str]], k: int) -> str:
        state_desc = {
            "state_id": state.state_id,
            "depth": state.depth,
            "time_now": state.t_now,
            "completed": sorted(list(state.constraints_met)),
            "remaining": state.remaining,
            "budget_total": state.resources.get("budget_total", 0.0),
            "budget_used": state.resources.get("budget_used", 0.0),
        }
        return (
            "You are a planning policy model. Select next best actions from available options.\n"
            "Return ONLY JSON with schema: {\"selected\": [int, ...]}.\n"
            f"Pick at most {k} unique indices.\n"
            f"Current state:\n{json.dumps(state_desc, ensure_ascii=False)}\n"
            f"Options:\n{json.dumps(options, ensure_ascii=False)}"
        )

    def _request(self, prompt: str) -> str:
        endpoint = urljoin(self.base_url, "chat/completions")
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0,
            "max_tokens": 300,
        }
        session = requests.Session()
        session.trust_env = False
        resp = session.post(endpoint, headers=headers, json=payload, timeout=self.timeout_seconds)
        resp.raise_for_status()
        data = resp.json()
        return ((data.get("choices") or [{}])[0].get("message") or {}).get("content", "")

    def generate_indices(self, state: SearchState, options: List[Dict[str, str]], k: int) -> List[int]:
        if not options:
            return []
        if not self.enabled:
            return list(range(min(k, len(options))))
        prompt = self._build_prompt(state, options, k)
        try:
            content = self._request(prompt).strip()
            # tolerate fenced output
            if "```" in content:
                parts = content.split("```")
                for part in parts:
                    part = part.strip()
                    if part.startswith("{") and part.endswith("}"):
                        content = part
                        break
            obj = json.loads(content)
            selected = obj.get("selected", [])
            valid = []
            for x in selected:
                if isinstance(x, int) and 0 <= x < len(options) and x not in valid:
                    valid.append(x)
                if len(valid) >= k:
                    break
            if valid:
                return valid
        except Exception as e:
            self.last_error = f"{type(e).__name__}: {e}"
        return list(range(min(k, len(options))))


class BeamSearchPlanner:
    def __init__(
        self,
        tkg: TemporalKnowledgeGraph,
        evaluator: MultiLevelEvaluator,
        action_generator: LLMActionGenerator,
        beam_width: int,
        branch_factor: int,
        max_depth: int,
        serial: bool,
    ) -> None:
        self.tkg = tkg
        self.evaluator = evaluator
        self.action_generator = action_generator
        self.beam_width = max(1, beam_width)
        self.branch_factor = max(1, branch_factor)
        self.max_depth = max(1, max_depth)
        self.serial = serial
        self.tree = SearchTree()
        self._state_counter = 0

    def _next_id(self) -> str:
        self._state_counter += 1
        return f"s{self._state_counter}"

    def _initial_state(self, budget_total: float) -> SearchState:
        state = SearchState(
            state_id=self._next_id(),
            parent_id=None,
            depth=0,
            t_now=0.0,
            resources={"budget_total": budget_total, "budget_used": 0.0},
            constraints_met=set(),
            remaining=list(self.tkg.nodes.keys()),
            plan=[],
            score_hard=1.0,
            score_soft=0.0,
            score_total=0.0,
            terminal=False,
        )
        self.tree.add(state)
        return state

    def _domain_options(self, node: TaskNode) -> List[str]:
        if node.domain == "timearena":
            raw = node.payload.get("prompting_options", "react,reflexion,selfplan")
            return [x.strip() for x in raw.split(",") if x.strip()]
        raw = node.payload.get("strategy_options", "direct,cot,react,reflexion")
        return [x.strip() for x in raw.split(",") if x.strip()]

    def _apply_option(self, node: TaskNode, option: str) -> Tuple[Dict[str, str], str]:
        p = dict(node.payload)
        if node.domain == "timearena":
            p["prompting"] = option
            return p, f"timearena:{option}"
        p["strategy"] = option
        return p, f"travelplanner:{option}"

    def _estimate_action(self, domain: str, label: str) -> Tuple[float, float]:
        if domain == "timearena":
            # approximate runtime and "cost risk"
            if "selfplan" in label:
                return 30.0, 200.0
            if "react" in label:
                return 40.0, 260.0
            return 45.0, 300.0
        if "direct" in label:
            return 45.0, 350.0
        if "cot" in label:
            return 55.0, 420.0
        if "react" in label:
            return 80.0, 520.0
        return 95.0, 600.0

    def _generate_actions(self, state: SearchState) -> List[SearchAction]:
        option_pool: List[Dict[str, str]] = []
        materialized: List[Tuple[str, str, Dict[str, str], str]] = []
        for task_id in state.remaining:
            node = self.tkg.nodes[task_id]
            for opt in self._domain_options(node):
                payload, label = self._apply_option(node, opt)
                materialized.append((task_id, node.domain, payload, label))
                option_pool.append(
                    {
                        "task_id": task_id,
                        "domain": node.domain,
                        "option": opt,
                        "label": label,
                    }
                )

        selected_indices = self.action_generator.generate_indices(state, option_pool, self.branch_factor)
        actions: List[SearchAction] = []
        for idx in selected_indices:
            task_id, domain, payload, label = materialized[idx]
            dur, cost = self._estimate_action(domain, label)
            actions.append(
                SearchAction(
                    task_id=task_id,
                    domain=domain,
                    payload=payload,
                    est_duration=dur,
                    est_cost=cost,
                    label=label,
                )
            )
        return actions

    def _execute_action(self, state: SearchState, action: SearchAction) -> SearchState:
        new_constraints = set(state.constraints_met)
        new_constraints.add(action.task_id)
        new_remaining = [x for x in state.remaining if x != action.task_id]
        new_resources = dict(state.resources)
        new_resources["budget_used"] = new_resources.get("budget_used", 0.0) + action.est_cost

        # Event-driven time jump: move to next decision point directly.
        t_next = state.t_now + action.est_duration
        child = SearchState(
            state_id=self._next_id(),
            parent_id=state.state_id,
            depth=state.depth + 1,
            t_now=t_next,
            resources=new_resources,
            constraints_met=new_constraints,
            remaining=new_remaining,
            plan=state.plan + [action],
            terminal=len(new_remaining) == 0,
        )
        return child

    def search(self, budget_total: float) -> Tuple[SearchState, SearchTree]:
        beam = [self._initial_state(budget_total)]
        best = beam[0]

        for _ in range(self.max_depth):
            candidates: List[SearchState] = []
            for state in beam:
                if state.terminal:
                    candidates.append(state)
                    continue
                actions = self._generate_actions(state)
                for action in actions:
                    hard_ok = self.evaluator.hard_check(state, action, self.tkg, self.serial)
                    if not hard_ok:
                        continue
                    nxt = self._execute_action(state, action)
                    soft = self.evaluator.soft_score(nxt, self.max_depth, self.serial)
                    total = self.evaluator.final_score(True, soft)
                    nxt.score_hard = 1.0
                    nxt.score_soft = soft
                    nxt.score_total = total
                    self.tree.add(nxt)
                    candidates.append(nxt)

            if not candidates:
                break
            candidates.sort(key=lambda s: s.score_total, reverse=True)
            beam = candidates[: self.beam_width]
            if beam and beam[0].score_total > best.score_total:
                best = beam[0]
            if all(s.terminal for s in beam):
                break

        beam.sort(key=lambda s: s.score_total, reverse=True)
        if beam:
            best = beam[0]
        return best, self.tree


def build_graph(args: argparse.Namespace) -> TemporalKnowledgeGraph:
    graph = TemporalKnowledgeGraph()
    now = time.time()

    if args.task in ("timearena", "both"):
        graph.add_node(
            TaskNode(
                task_id="timearena_eval",
                domain="timearena",
                earliest_start=now,
                payload={
                    "task_name": args.ta_task_name,
                    "prompting": args.ta_prompting,
                    "prompting_options": args.ta_prompting_options,
                    "total_time": str(args.ta_total_time),
                    "save_path": args.ta_save_path,
                    "save_name": args.ta_save_name,
                    "lm": args.ta_lm,
                    "model_name": args.ta_model_name or "",
                },
            )
        )

    if args.task in ("travelplanner", "both"):
        deps = ["timearena_eval"] if args.task == "both" and args.serial else []
        graph.add_node(
            TaskNode(
                task_id="travelplanner_eval",
                domain="travelplanner",
                earliest_start=now,
                depends_on=deps,
                payload={
                    "set_type": args.tp_set_type,
                    "output_dir": args.tp_output_dir,
                    "strategy": args.tp_strategy,
                    "strategy_options": args.tp_strategy_options,
                    "model_name": args.tp_model_name or "",
                },
            )
        )
    return graph


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TAES temporal event search runner")
    parser.add_argument("--task", choices=["timearena", "travelplanner", "both"], default="both")
    parser.add_argument("--serial", action="store_true", help="Run both tasks in series with dependency.")
    parser.add_argument("--max_retries", type=int, default=1)
    parser.add_argument("--timeout_seconds", type=int, default=3600)
    parser.add_argument("--report_path", type=str, default="./taes_report.json")
    parser.add_argument("--beam_width", type=int, default=3)
    parser.add_argument("--branch_factor", type=int, default=3)
    parser.add_argument("--max_depth", type=int, default=4)
    parser.add_argument("--epsilon", type=float, default=0.1)
    parser.add_argument("--alpha_parallel", type=float, default=0.2)
    parser.add_argument("--budget_total", type=float, default=3000.0)
    parser.add_argument("--min_daily_budget", type=float, default=150.0)
    parser.add_argument("--search_only", action="store_true", help="Only run search, do not execute subprocess tasks.")
    parser.add_argument("--generator_model_name", type=str, default="", help="Model used by LLM action generator.")
    parser.add_argument("--generator_timeout", type=int, default=60, help="Timeout for each action-generation call.")

    # TimeArena config
    parser.add_argument("--ta_task_name", type=str, default="household1")
    parser.add_argument("--ta_prompting", choices=["react", "reflexion", "selfplan"], default="reflexion")
    parser.add_argument("--ta_total_time", type=int, default=60)
    parser.add_argument("--ta_save_path", type=str, default="./trajectory/taes")
    parser.add_argument("--ta_save_name", type=str, default="taes_timearena")
    parser.add_argument("--ta_lm", type=str, default="custom")
    parser.add_argument("--ta_model_name", type=str, default="")
    parser.add_argument("--ta_prompting_options", type=str, default="react,reflexion,selfplan")

    # TravelPlanner config
    parser.add_argument("--tp_set_type", type=str, default="validation")
    parser.add_argument("--tp_output_dir", type=str, default="./evaluation/validation")
    parser.add_argument("--tp_strategy", choices=["direct", "cot", "react", "reflexion"], default="direct")
    parser.add_argument("--tp_model_name", type=str, default="")
    parser.add_argument("--tp_strategy_options", type=str, default="direct,cot,react,reflexion")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = Path(__file__).resolve().parent
    profile = ConstraintProfile(
        max_retries=max(args.max_retries, 0),
        timeout_seconds=max(args.timeout_seconds, 1),
    )
    graph = build_graph(args)
    executor = TAESExecutor(root=root, profile=profile)
    search_env = executor._load_env()

    evaluator = MultiLevelEvaluator(
        epsilon=args.epsilon,
        alpha_parallel=args.alpha_parallel,
        min_daily_budget=max(1.0, args.min_daily_budget),
        max_time=max(1.0, float(args.timeout_seconds)),
    )
    action_generator = LLMActionGenerator(
        env=search_env,
        timeout_seconds=max(1, args.generator_timeout),
        model_name=args.generator_model_name,
    )
    planner = BeamSearchPlanner(
        tkg=graph,
        evaluator=evaluator,
        action_generator=action_generator,
        beam_width=args.beam_width,
        branch_factor=args.branch_factor,
        max_depth=args.max_depth,
        serial=args.serial,
    )
    best_state, tree = planner.search(budget_total=max(1.0, args.budget_total))

    results: List[Dict[str, str]] = []
    overall_success = True
    execution_plan = best_state.plan if best_state.plan else []
    if not args.search_only:
        if not execution_plan:
            # fallback when search does not produce actions
            for node in graph.plan_order():
                action = SearchAction(
                    task_id=node.task_id,
                    domain=node.domain,
                    payload=node.payload,
                    est_duration=0.0,
                    est_cost=0.0,
                    label=f"fallback:{node.domain}",
                )
                execution_plan.append(action)

        for action in execution_plan:
            node = graph.nodes[action.task_id]
            merged_payload = dict(node.payload)
            merged_payload.update(action.payload)
            runtime_node = TaskNode(
                task_id=node.task_id,
                domain=node.domain,
                earliest_start=node.earliest_start,
                latest_end=node.latest_end,
                duration_hint=node.duration_hint,
                depends_on=node.depends_on,
                payload=merged_payload,
            )
            result = executor.execute(runtime_node)
            result["search_label"] = action.label
            result["search_score_total"] = f"{best_state.score_total:.6f}"
            results.append(result)
            print(f"[TAES] {runtime_node.task_id} -> {result['status']} (code={result['return_code']})")
            if result["status"] != "success":
                overall_success = False
                if profile.stop_on_failure:
                    break

    report = {
        "task": args.task,
        "serial": args.serial,
        "search": {
            "beam_width": args.beam_width,
            "branch_factor": args.branch_factor,
            "max_depth": args.max_depth,
            "generator_enabled": action_generator.enabled,
            "generator_model": action_generator.model_name,
            "generator_last_error": action_generator.last_error,
            "best_state_id": best_state.state_id,
            "best_score_hard": best_state.score_hard,
            "best_score_soft": best_state.score_soft,
            "best_score_total": best_state.score_total,
            "best_plan": [
                {
                    "task_id": a.task_id,
                    "domain": a.domain,
                    "label": a.label,
                    "payload": a.payload,
                    "est_duration": a.est_duration,
                    "est_cost": a.est_cost,
                }
                for a in best_state.plan
            ],
            "tree_nodes": len(tree.nodes),
            "tree_edges": sum(len(v) for v in tree.children.values()),
        },
        "search_only": args.search_only,
        "overall_success": overall_success,
        "results": results,
    }
    report_path = Path(args.report_path)
    if not report_path.is_absolute():
        report_path = root / report_path
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[TAES] report saved: {report_path}")
    return 0 if overall_success else 1


if __name__ == "__main__":
    raise SystemExit(main())
