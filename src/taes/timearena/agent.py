"""TAES Agent for TimeArena.

Plans ahead via beam search on an internal state model, then executes
actions one at a time, re-planning when the environment state changes.
"""

import logging
from typing import List, Optional

from ..llm_interface import get_llm, call_llm, parse_json_from_response
from .state import TimeArenaState, TimeArenaEvent
from .checker import check_all
from .evaluator import evaluate_state
from .prompts import build_action_prompt, build_trajectory_prompt

logger = logging.getLogger(__name__)


class TAES_Agent:
    """TAES-based agent for TimeArena."""

    def __init__(
        self,
        B: int = 3,
        K: int = 3,
        plan_depth: int = 8,
        use_llm_candidates: bool = True,
        llm_root_only: bool = False,
        llm_top_state_per_depth: bool = False,
        proposal_horizon: int = 4,
    ):
        self.B = B
        self.K = K
        self.plan_depth = plan_depth
        self.use_llm_candidates = use_llm_candidates
        self.llm_root_only = llm_root_only
        self.llm_top_state_per_depth = llm_top_state_per_depth
        self.proposal_horizon = proposal_horizon
        self.name = 'taes'
        self.acting = "\nIn t={0}, your action is:"
        self.llm = get_llm() if use_llm_candidates else None
        self.planned_actions: List[str] = []
        self.internal_state: Optional[TimeArenaState] = None
        self.env_ref = None
        self._planning_root_time = 0
        candidate_mode = f'llm-trajectory-H{proposal_horizon}' if use_llm_candidates else 'heuristic'
        if use_llm_candidates and llm_root_only:
            candidate_mode = f'llm-root-only-H{proposal_horizon}'
        elif use_llm_candidates and llm_top_state_per_depth:
            candidate_mode = f'llm-top-state-per-depth-H{proposal_horizon}'
        print(f"TAES_Agent loaded (B={B}, K={K}, depth={plan_depth}, candidates={candidate_mode})")

    def initialize_state(self, env, total_time: int):
        self.env_ref = env
        self.internal_state = TimeArenaState.from_environment(env, t_now=0, total_time=total_time)

    def act(self, message, calling, current_time, historyRecording):
        if self.env_ref:
            self.internal_state = TimeArenaState.from_environment(
                self.env_ref, t_now=current_time, total_time=self.internal_state.total_time
            )

        # Try to use existing plan
        if self.planned_actions:
            action = self.planned_actions.pop(0)
            available = self.internal_state.get_available_actions()
            if action in available:
                # Don't blindly follow planned waits when productive actions exist
                if action == 'wait':
                    non_wait = [a for a in available if a != 'wait']
                    if non_wait:
                        self.planned_actions = []  # discard stale plan, re-plan below
                    else:
                        historyRecording.append({
                            'time': current_time, 'taes_action': action, 'from_plan': True,
                        })
                        return message, action, historyRecording
                else:
                    historyRecording.append({
                        'time': current_time, 'taes_action': action, 'from_plan': True,
                    })
                    return message, action, historyRecording

        # Re-plan
        self.planned_actions = self._plan(current_time)

        if self.planned_actions:
            action = self.planned_actions.pop(0)
        else:
            available = self.internal_state.get_available_actions()
            non_wait = [a for a in available if a != 'wait']
            action = non_wait[0] if non_wait else 'wait'

        historyRecording.append({
            'time': current_time, 'taes_action': action, 'from_plan': False,
            'plan_length': len(self.planned_actions) + 1,
        })
        return message, action, historyRecording

    def step(self, action, calling, env, historyRecording, message, current_time, score):
        observation, increment, isCompleted, agent_occupy, wrong_message = env.step(action)

        if wrong_message:
            self.planned_actions = []

        if observation is not None:
            historyRecording.append({
                'time': current_time, 'observation': observation,
                'increment': increment, 'progress score': score,
                'isCompleted': isCompleted, 'action': action,
            })

        return observation, increment, isCompleted, agent_occupy, wrong_message, historyRecording, message

    def _plan(self, current_time: int) -> List[str]:
        if self.internal_state is None or self.internal_state.is_terminal():
            return []

        self._planning_root_time = current_time
        max_depth = min(self.plan_depth, max(1, self.internal_state.total_time - current_time))
        best_state = self._beam_search_trajectory_llm(self.internal_state, max_depth)

        if best_state is None:
            return []

        actions = []
        for action_str, timestep in best_state.completed_actions:
            if timestep >= current_time:
                actions.append(action_str)

        if not actions:
            available = self.internal_state.get_available_actions()
            if available:
                actions = [available[0]]

        return actions

    def _beam_search_trajectory_llm(self, initial_state: TimeArenaState, max_depth: int) -> Optional[TimeArenaState]:
        beam = [(evaluate_state(initial_state), initial_state)]
        finished = []
        depth = 0

        while depth < max_depth:
            if not beam:
                break

            beam.sort(key=lambda item: item[0], reverse=True)
            candidates = []
            step_budget = min(self.proposal_horizon, max_depth - depth)

            for rank, (_score, state) in enumerate(beam):
                if state.is_terminal():
                    finished.append((evaluate_state(state), state))
                    continue

                allow_llm = self._allow_llm_for_state(state, rank)
                trajectories = self._generate_trajectories(state, horizon=step_budget, allow_llm=allow_llm)
                if not trajectories:
                    finished.append((evaluate_state(state), state))
                    continue

                for trajectory in trajectories:
                    new_state = self._apply_valid_prefix(state, trajectory)
                    if new_state is not None and new_state is not state:
                        candidates.append((evaluate_state(new_state), new_state))

            if not candidates:
                break

            candidates.sort(key=lambda item: item[0], reverse=True)
            beam = candidates[:self.B]
            depth += step_budget

        finished.extend((evaluate_state(state), state) for _score, state in beam)
        if not finished:
            return None
        finished.sort(key=lambda item: item[0], reverse=True)
        return finished[0][1]

    def _apply_valid_prefix(self, state: TimeArenaState, trajectory: List) -> Optional[TimeArenaState]:
        current = state
        applied = 0
        for item in trajectory:
            event = item
            if isinstance(item, str):
                event = self._match_action_to_event(item, current)
                if event is None:
                    break
            if current.is_terminal() or not check_all(current, event):
                break
            current = event.apply(current.copy())
            applied += 1
        if applied == 0:
            return None
        return current

    def _allow_llm_for_state(self, state: TimeArenaState, beam_rank: int) -> bool:
        if not self.use_llm_candidates:
            return False
        if self.llm_root_only:
            return state.t_now == self._planning_root_time
        if self.llm_top_state_per_depth:
            return beam_rank == 0
        return True

    def _generate_candidates(self, state: TimeArenaState, allow_llm: bool = True) -> List[TimeArenaEvent]:
        available_actions = state.get_available_actions()

        if len(available_actions) <= 1:
            if 'wait' in available_actions:
                return [TimeArenaEvent('wait', '', '', 1, False)]
            return []

        heuristic_events = self._heuristic_events(available_actions, state, limit=self.K)

        if len(available_actions) <= self.K:
            events = [self._action_str_to_event(a, state) for a in available_actions if a != 'wait']
            # Include wait when background tasks are running — they may need to
            # finish before dependent actions become valid
            if state.running_tasks:
                events.append(TimeArenaEvent('wait', '', '', 1, False))
            return events

        if not allow_llm:
            return heuristic_events

        prompt = build_action_prompt(state, K=self.K)
        try:
            response = call_llm(self.llm, prompt)
            candidates = parse_json_from_response(response)

            if not candidates or not isinstance(candidates, list):
                return heuristic_events

            events = list(heuristic_events)
            seen = {event.action_string for event in events}
            for candidate in candidates[:self.K]:
                if isinstance(candidate, str):
                    action_str = candidate.strip()
                    if action_str in available_actions or action_str == 'wait':
                        if action_str not in seen:
                            events.append(self._action_str_to_event(action_str, state))
                            seen.add(action_str)
                    else:
                        for avail in available_actions:
                            if action_str.lower() == avail.lower():
                                if avail not in seen:
                                    events.append(self._action_str_to_event(avail, state))
                                    seen.add(avail)
                                break

            if not events:
                return heuristic_events
            return events

        except Exception as e:
            logger.error(f"LLM candidate generation failed: {e}")
            return heuristic_events

    def _generate_trajectories(
        self, state: TimeArenaState, horizon: int, allow_llm: bool = True
    ) -> List[List[TimeArenaEvent]]:
        if not allow_llm:
            return self._heuristic_trajectories(state, horizon=horizon)

        available_actions = state.get_available_actions()
        if len(available_actions) <= 1:
            if 'wait' in available_actions:
                return [[TimeArenaEvent('wait', '', '', 1, False)]]
            return []

        prompt = build_trajectory_prompt(state, K=self.K, H=horizon)
        candidate_sequences = self._call_llm_for_trajectories(prompt)

        trajectories = []
        for candidate_sequences_for_attempt in (candidate_sequences,):
            trajectories = self._parse_llm_trajectories(candidate_sequences_for_attempt, state, horizon)
            if trajectories:
                return trajectories

        retry_prompt = prompt + "\nReturn valid JSON only. Every first action must exactly match AVAILABLE FIRST ACTIONS."
        retry_sequences = self._call_llm_for_trajectories(retry_prompt)
        return self._parse_llm_trajectories(retry_sequences, state, horizon)

    def _call_llm_for_trajectories(self, prompt: str):
        try:
            response = call_llm(self.llm, prompt)
            return parse_json_from_response(response)
        except Exception as e:
            logger.error(f"LLM trajectory generation failed: {e}")
            return None

    def _parse_llm_trajectories(self, candidate_sequences, state: TimeArenaState, horizon: int) -> List[List[str]]:
        trajectories = []
        if not isinstance(candidate_sequences, list):
            return trajectories

        for sequence in candidate_sequences[:self.K]:
            if isinstance(sequence, str):
                sequence = [sequence]
            if not isinstance(sequence, list):
                continue
            actions = []
            for action in sequence[:horizon]:
                if not isinstance(action, str):
                    continue
                actions.append(action.strip())
            if actions and self._match_action_to_event(actions[0], state):
                trajectories.append(actions)
        return trajectories

    def _heuristic_trajectories(self, state: TimeArenaState, horizon: int) -> List[List[TimeArenaEvent]]:
        trajectories = []
        first_events = self._heuristic_events(state.get_available_actions(), state, limit=self.K)
        for first_event in first_events:
            current = state
            trajectory = []
            for step in range(horizon):
                if step == 0:
                    event = first_event
                else:
                    next_events = self._heuristic_events(current.get_available_actions(), current, limit=1)
                    if not next_events:
                        break
                    event = next_events[0]
                if not check_all(current, event):
                    break
                trajectory.append(event)
                current = event.apply(current.copy())
            if trajectory:
                trajectories.append(trajectory)
        return trajectories

    def _match_action_to_event(self, action_str: str, state: TimeArenaState) -> Optional[TimeArenaEvent]:
        if action_str == 'wait':
            return TimeArenaEvent('wait', '', '', 1, False)
        available_actions = state.get_available_actions()
        if action_str in available_actions:
            return self._action_str_to_event(action_str, state)
        for available in available_actions:
            if action_str.lower() == available.lower():
                return self._action_str_to_event(available, state)
        return None

    def _heuristic_events(self, available_actions: List[str], state: TimeArenaState, limit: int) -> List[TimeArenaEvent]:
        non_wait = [action for action in available_actions if action != 'wait']
        ranked = sorted(non_wait, key=lambda action: self._action_priority(action, state), reverse=True)
        events = [self._action_str_to_event(action, state) for action in ranked[:limit]]
        if state.running_tasks and not non_wait:
            events.append(TimeArenaEvent('wait', '', '', 1, False))
        return events

    def _action_priority(self, action_str: str, state: TimeArenaState) -> float:
        event = self._action_str_to_event(action_str, state)
        if event.action_string == 'wait':
            return -1.0

        score = 0.0
        if not event.is_blocking:
            score += 100.0 + event.duration
        else:
            score += max(0.0, 10.0 - event.duration)

        score += 20.0 * self._unlock_count(state, event)

        if event.action_key.startswith('bake_in'):
            score += 70.0
        elif event.action_key.startswith('add_to'):
            score += 45.0
        elif event.action_key.startswith(('cook_in', 'fry_in', 'heat')):
            score += 10.0
        elif event.action_key == 'wash':
            todos = state.object_todo.get(event.obj_name, {})
            if 'add_to_2' in todos or 'bake_in_1' in todos:
                score += 45.0

        return score

    def _unlock_count(self, state: TimeArenaState, event: TimeArenaEvent) -> int:
        produced = [(event.obj_name, event.action_key)]
        if event.secondary_obj:
            produced.append((event.secondary_obj, event.action_key.replace('_1', '_2')))

        count = 0
        for obj_name, deps in state.object_dependencies.items():
            for action_key, dep in deps.items():
                if state.object_todo.get(obj_name, {}).get(action_key, 0) <= 0:
                    continue
                if isinstance(dep, str):
                    if (obj_name, dep) in produced:
                        count += 1
                elif isinstance(dep, dict):
                    for dep_obj, dep_action in dep.items():
                        if (dep_obj, dep_action) in produced:
                            count += 1
                            break
        return count

    def _action_str_to_event(self, action_str: str, state: TimeArenaState) -> TimeArenaEvent:
        if action_str == 'wait':
            return TimeArenaEvent('wait', '', '', 1, False)

        parts = action_str.split()
        if len(parts) == 2:
            action_name, obj_name = parts[0], parts[1]
            action_key = action_name
            secondary_obj = None
        elif len(parts) == 4:
            action_name = f"{parts[0]}_{parts[2]}"
            obj_name = parts[1]
            secondary_obj = parts[3]
            action_key = f"{action_name}_1"
        else:
            return TimeArenaEvent(action_str, '', '', 1, False)

        duration = state.object_todo.get(obj_name, {}).get(action_key, 1)
        # Look up blocking status: try full action name first, then first word
        if len(parts) == 4:
            occupy_key = f"{parts[0]} {parts[2]}"
        else:
            occupy_key = parts[0]
        is_blocking = state.action_occupy.get(occupy_key, state.action_occupy.get(parts[0], True))

        return TimeArenaEvent(action_str, obj_name, action_key, duration, is_blocking, secondary_obj)
