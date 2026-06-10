import argparse
import ast
import collections
import json
import os
import sys

from datasets import load_dataset

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "TravelPlanner"))

from src.taes.travelplanner.checker import explain_failures  # noqa: E402
from src.taes.travelplanner.evaluator import evaluate_state  # noqa: E402
from src.taes.travelplanner.planner import TAESPlanner  # noqa: E402
from src.taes.travelplanner.state import TravelPlannerState  # noqa: E402
from src.taes.beam_search import _select_beam  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--set_type", default="validation")
    parser.add_argument("--idx", type=int, required=True)
    parser.add_argument("--beam_width", type=int, default=5)
    parser.add_argument("--branch_factor", type=int, default=3)
    parser.add_argument("--summary_only", action="store_true")
    parser.add_argument("--empty_details", action="store_true")
    args = parser.parse_args()

    dataset = load_dataset("osunlp/TravelPlanner", args.set_type)[args.set_type]
    query_data = dict(dataset[args.idx - 1])
    if isinstance(query_data.get("local_constraint"), str):
        query_data["local_constraint"] = ast.literal_eval(query_data["local_constraint"])

    planner = TAESPlanner(model_name="custom", B=args.beam_width, K=args.branch_factor)
    initial_state = TravelPlannerState.from_query_data(query_data)
    beam = [(evaluate_state(initial_state), initial_state)]
    trace = []

    for depth in range(query_data["days"]):
        candidates = []
        depth_info = {
            "depth": depth + 1,
            "beam_size": len(beam),
            "states": [],
            "failure_counts": collections.Counter(),
        }
        for score, state in beam:
            events = planner._generate_candidates(state, query_data["query"], query_data["reference_information"])
            state_info = {
                "score": score,
                "current_day": state.current_day,
                "current_city": state.current_city,
                "planned_days": len(state.daily_plans),
                "generated": len(events),
                "accepted": 0,
                "accepted_plans": [],
                "rejected": [],
            }
            for event in events:
                failures = explain_failures(state, event)
                if failures:
                    state_info["rejected"].append({"day_plan": event.day_plan, "failures": failures})
                    depth_info["failure_counts"].update(failures)
                    continue
                new_state = event.apply(state.copy())
                candidates.append((evaluate_state(new_state), new_state))
                state_info["accepted"] += 1
                state_info["accepted_plans"].append(
                    {
                        "day_plan": event.day_plan,
                        "estimated_cost": event.estimated_cost,
                        "next_budget_remaining": new_state.budget_remaining,
                        "next_score": evaluate_state(new_state),
                    }
                )
            depth_info["states"].append(state_info)
        candidates.sort(key=lambda x: x[0], reverse=True)
        beam = _select_beam(
            candidates,
            args.beam_width,
            diversity_key=lambda state: state.beam_diversity_key(),
        )
        depth_info["next_beam_size"] = len(beam)
        depth_info["failure_counts"] = dict(depth_info["failure_counts"])
        trace.append(depth_info)
        if args.summary_only:
            top_failures = sorted(
                depth_info["failure_counts"].items(),
                key=lambda item: item[1],
                reverse=True,
            )[:5]
            print(
                json.dumps(
                    {
                        "depth": depth_info["depth"],
                        "beam_size": depth_info["beam_size"],
                        "accepted": sum(s["accepted"] for s in depth_info["states"]),
                        "next_beam_size": depth_info["next_beam_size"],
                        "top_failures": top_failures,
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
            if args.empty_details and depth_info["next_beam_size"] == 0:
                print(json.dumps({"empty_depth_details": depth_info}, ensure_ascii=False, indent=2), flush=True)
        if not beam:
            break

    if not args.summary_only:
        print(json.dumps(trace, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
