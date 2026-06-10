import argparse
import ast
import json
import os
import sys

from datasets import load_dataset

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "TravelPlanner"))

from src.taes.travelplanner.checker import explain_failures  # noqa: E402
from src.taes.travelplanner.planner import TAESPlanner  # noqa: E402
from src.taes.travelplanner.state import TravelPlannerState  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--set_type", default="validation")
    parser.add_argument("--idx", type=int, required=True)
    parser.add_argument("--beam_width", type=int, default=5)
    parser.add_argument("--branch_factor", type=int, default=5)
    args = parser.parse_args()

    dataset = load_dataset("osunlp/TravelPlanner", args.set_type)[args.set_type]
    query_data = dict(dataset[args.idx - 1])
    if isinstance(query_data.get("local_constraint"), str):
        query_data["local_constraint"] = ast.literal_eval(query_data["local_constraint"])

    planner = TAESPlanner(model_name="custom", B=args.beam_width, K=args.branch_factor)
    state = TravelPlannerState.from_query_data(query_data)
    events = planner._generate_candidates(state, query_data["query"], query_data["reference_information"])
    output = []
    for event in events:
        output.append(
            {
                "day_plan": event.day_plan,
                "estimated_cost": event.estimated_cost,
                "destination_city": event.destination_city,
                "failures": explain_failures(state, event),
            }
        )
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
