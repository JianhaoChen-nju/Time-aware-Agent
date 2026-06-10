import argparse
import ast
import json
import os
import re
import sys

from datasets import load_dataset

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "TravelPlanner"))

from src.taes.travelplanner.checker import check_all  # noqa: E402
from src.taes.travelplanner.costs import compute_day_cost  # noqa: E402
from src.taes.travelplanner.state import TravelPlannerEvent, TravelPlannerState  # noqa: E402


def _destination_for_day(current_city, fallback):
    match = re.search(r"from\s+(.+?)\s+to\s+([^,]+)(?=[,\s]|$)", current_city or "")
    if match:
        return match.group(2).strip()
    return current_city.strip() if current_city else fallback


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--set_type", default="validation")
    parser.add_argument("--submission", required=True)
    parser.add_argument("--indices", required=True)
    args = parser.parse_args()

    dataset = list(load_dataset("osunlp/TravelPlanner", args.set_type)[args.set_type])
    predictions = [json.loads(line) for line in open(args.submission, encoding="utf-8")]
    indices = [int(x) for x in args.indices.split(",") if x.strip()]

    for idx in indices:
        query_data = dict(dataset[idx - 1])
        if isinstance(query_data.get("local_constraint"), str):
            query_data["local_constraint"] = ast.literal_eval(query_data["local_constraint"])
        state = TravelPlannerState.from_query_data(query_data)
        rejected_day = None
        plan = predictions[idx - 1].get("plan")
        if not plan:
            print(idx, "EMPTY", query_data["level"], query_data["days"], query_data["local_constraint"])
            continue
        for day_plan in plan:
            cost = compute_day_cost(day_plan, query_data["people_number"])
            event = TravelPlannerEvent(
                day_plan=day_plan,
                estimated_cost=cost,
                destination_city=_destination_for_day(day_plan.get("current_city", ""), state.current_city),
            )
            if not check_all(state, event):
                rejected_day = day_plan.get("days")
                break
            state = event.apply(state)
        verdict = "ACCEPT" if rejected_day is None else f"REJECT day {rejected_day}"
        print(idx, verdict, query_data["level"], query_data["days"], query_data["local_constraint"])


if __name__ == "__main__":
    main()
