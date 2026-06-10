import argparse
import json
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "..")))
from commonsense_constraint import evaluation as commonsense_eval
from datasets import load_dataset
from hard_constraint import evaluation as hard_eval


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--set_type", default="validation")
    parser.add_argument("--evaluation_file_path", required=True)
    args = parser.parse_args()

    dataset = list(load_dataset("osunlp/TravelPlanner", args.set_type)[args.set_type])
    predictions = load_jsonl(args.evaluation_file_path)

    empty = []
    failed = []
    passed = []
    for idx, (query_data, pred) in enumerate(zip(dataset, predictions), start=1):
        if isinstance(query_data.get("local_constraint"), str):
            query_data["local_constraint"] = eval(query_data["local_constraint"])
        plan = pred.get("plan")
        if not plan:
            empty.append(
                {
                    "idx": idx,
                    "level": query_data["level"],
                    "days": query_data["days"],
                    "query": query_data["query"],
                }
            )
            continue

        commonsense = commonsense_eval(query_data, plan)
        hard = None
        if commonsense and commonsense["is_not_absent"][0] and commonsense["is_valid_information_in_sandbox"][0]:
            hard = hard_eval(query_data, plan)

        commonsense_failures = {
            key: value[1]
            for key, value in (commonsense or {}).items()
            if value[0] is not None and value[0] is False
        }
        hard_failures = {
            key: value[1]
            for key, value in (hard or {}).items()
            if value[0] is not None and value[0] is False
        }
        record = {
            "idx": idx,
            "level": query_data["level"],
            "days": query_data["days"],
            "query": query_data["query"],
            "commonsense_failures": commonsense_failures,
            "hard_failures": hard_failures,
        }
        if commonsense_failures or hard_failures or hard is None:
            failed.append(record)
        else:
            passed.append(record)

    by_bucket = {}
    for item in empty:
        key = f"{item['level']}-{item['days']}"
        by_bucket[key] = by_bucket.get(key, 0) + 1

    print(json.dumps(
        {
            "total": len(predictions),
            "passed": len(passed),
            "empty": len(empty),
            "non_empty_failed": len(failed),
            "empty_by_bucket": by_bucket,
            "empty_indices": [x["idx"] for x in empty],
            "passed_indices": [x["idx"] for x in passed],
            "failed": failed,
        },
        ensure_ascii=False,
        indent=2,
    ))


if __name__ == "__main__":
    main()
