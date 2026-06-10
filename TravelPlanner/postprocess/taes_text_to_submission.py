import argparse
import json
import os
import re

from datasets import load_dataset


FIELDS = {
    "current city": "current_city",
    "transportation": "transportation",
    "breakfast": "breakfast",
    "attraction": "attraction",
    "lunch": "lunch",
    "dinner": "dinner",
    "accommodation": "accommodation",
}


def parse_plan_text(text):
    if not text:
        return None

    days = []
    current = None
    last_field = None
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        day_match = re.match(r"^Day\s+(\d+)\s*:\s*$", line, re.IGNORECASE)
        if day_match:
            if current is not None:
                days.append(current)
            current = {"days": int(day_match.group(1))}
            last_field = None
            continue
        if current is None:
            continue
        if ":" not in line:
            if last_field:
                current[last_field] = f"{current[last_field]}\n{line}"
            continue
        key, value = line.split(":", 1)
        mapped = FIELDS.get(key.strip().lower())
        if mapped:
            current[mapped] = value.strip()
            last_field = mapped

    if current is not None:
        days.append(current)

    if not days:
        return None

    for day in days:
        for field in FIELDS.values():
            day.setdefault(field, "-")
    return days


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--set_type", default="validation")
    parser.add_argument("--model_name", default="custom")
    parser.add_argument("--strategy", default="taes")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--submission_file", required=True)
    args = parser.parse_args()

    dataset = load_dataset("osunlp/TravelPlanner", args.set_type)[args.set_type]
    result_key = f"{args.model_name}_{args.strategy}_sole-planning_results"

    with open(args.submission_file, "w", encoding="utf-8") as out:
        for idx, query_data in enumerate(dataset, start=1):
            path = os.path.join(args.output_dir, args.set_type, f"generated_plan_{idx}.json")
            plan = None
            if os.path.exists(path):
                with open(path, encoding="utf-8") as f:
                    generated = json.load(f)
                plan = parse_plan_text(generated[-1].get(result_key, ""))
            unit = {"idx": idx, "query": query_data["query"], "plan": plan}
            out.write(json.dumps(unit, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
