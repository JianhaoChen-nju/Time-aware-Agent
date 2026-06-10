import matplotlib.pyplot as plt
import json
import glob
import pdb
import argparse
import os
import re


def Completion_Rate(results):
    n = 0
    for task in results:
        if task["completed"] or task["score"] == 100:
            n += 1
    return round(n / len(results)*100,2)

def Average_Completion_Time(results):
    n = 0
    num = 0
    for task in results:
        if task["completed"] or task["score"] == 100:
            n += task["time"]
            num += 1
    if num == 0:
        return 0
    return round(n / num, 2)

def Average_Progress_Score(results):
    n = 0
    for task in results:
        n += task["score"]
    return round(n / len(results), 2)

def Completion_Speed(results):
    n = 0
    time = 0
    for task in results:
        n += task["score"]
        time += task["time"]
    return round(n / time, 2)


def _is_timearena_63_case(file_path):
    """Keep the last 7 cases for each single/double/triple TimeArena group."""
    stem = os.path.splitext(os.path.basename(file_path))[0]
    match = re.match(r"^(cooking|household|lab)(\d+)(?:_|$)", stem)
    if not match:
        return False
    first_idx = int(match.group(2))
    return 4 <= first_idx <= 10


def cal_metrics(path, timearena_63=False):
    models_result = []
    for file_path in glob.glob(f"{path}/*"):
        if timearena_63 and not _is_timearena_63_case(file_path):
            continue
        with open(file_path, 'r') as f:
            data = json.load(f)
        i = 0
        # for index, item in enumerate(data):
        #     if type(item) == dict:
        #         i = index
        #         break
        Time_score = {}
        for index, item in enumerate(data):
            if type(item)==dict and 'time' in item.keys() and 'progress score' in item.keys():
                Time_score[item['time']] = item['progress score']
        scores = [v for k,v in Time_score.items()]
        summary = data[-1] if data and isinstance(data[-1], dict) else {}
        if 'total_score' in summary and 'total_time' in summary:
            models_result.append({
                "score": summary.get('total_score', 0),
                "time": summary.get('total_time', 0),
                "completed": bool(summary.get('isCompleted', False)),
            })
        elif scores:
            final_score = scores[-1]
            models_result.append({
                "score": final_score,
                "time": scores.index(final_score) + 1,
                "completed": final_score == 100,
            })
    print(len(models_result))
    if not models_result:
        return 0, 0, 0, 0
    return Average_Progress_Score(models_result), Completion_Speed(models_result), Completion_Rate(models_result), Average_Completion_Time(models_result)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, help='Trajectory folder path')
    parser.add_argument("--timearena_63", action="store_true",
                        help="Only count TimeArena cases whose first task index is 4..10.")
    args = parser.parse_args()
    AS, CS, CR, CT = cal_metrics(args.path, timearena_63=args.timearena_63)
    print(f"AS: {AS} ; CS: {CS} ; CR : {CR} ; CT: {CT}")
