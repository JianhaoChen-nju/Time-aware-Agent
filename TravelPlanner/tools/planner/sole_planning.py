import os
import re
import sys
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "../..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../..")))
os.chdir(os.path.dirname(os.path.abspath(__file__)))
from agents.prompts import planner_agent_prompt, cot_planner_agent_prompt, react_planner_agent_prompt,react_reflect_planner_agent_prompt,reflect_prompt
# from utils.func import get_valid_name_city,extract_before_parenthesis, extract_numbers_from_filenames
import json
import signal
import time
from langchain.callbacks import get_openai_callback

from tqdm import tqdm
from tools.planner.apis import Planner, ReactPlanner, ReactReflectPlanner
from src.taes.travelplanner import TAESPlanner
import openai
import argparse
from datasets import load_dataset




def load_line_json_data(filename):
    data = []
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f.read().strip().split('\n'):
            unit = json.loads(line)
            data.append(unit)
    return data

def extract_numbers_from_filenames(directory):
    # Define the pattern to match files
    pattern = r'annotation_(\d+).json'

    # List all files in the directory
    files = os.listdir(directory)

    # Extract numbers from filenames that match the pattern
    numbers = [int(re.search(pattern, file).group(1)) for file in files if re.match(pattern, file)]

    return numbers


def catch_openai_api_error():
    error = sys.exc_info()[0]
    if error == openai.APIConnectionError:
        print("APIConnectionError")
    elif error == openai.RateLimitError:
        print("RateLimitError")
        time.sleep(60)
    elif error == openai.APIError:
        print("APIError")
    elif error == openai.AuthenticationError:
        print("AuthenticationError")
    else:
        print("API error:", error)


class AttemptTimeout(Exception):
    pass


def _timeout_handler(_signum, _frame):
    raise AttemptTimeout()


def run_with_timeout(timeout_seconds, fn):
    if not timeout_seconds or timeout_seconds <= 0:
        return fn()
    old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
    signal.alarm(timeout_seconds)
    try:
        return fn()
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)


def is_empty_planner_result(value):
    if value is None:
        return True
    if isinstance(value, str):
        return not value.strip()
    if isinstance(value, (list, dict, tuple, set)):
        return len(value) == 0
    return False


if __name__ == "__main__":

    # model_name= ['gpt-3.5-turbo-1106','gpt-4-1106-preview','gemini','mixtral'][1]
    # set_type = ['dev','test'][0]
    # strategy = ['direct','cot','react','reflexion'][0]

    parser = argparse.ArgumentParser()
    parser.add_argument("--set_type", type=str, default="validation")
    parser.add_argument("--model_name", type=str, default="gpt-3.5-turbo-1106")
    parser.add_argument("--output_dir", type=str, default="./")
    parser.add_argument("--strategy", type=str, default="direct")
    parser.add_argument("--start_idx", type=int, default=1)
    parser.add_argument("--end_idx", type=int, default=None)
    parser.add_argument("--skip_existing", action="store_true")
    parser.add_argument("--max_attempts", type=int, default=3)
    parser.add_argument("--attempt_timeout", type=int, default=0)
    parser.add_argument("--beam_width", type=int, default=5)
    parser.add_argument("--branch_factor", type=int, default=3)
    parser.add_argument("--disable_taes_direct_fallback", action="store_true")
    args = parser.parse_args()
    directory = f'{args.output_dir}/{args.set_type}'
    if args.set_type == 'train':
        query_data_list  = load_dataset('osunlp/TravelPlanner','train')['train']
    elif args.set_type == 'validation':
        query_data_list  = load_dataset('osunlp/TravelPlanner','validation')['validation']
    elif args.set_type == 'test':
        query_data_list  = load_dataset('osunlp/TravelPlanner','test')['test']
    end_idx = args.end_idx or len(query_data_list)
    numbers = [i for i in range(args.start_idx, min(end_idx, len(query_data_list)) + 1)]

    if args.strategy == 'direct':
        planner = Planner(model_name=args.model_name, agent_prompt=planner_agent_prompt)
    elif args.strategy == 'cot':
        planner = Planner(model_name=args.model_name, agent_prompt=cot_planner_agent_prompt)
    elif args.strategy == 'react':
        planner = ReactPlanner(model_name=args.model_name, agent_prompt=react_planner_agent_prompt)
    elif args.strategy == 'reflexion':
        planner = ReactReflectPlanner(model_name=args.model_name, agent_prompt=react_reflect_planner_agent_prompt,reflect_prompt=reflect_prompt)
    elif args.strategy == 'taes':
        planner = TAESPlanner(model_name=args.model_name, B=args.beam_width, K=args.branch_factor)
        direct_fallback_planner = None
        if not args.disable_taes_direct_fallback:
            direct_fallback_planner = Planner(model_name=args.model_name, agent_prompt=planner_agent_prompt)


    with get_openai_callback() as cb:
        for number in tqdm(numbers[:]):
            
            query_data = query_data_list[number-1]
            reference_information = query_data['reference_information']
            output_file = os.path.join(f'{args.output_dir}/{args.set_type}/generated_plan_{number}.json')
            result_key = f'{args.model_name}_{args.strategy}_sole-planning_results'

            if args.skip_existing and os.path.exists(output_file):
                try:
                    existing_result = json.load(open(output_file))
                    if existing_result and existing_result[-1].get(result_key):
                        continue
                except Exception:
                    pass

            attempts = 0
            used_taes_direct_fallback = False
            while True:
                    attempts += 1
                    try:
                        if args.strategy == 'taes':
                            planner_results = run_with_timeout(
                                args.attempt_timeout,
                                lambda: planner.run(reference_information, query_data['query'], query_data),
                            )
                        elif args.strategy in ['react','reflexion']:
                            planner_results, scratchpad = run_with_timeout(
                                args.attempt_timeout,
                                lambda: planner.run(reference_information, query_data['query']),
                            )
                        else:
                            planner_results = run_with_timeout(
                                args.attempt_timeout,
                                lambda: planner.run(reference_information, query_data['query']),
                            )
                    except AttemptTimeout:
                        print(f"Planner attempt timed out for query {number} after {args.attempt_timeout}s.")
                        planner_results = None
                    if not is_empty_planner_result(planner_results):
                        break
                    if attempts >= args.max_attempts:
                        if args.strategy == 'taes' and direct_fallback_planner is not None:
                            print(f"TAES returned empty result for query {number} after {attempts} attempts; running direct fallback.")
                            try:
                                planner_results = run_with_timeout(
                                    args.attempt_timeout,
                                    lambda: direct_fallback_planner.run(reference_information, query_data['query']),
                                )
                            except AttemptTimeout:
                                print(f"Direct fallback timed out for query {number} after {args.attempt_timeout}s.")
                                planner_results = None
                            if not is_empty_planner_result(planner_results):
                                used_taes_direct_fallback = True
                                break
                            print(f"Direct fallback returned empty for query {number}; writing empty result.")
                        else:
                            print(f"Planner returned empty result for query {number} after {attempts} attempts; writing empty result.")
                        planner_results = ""
                        break
            print(planner_results)
            # check if the directory exists
            if not os.path.exists(os.path.join(f'{args.output_dir}/{args.set_type}')):
                os.makedirs(os.path.join(f'{args.output_dir}/{args.set_type}'))
            if not os.path.exists(output_file):
                result =  [{}]
            else:
                result = json.load(open(output_file))
            if args.strategy in ['react','reflexion']:
                result[-1][f'{args.model_name}_{args.strategy}_sole-planning_results_logs'] = scratchpad 
            result[-1][result_key] = planner_results
            if args.strategy == 'taes':
                result[-1][f'{args.model_name}_{args.strategy}_direct_fallback_used'] = used_taes_direct_fallback
                if used_taes_direct_fallback:
                    result[-1][f'{args.model_name}_{args.strategy}_fallback'] = 'direct'
            # write to json file
            with open(output_file, 'w') as f:
                json.dump(result, f, indent=4)
        print(cb)
