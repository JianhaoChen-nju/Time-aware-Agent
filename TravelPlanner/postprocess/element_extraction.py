import argparse
from datasets import load_dataset
from tqdm import tqdm
import json

def parse_plan_result(raw):
    if '```json' in raw:
        result = raw.split('```json')[1].split('```')[0]
    elif '```' in raw:
        result = raw.split('```')[1].split('```')[0]
    else:
        result = raw.split('\t', 1)[1] if '\t' in raw else raw
    result = result.strip()

    try:
        return json.loads(result)
    except Exception:
        pass

    if '{{' in result or '}}' in result:
        cleaned = result.replace('{{', '{').replace('}}', '}')
        try:
            return json.loads(cleaned)
        except Exception:
            result = cleaned

    # Some models emit consecutive JSON objects instead of a JSON array.
    decoder = json.JSONDecoder()
    values = []
    pos = 0
    while pos < len(result):
        while pos < len(result) and (result[pos].isspace() or result[pos] == ','):
            pos += 1
        if pos >= len(result):
            break
        value, end = decoder.raw_decode(result, pos)
        values.append(value)
        pos = end
    if values:
        return values

    return eval(result)

def normalize_plan(parsed_result):
    if isinstance(parsed_result, dict):
        if isinstance(parsed_result.get('items'), list):
            return parsed_result['items']
        day_keys = [key for key in parsed_result if str(key).lower().startswith('day')]
        if day_keys:
            return [parsed_result[key] for key in sorted(day_keys, key=lambda x: int(''.join(ch for ch in str(x) if ch.isdigit()) or 0))]
        return [parsed_result]
    return parsed_result


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--set_type", type=str, default="validation")
    parser.add_argument("--model_name", type=str, default="gpt-3.5-turbo-1106")
    parser.add_argument("--mode", type=str, default="two-stage")
    parser.add_argument("--strategy", type=str, default="direct")
    parser.add_argument("--output_dir", type=str, default="./")
    parser.add_argument("--tmp_dir", type=str, default="./")

    args = parser.parse_args()

    if args.mode == 'two-stage':
        suffix = ''
    elif args.mode == 'sole-planning':
        suffix = f'_{args.strategy}'


    results = open(f'{args.tmp_dir}/{args.set_type}_{args.model_name}{suffix}_{args.mode}.txt','r').read().strip().split('\n')
    
    if args.set_type == 'train':
        query_data_list  = load_dataset('osunlp/TravelPlanner','train')['train']
    elif args.set_type == 'validation':
        query_data_list  = load_dataset('osunlp/TravelPlanner','validation')['validation']
    elif args.set_type == 'test':
        query_data_list  = load_dataset('osunlp/TravelPlanner','test')['test']

    idx_number_list = [i for i in range(1,len(query_data_list)+1)]
    for idx in tqdm(idx_number_list[:]):
        generated_plan = json.load(open(f'{args.output_dir}/{args.set_type}/generated_plan_{idx}.json'))
        if generated_plan[-1][f'{args.model_name}{suffix}_{args.mode}_results'] not in ["","Max Token Length Exceeded."] :
            try:
                raw = results[idx-1]
                parsed_result = parse_plan_result(raw)
            except:
                print(f"{idx}:\n{results[idx-1]}\nThis plan cannot be parsed.")
                generated_plan[-1][f'{args.model_name}{suffix}_{args.mode}_parsed_results'] = None
                with open(f'{args.output_dir}/{args.set_type}/generated_plan_{idx}.json','w') as f:
                    json.dump(generated_plan,f)
                continue
            try:
                if args.mode == 'two-stage':
                    parsed_result = normalize_plan(parsed_result)
                    generated_plan[-1][f'{args.model_name}{suffix}_{args.mode}_parsed_results'] = parsed_result
                else:
                    parsed_result = normalize_plan(parsed_result)
                    generated_plan[-1][f'{args.model_name}{suffix}_{args.mode}_parsed_results'] = parsed_result
            except:
                print(f"{idx}:\n{raw}\n This is an illegal json format. Please modify it manualy when this occurs.")
                generated_plan[-1][f'{args.model_name}{suffix}_{args.mode}_parsed_results'] = None
        else:
            if args.mode == 'two-stage':
                generated_plan[-1][f'{args.model_name}{suffix}_{args.mode}_parsed_results'] = None
            else:
                generated_plan[-1][f'{args.model_name}{suffix}_{args.mode}_parsed_results'] = None
  
        with open(f'{args.output_dir}/{args.set_type}/generated_plan_{idx}.json','w') as f:
            json.dump(generated_plan,f)
