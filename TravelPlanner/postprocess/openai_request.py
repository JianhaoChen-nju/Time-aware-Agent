import os
import sys
import time
import json
from tqdm import tqdm
from typing import Iterable, List, TypeVar
from openai import OpenAI
from datasets import load_dataset
from dotenv import load_dotenv

# Load .env from project root
load_dotenv(os.path.join(os.path.dirname(__file__), '..', '..', '.env'))

T = TypeVar('T')

# Use LLM_API_KEY/LLM_BASE_URL from .env, fallback to OPENAI_API_KEY
API_KEY = os.environ.get('LLM_API_KEY', os.environ.get('OPENAI_API_KEY', ''))
BASE_URL = os.environ.get('LLM_BASE_URL', None)
MODEL_NAME = os.environ.get('LLM_MODEL_NAME', 'gpt-4-1106-preview')

client = OpenAI(api_key=API_KEY, base_url=BASE_URL)


def prompt_chatgpt(system_input, user_input, temperature, save_path, index, history=[], model_name=None):
    '''
    :param system_input: "You are a helpful assistant."
    :param user_input: your text here
    :param model_name: ignored, uses MODEL_NAME from .env
    return: assistant_output, (updated) history, money cost
    '''
    # Always use the model from .env
    actual_model = MODEL_NAME

    if len(history) == 0:
        history = [{"role": "system", "content": system_input}]
    history.append({"role": "user", "content": user_input})

    while True:
        try:
            completion = client.chat.completions.create(
                model=actual_model,
                messages=history,
                temperature=1,
                timeout=120
            )
            break
        except Exception as e:
            print(f"API error: {e}, retrying...")
            time.sleep(2)

    assistant_output = completion.choices[0].message.content
    history.append({"role": "assistant", "content": assistant_output})

    with open(save_path, 'a+', encoding='utf-8') as f:
        output_line = str(index) + "\t" + "\t".join(x for x in assistant_output.split("\n"))
        f.write(output_line + '\n')

    return assistant_output, history, 0  # cost tracking not needed for custom model


def build_plan_format_conversion_prompt(directory, set_type='validation', model_name='gpt4', strategy='direct', mode='two-stage'):
    prompt_list = []
    prefix = """Please assist me in extracting valid information from a given natural language text and reconstructing it in JSON format, as demonstrated in the following example. If transportation details indicate a journey from one city to another (e.g., from A to B), the 'current_city' should be updated to the destination city (in this case, B). Use a ';' to separate different attractions, with each attraction formatted as 'Name, City'. If there's information about transportation, ensure that the 'current_city' aligns with the destination mentioned in the transportation details (i.e., the current city should follow the format 'from A to B'). Also, ensure that all flight numbers and costs are followed by a colon (i.e., 'Flight Number:' and 'Cost:'), consistent with the provided example. Each item should include ['day', 'current_city', 'transportation', 'breakfast', 'attraction', 'lunch', 'dinner', 'accommodation']. Replace non-specific information like 'eat at home/on the road' with '-'. Additionally, delete any '$' symbols.
-----EXAMPLE-----
 [{{
        "days": 1,
        "current_city": "from Dallas to Peoria",
        "transportation": "Flight Number: 4044830, from Dallas to Peoria, Departure Time: 13:10, Arrival Time: 15:01",
        "breakfast": "-",
        "attraction": "Peoria Historical Society, Peoria;Peoria Holocaust Memorial, Peoria;",
        "lunch": "-",
        "dinner": "Tandoor Ka Zaika, Peoria",
        "accommodation": "Bushwick Music Mansion, Peoria"
    }},
    {{
        "days": 2,
        "current_city": "Peoria",
        "transportation": "-",
        "breakfast": "Tandoor Ka Zaika, Peoria",
        "attraction": "Peoria Riverfront Park, Peoria;The Peoria PlayHouse, Peoria;Glen Oak Park, Peoria;",
        "lunch": "Cafe Hashtag LoL, Peoria",
        "dinner": "The Curzon Room - Maidens Hotel, Peoria",
        "accommodation": "Bushwick Music Mansion, Peoria"
    }},
    {{
        "days": 3,
        "current_city": "from Peoria to Dallas",
        "transportation": "Flight Number: 4045904, from Peoria to Dallas, Departure Time: 07:09, Arrival Time: 09:20",
        "breakfast": "-",
        "attraction": "-",
        "lunch": "-",
        "dinner": "-",
        "accommodation": "-"
    }}]
-----EXAMPLE END-----
"""
    if set_type == 'train':
        query_data_list = load_dataset('osunlp/TravelPlanner', 'train')['train']
    elif set_type == 'validation':
        query_data_list = load_dataset('osunlp/TravelPlanner', 'validation')['validation']
    elif set_type == 'test':
        query_data_list = load_dataset('osunlp/TravelPlanner', 'test')['test']

    idx_number_list = [i for i in range(1, len(query_data_list) + 1)]
    if mode == 'two-stage':
        suffix = ''
    elif mode == 'sole-planning':
        suffix = f'_{strategy}'
    for idx in tqdm(idx_number_list):
        generated_plan = json.load(open(f'{directory}/{set_type}/generated_plan_{idx}.json'))
        if generated_plan[-1][f'{model_name}{suffix}_{mode}_results'] and generated_plan[-1][f'{model_name}{suffix}_{mode}_results'] != "":
            prompt = prefix + "Text:\n" + generated_plan[-1][f'{model_name}{suffix}_{mode}_results'] + "\nJSON:\n"
        else:
            prompt = ""
        prompt_list.append(prompt)
    return prompt_list
