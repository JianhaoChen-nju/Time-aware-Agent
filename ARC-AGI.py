import json
import numpy as np
import time
import concurrent.futures
from tqdm import tqdm
from datasets import load_dataset
from openai import OpenAI, APIStatusError
from cached_openai import create_mysql_cached_client

# --- 配置区域 ---

# 设置 OpenAI API 客户端
# 请确保你的 API 服务地址和 Key 是正确的
http_client = create_mysql_cached_client(config_file="config.json")
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key="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",
    http_client=http_client
)

# 要评测的模型列表
model_list=[
    "openai/gpt-5-chat",
    # "google/gemini-2.5-pro",
    # "google/gemini-2.5-flash",
    "anthropic/claude-opus-4.1",
    "anthropic/claude-sonnet-4",
    # "qwen/qwen3-235b-a22b-2507"
]

# --- 辅助函数 ---

def visualize_grid(grid):
    """将网格（列表的列表）转换为易于阅读的字符串表示。"""
    if grid is None:
        return "None"
    return "\n".join([" ".join([str(cell) for cell in row]) for row in grid])

def are_grids_equal(grid1, grid2):
    """
    安全地比较两个网格。首先检查维度是否一致，然后才进行元素级比较，
    以避免因形状不匹配而导致的 ValueError。
    """
    if grid1 is None or grid2 is None:
        return False
    try:
        arr1 = np.array(grid1)
        arr2 = np.array(grid2)
        if arr1.shape != arr2.shape:
            return False
        return np.array_equal(arr1, arr2)
    except ValueError:
        return False

# --- 核心求解函数 ---

def solve_arc_task(train_examples, test_input, model_name):
    """
    核心求解函数，使用指定的大模型来解决 ARC-AGI 任务。
    返回: tuple: (预测的网格, 原始模型输出字符串, API使用情况对象, 生成的prompt)
    """
    prompt = "你是一个专家级的AI，任务是解决一个抽象推理谜题。\n"
    prompt += "每个谜题都由一组训练示例（输入网格和相应的输出网格）组成，其中包含一个抽象的转换规则。你需要推断出这个规则，并将其应用到一个新的测试输入网格上。\n\n"
    
    prompt += "--- 训练示例 ---\n"
    for i, example in enumerate(train_examples):
        prompt += f"示例 {i+1}:\n输入网格:\n{visualize_grid(example['input'])}\n\n输出网格:\n{visualize_grid(example['output'])}\n\n"
    
    prompt += f"--- 测试任务 ---\n测试输入网格:\n{visualize_grid(test_input)}\n\n"
    prompt += "请根据训练示例中观察到的规则，预测测试输入的输出网格。\n"
    prompt += "请仅输出一个 JSON 列表，格式为列表的列表（例如：[[1, 2], [3, 4]]），不要包含任何其他文字或解释。"
    
    messages = [{"role": "user", "content": prompt}]

    try:
        response = client.chat.completions.create(model=model_name, messages=messages, extra_body={"usage": {"include": True}})
        model_output_str = response.choices[0].message.content.strip()
        usage_stats = response.usage
        
        parsing_output = model_output_str
        if parsing_output.startswith("```json"):
            parsing_output = parsing_output[7:-3].strip()
        
        predicted_grid = json.loads(parsing_output)
        
        if isinstance(predicted_grid, list) and all(isinstance(row, list) for row in predicted_grid):
            return predicted_grid, model_output_str, usage_stats, prompt
        else:
            print(f"警告: 模型 {model_name} 输出格式不正确。输出: {model_output_str}")
            return None, model_output_str, usage_stats, prompt
    except APIStatusError as e:
        error_message = str(e.response.text)
        if "unsupported_country_region_territory" in error_message:
            print(f"警告: 模型 {model_name} 因地理区域限制被跳过。")
            return None, f"Skipped due to geo-restriction: {error_message}", None, prompt
        else:
            print(f"调用模型 {model_name} 时发生API状态错误: {e}")
            return None, f"Error: {e}", None, prompt
    except Exception as e:
        print(f"调用模型 {model_name} 失败或解析输出时出错: {e}")
        return None, f"Error: {e}", None, prompt

# --- 并行处理的工作单元 ---

def process_task_worker(args):
    """
    被每个线程调用的工作函数，处理单个任务并返回所有需要的信息。
    """
    index, task, model_name = args
    
    # --- 关键修改：安全地解析可能为JSON字符串的数据 ---
    train_data_raw = task["train"]
    test_data_raw = task["test"]

    # 如果数据是字符串，则用 json.loads() 解析它
    train_examples = json.loads(train_data_raw) if isinstance(train_data_raw, str) else train_data_raw
    test_pair = json.loads(test_data_raw) if isinstance(test_data_raw, str) else test_data_raw
    
    # 现在可以安全地访问解析后的对象
    test_input = test_pair[0]["input"]
    correct_output = test_pair[0]["output"]
    # --- 修改结束 ---
    
    predicted_output, raw_output, usage, prompt_text = solve_arc_task(train_examples, test_input, model_name)
    
    is_correct = False
    is_skipped = raw_output and "Skipped due to geo-restriction" in raw_output
    
    if not is_skipped and predicted_output:
        is_correct = are_grids_equal(predicted_output, correct_output)

    origin_query = "--- 训练示例 ---\n"
    for j, example in enumerate(train_examples):
        origin_query += f"示例 {j+1}:\n输入:\n{visualize_grid(example['input'])}\n输出:\n{visualize_grid(example['output'])}\n\n"
    origin_query += "--- 测试任务 ---\n输入:\n" + visualize_grid(test_input)

    record = {
        "index": index,
        "query": prompt_text,
        "origin_query": origin_query,
        "prediction": predicted_output,
        "full_prediction": [raw_output],
        "raw_output": [raw_output],
        "answer": correct_output,
        "is_correct": is_correct,
        "model_name": model_name,
        "usage": {
            "prompt_tokens": usage.prompt_tokens if usage else 0,
            "completion_tokens": usage.completion_tokens if usage else 0,
            "cost": getattr(usage, "cost", 0) if usage else 0,
        }
    }
    
    return index, record, is_correct, usage, is_skipped

# --- 评测执行函数 ---

def run_evaluation_for_model(model_name, dataset):
    """为单个模型并行执行完整的评测流程并保存结果。"""
    
    # 更新文件名以反映数据集版本
    output_filename = f"arc-agi-v1_{model_name.replace('/', '_')}.json"
    print(f"\n{'='*20}\n开始评测模型: {model_name}\n{'='*20}")

    results = {"performance": 0, "time_taken": 0, "prompt_tokens": 0, "completion_tokens": 0, "cost": 0, "model_counts": {model_name: 0}, "records": []}
    
    total_tasks_processed = 0
    solved_tasks = 0
    start_time = time.time()

    tasks_to_process = [(i, task, model_name) for i, task in enumerate(dataset)]
    all_records = [None] * len(dataset)

    # 使用 ThreadPoolExecutor 进行并行处理
    with concurrent.futures.ThreadPoolExecutor(max_workers=130) as executor:
        future_to_task = {executor.submit(process_task_worker, task_args): task_args for task_args in tasks_to_process}
        
        for future in tqdm(concurrent.futures.as_completed(future_to_task), total=len(tasks_to_process), desc=f"评测 {model_name}"):
            try:
                index, record, is_correct, usage, is_skipped = future.result()
                all_records[index] = record
                
                if not is_skipped:
                    if is_correct:
                        solved_tasks += 1
                    
                    if usage:
                        results["prompt_tokens"] += usage.prompt_tokens or 0
                        results["completion_tokens"] += usage.completion_tokens or 0
                        results["cost"] += getattr(usage, "cost", 0) or 0
                    
                    results["model_counts"][model_name] += 1
                    total_tasks_processed += 1

            except Exception as exc:
                task_args = future_to_task[future]
                print(f"任务 {task_args[0]} 生成了一个异常: {exc}")

    results["records"] = all_records
    end_time = time.time()
    
    results["time_taken"] = end_time - start_time
    if total_tasks_processed > 0:
        results["performance"] = solved_tasks / total_tasks_processed
    
    print(f"\n评测完成，正在将 {model_name} 的结果保存到 {output_filename}...")
    try:
        with open(output_filename, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=4, ensure_ascii=False)
        print("   结果保存成功。")
    except Exception as e:
        print(f"   保存文件失败: {e}")

    print(f"\n--- 模型 {model_name} 评测总结 ---")
    print(f"总评测任务数: {total_tasks_processed}")
    print(f"正确解决的任务数: {solved_tasks}")
    if total_tasks_processed > 0:
        print(f"准确率: {results['performance']:.4f}")
    print(f"总耗时: {results['time_taken']:.2f} 秒")
    print(f"总成本: ${results['cost']:.6f}")
    print(f"总 Prompt Tokens: {results['prompt_tokens']}")
    print(f"总 Completion Tokens: {results['completion_tokens']}")

# --- 主程序入口 ---

def main():
    """主函数，加载数据并为列表中的每个模型启动评测。"""
    print("1. 开始加载 ARC-AGI v1 数据集...")
    try:
        # 更新为 v1 数据集
        dataset = load_dataset("Asap7772/arc-agi-all", split="test")
        print(f"   数据集加载成功，共 {len(dataset)} 个任务。")
    except Exception as e:
        print(f"   加载数据集失败。请确保网络连接正常，并且有权限访问该数据集。\n   错误信息: {e}")
        return

    for model_name in model_list:
        run_evaluation_for_model(model_name, dataset)

if __name__ == "__main__":
    main()
