# Time-aware-Agent

This repository implements TAES, a neuro-symbolic planning framework for
time-aware and constraint-aware agents. TAES combines LLM candidate generation,
symbolic verification, heuristic state scoring, and beam search.

The current implementation supports two benchmarks:

- `TimeArena`: time-aware multi-task execution.
- `TravelPlanner`: multi-day travel planning with hard constraints.

## Repository Structure

```text
src/taes/                 TAES implementation
TimeArena/                TimeArena benchmark
TravelPlanner/            TravelPlanner benchmark
scripts/                  Experiment scripts
TAES_ALGORITHM.md         Algorithm notes
TKG_IN_TAES.md            Temporal knowledge graph notes
EXPERIMENT_RESULTS.md     Experiment summary
```

## Environment

Create a Python environment and install dependencies:

```bash
conda create -n timeagent python=3.9
conda activate timeagent

pip install -r TimeArena/requirements.txt
pip install -r TravelPlanner/requirements.txt
pip install langchain langchain-openai openai datasets pandas tqdm python-dotenv tenacity
```

Configure the model endpoint in `.env`:

```bash
LLM_MODEL_NAME=your-model-name
LLM_API_KEY=your-api-key
LLM_BASE_URL=https://your-endpoint/v1
```

For TravelPlanner, make sure the Hugging Face dataset is available. 
Download the [database](https://drive.google.com/file/d/1pF1Sw6pBmq2sFkJvm-LzJOqrmfWoQgxE/view?usp=drive_link) and unzip it to the `TravelPlanner` directory (i.e., `your/path/TravelPlanner`).
If it is
already cached locally, you can run with:

```bash
export HF_DATASETS_OFFLINE=1
```

## Run TravelPlanner

TravelPlanner is evaluated on the `validation` split with 180 examples.

Run the direct baseline:

```bash
ROOT=$(pwd)
PY=python

PYTHONPATH="$ROOT/TravelPlanner:$ROOT" $PY -u TravelPlanner/tools/planner/sole_planning.py \
  --set_type validation \
  --model_name custom \
  --strategy direct \
  --output_dir output_travelplanner_direct \
  --skip_existing
```

Run TAES:

```bash
ROOT=$(pwd)
PY=python

PYTHONPATH="$ROOT/TravelPlanner:$ROOT" $PY -u TravelPlanner/tools/planner/sole_planning.py \
  --set_type validation \
  --model_name custom \
  --strategy taes \
  --output_dir output_travelplanner_taes \
  --skip_existing \
  --beam_width 5 \
  --branch_factor 3
```

Postprocess and evaluate:

```bash
ROOT=$(pwd)
OUT=output_travelplanner_taes
STRATEGY=taes

mkdir -p "$OUT/tmp"

cd TravelPlanner/postprocess
python -u parsing.py \
  --set_type validation \
  --model_name custom \
  --mode sole-planning \
  --strategy "$STRATEGY" \
  --output_dir "$ROOT/$OUT" \
  --tmp_dir "$ROOT/$OUT/tmp"

python -u element_extraction.py \
  --set_type validation \
  --model_name custom \
  --mode sole-planning \
  --strategy "$STRATEGY" \
  --output_dir "$ROOT/$OUT" \
  --tmp_dir "$ROOT/$OUT/tmp"

python -u combination.py \
  --set_type validation \
  --model_name custom \
  --mode sole-planning \
  --strategy "$STRATEGY" \
  --output_dir "$ROOT/$OUT" \
  --submission_file_dir "$ROOT/$OUT"

cd ../evaluation
python -u eval.py \
  --set_type validation \
  --evaluation_file_path "$ROOT/$OUT/validation_custom_${STRATEGY}_sole-planning_submission.jsonl"
```

For the direct baseline postprocessing, set:

```bash
OUT=output_travelplanner_direct
STRATEGY=direct
```

## Run TimeArena

TimeArena is evaluated on 63 cases: task indices `4..10` for single-task,
two-task, and three-task settings across the `household`, `cooking`, and `lab`
domains. The time budget is `n_tasks * 40`.

Run a direct baseline case:

```bash
cd TimeArena
python -u LLM_test.py \
  --taskName household4 \
  --lm custom \
  --total_time 40 \
  --save_path trajectory/direct \
  --save_name household4
```

Run a TAES case:

```bash
cd TimeArena
python -u LLM_test.py \
  --taskName household4 \
  --prompting taes \
  --lm custom \
  --total_time 40 \
  --save_path trajectory/taes \
  --save_name household4
```

Calculate metrics:

```bash
python -u TimeArena/metrics/cal_metric.py \
  --path TimeArena/trajectory/taes \
  --timearena_63
```

## Full Experiment Script

The DeepSeek-V4-Pro experiment can be launched with:

```bash
scripts/run_deepseek_v4_experiment.sh
```

The script runs both benchmarks for direct baseline and TAES, then writes
outputs under:

```text
output_deepseek_v4_tp_baseline/
output_deepseek_v4_tp_taes/
TimeArena/trajectory/deepseek_v4_baseline63_n40/
TimeArena/trajectory/deepseek_v4_taes63_n40/
```

## Results

### GPT-5.5

TravelPlanner validation set:

| Method | Delivery | Commonsense Micro | Commonsense Macro | Hard Micro | Hard Macro | Final Pass Rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Direct baseline | 89.44 | 82.29 | 42.22 | 72.86 | 73.33 | 39.44 |
| TAES | 100.00 | 100.00 | 100.00 | 100.00 | 100.00 | 100.00 |

TimeArena 63-case subset:

| Method | AS | CS | CR | CT |
| --- | ---: | ---: | ---: | ---: |
| Direct baseline | 98.86 | 2.69 | 85.71 | 30.02 |
| TAES | 100.00 | 3.43 | 100.00 | 29.16 |

### DeepSeek-V4-Pro

TravelPlanner validation set:

| Method | Delivery | Commonsense Micro | Commonsense Macro | Hard Micro | Hard Macro | Final Pass Rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Direct baseline | 100.00 | 86.94 | 26.11 | 60.48 | 43.89 | 12.78 |
| TAES | 100.00 | 99.10 | 93.89 | 94.52 | 92.78 | 92.22 |

TimeArena 63-case subset:

| Method | AS | CS | CR | CT |
| --- | ---: | ---: | ---: | ---: |
| Direct baseline | 92.73 | 2.29 | 82.54 | 34.48 |
| TAES | 100.00 | 3.39 | 100.00 | 29.54 |

Metric abbreviations for TimeArena:

- `AS`: average progress score
- `CS`: completion speed
- `CR`: completion rate
- `CT`: average completion time

