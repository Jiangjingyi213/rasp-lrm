#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-python3}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

CONFIG_DIR="${CONFIG_DIR:-/tmp/rasp_griffin_budget_configs}"
GSM8K_LIMIT="${GSM8K_LIMIT:-1319}"
MATH500_LIMIT="${MATH500_LIMIT:-500}"
DATASETS="${DATASETS:-gsm8k math500}"
TAGS="${TAGS:-dense griffin_p20 griffin_p40 griffin_p60}"
CLEAN_RUN_DIR="${CLEAN_RUN_DIR:-1}"

mkdir -p "$CONFIG_DIR"

"$PYTHON" - "$CONFIG_DIR" "$GSM8K_LIMIT" "$MATH500_LIMIT" <<'PY'
from pathlib import Path
import sys

config_dir = Path(sys.argv[1])
gsm8k_limit = int(sys.argv[2])
math500_limit = int(sys.argv[3])
config_dir.mkdir(parents=True, exist_ok=True)

base_model = """\
  name_or_path: Qwen/Qwen3-1.7B
  dtype: float32
  device_map: auto
  trust_remote_code: true
  attn_implementation: eager
"""

prompt = """\
prompt:
  use_chat_template: true
  enable_thinking: false
  system: You are a careful math reasoning assistant.
"""

budgets = {
    "p20": 0.80,
    "p40": 0.60,
    "p60": 0.40,
}

def write_config(dataset: str, tag: str, density: float | None) -> None:
    run_dir = f"runs/eval_{tag}_qwen3_{dataset}_budget"
    model = base_model
    if density is not None:
        model += f"""\
  adapter: griffin_qwen3
  griffin_density: {density}
  griffin_selection_method: topk
  griffin_mode: gen
"""

    if dataset == "gsm8k":
        data = f"""\
data:
  dataset: gsm8k
  split: test
  limit: {gsm8k_limit}
"""
        max_new_tokens = 768
    elif dataset == "math500":
        data = f"""\
data:
  dataset: math500
  name_or_path: HuggingFaceH4/MATH-500
  split: test
  limit: {math500_limit}
"""
        max_new_tokens = 1024
    else:
        raise ValueError(dataset)

    text = f"""\
seed: 1

model:
{model}
{prompt}
{data}
generation:
  max_input_tokens: 2048
  max_new_tokens: {max_new_tokens}
  temperature: 0.0
  top_p: 1.0

paths:
  run_dir: {run_dir}
  trajectories: {run_dir}/01_trajectories.jsonl
"""
    (config_dir / f"{dataset}_{tag}.yaml").write_text(text, encoding="utf-8")

for dataset in ("gsm8k", "math500"):
    write_config(dataset, "dense", None)
    for tag, density in budgets.items():
        write_config(dataset, f"griffin_{tag}", density)
PY

for DATASET in $DATASETS; do
  for TAG in $TAGS; do
    CONFIG="$CONFIG_DIR/${DATASET}_${TAG}.yaml"
    RUN_DIR="$("$PYTHON" - <<PY
import yaml
cfg = yaml.safe_load(open("$CONFIG"))
print(cfg["paths"]["run_dir"])
PY
)"
    echo "==> $CONFIG"
    if [[ "$CLEAN_RUN_DIR" == "1" ]]; then
      rm -rf "$RUN_DIR"
    fi
    "$PYTHON" -m src.main_generate --config "$CONFIG"
  done
done

echo "GRIFFIN budget sweep complete."
