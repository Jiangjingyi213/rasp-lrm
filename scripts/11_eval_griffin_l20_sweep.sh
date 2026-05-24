#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-python3}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

CONFIG_DIR="${CONFIG_DIR:-/tmp/rasp_griffin_l20_configs}"
mkdir -p "$CONFIG_DIR"

"$PYTHON" - <<'PY'
from pathlib import Path

config_dir = Path("/tmp/rasp_griffin_l20_configs")
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

def write_config(dataset: str, density: float | None) -> None:
    tag = "dense" if density is None else f"griffin_d{str(density).replace('.', '')}"
    run_dir = f"runs/eval_{tag}_qwen3_{dataset}_l20"
    model = base_model
    if density is not None:
        model += f"""\
  adapter: griffin_qwen3
  griffin_density: {density}
  griffin_selection_method: topk
  griffin_mode: gen
"""
    if dataset == "gsm8k":
        data = """\
data:
  dataset: gsm8k
  split: test
  limit: 20
"""
        max_new_tokens = 384
    elif dataset == "math500":
        data = """\
data:
  dataset: math500
  name_or_path: HuggingFaceH4/MATH-500
  split: test
  limit: 20
"""
        max_new_tokens = 512
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
    write_config(dataset, None)
    for density in (0.98, 0.95, 0.90, 0.80):
        write_config(dataset, density)
PY

for DATASET in gsm8k math500; do
  for TAG in dense griffin_d098 griffin_d095 griffin_d09 griffin_d08; do
    CONFIG="$CONFIG_DIR/${DATASET}_${TAG}.yaml"
    RUN_DIR="$("$PYTHON" - <<PY
import yaml
cfg=yaml.safe_load(open("$CONFIG"))
print(cfg["paths"]["run_dir"])
PY
)"
    echo "==> $CONFIG"
    rm -rf "$RUN_DIR"
    "$PYTHON" -m src.main_generate --config "$CONFIG"
  done
done

echo "GRIFFIN L20 sweep complete."
