#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-python3}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

CONFIG_DIR="${CONFIG_DIR:-/tmp/rasp_flap_mlp_qwen3_budget_configs}"
GSM8K_LIMIT="${GSM8K_LIMIT:-1319}"
MATH500_LIMIT="${MATH500_LIMIT:-500}"
DATASETS="${DATASETS:-gsm8k math500}"
TAGS="${TAGS:-flap_mlp_p20 flap_mlp_p40 flap_mlp_p60}"
CLEAN_RUN_DIR="${CLEAN_RUN_DIR:-1}"
FLAP_METRIC="${FLAP_METRIC:-WIFV}"
FLAP_STRUCTURE="${FLAP_STRUCTURE:-AL-AM}"
FLAP_CALIBRATION_DATASET="${FLAP_CALIBRATION_DATASET:-wikitext2}"
FLAP_CALIBRATION_SAMPLES="${FLAP_CALIBRATION_SAMPLES:-32}"
FLAP_BIAS_COMPENSATION="${FLAP_BIAS_COMPENSATION:-true}"

mkdir -p "$CONFIG_DIR"

"$PYTHON" - "$CONFIG_DIR" "$GSM8K_LIMIT" "$MATH500_LIMIT" "$FLAP_METRIC" "$FLAP_STRUCTURE" "$FLAP_CALIBRATION_DATASET" "$FLAP_CALIBRATION_SAMPLES" "$FLAP_BIAS_COMPENSATION" <<'PY'
from pathlib import Path
import sys

config_dir = Path(sys.argv[1])
gsm8k_limit = int(sys.argv[2])
math500_limit = int(sys.argv[3])
flap_metric = sys.argv[4]
flap_structure = sys.argv[5]
flap_calibration_dataset = sys.argv[6]
flap_calibration_samples = int(sys.argv[7])
flap_bias_compensation = sys.argv[8].lower() in {"1", "true", "yes", "y"}
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
    "p20": 0.20,
    "p40": 0.40,
    "p60": 0.60,
}

def write_config(dataset: str, tag: str, ratio: float) -> None:
    metric_tag = flap_metric.lower()
    structure_tag = flap_structure.lower().replace("-", "")
    calib_tag = flap_calibration_dataset.lower().replace("-", "")
    run_dir = f"runs/eval_flap_mlp_{metric_tag}_{structure_tag}_{calib_tag}_{tag}_qwen3_{dataset}_budget"
    model = base_model + f"""\
  adapter: flap_mlp_qwen3
  flap_pruning_ratio: {ratio}
  flap_metric: {flap_metric}
  flap_structure: {flap_structure}
  flap_calibration_dataset: {flap_calibration_dataset}
  flap_calibration_samples: {flap_calibration_samples}
  flap_bias_compensation: {str(flap_bias_compensation).lower()}
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
  flap_mlp_summary: {run_dir}/00_flap_mlp_summary.json
"""
    (config_dir / f"{dataset}_flap_mlp_{tag}.yaml").write_text(text, encoding="utf-8")

for dataset in ("gsm8k", "math500"):
    for tag, ratio in budgets.items():
        write_config(dataset, tag, ratio)
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

echo "FLAP-MLP Qwen3 budget sweep complete."
