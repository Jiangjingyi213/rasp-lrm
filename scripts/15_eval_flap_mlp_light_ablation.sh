#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-python3}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

RUN_ROOT="${RUN_ROOT:-runs/flap_mlp_light_ablation}"
CONFIG_DIR="${CONFIG_DIR:-/tmp/rasp_flap_mlp_light_ablation_configs}"
GSM8K_LIMIT="${GSM8K_LIMIT:-20}"
MATH500_LIMIT="${MATH500_LIMIT:-20}"
DATASETS="${DATASETS:-gsm8k math500}"
TAGS="${TAGS:-flap_mlp_p05 flap_mlp_p10}"
FLAP_METRIC="${FLAP_METRIC:-WIFN}"
FLAP_STRUCTURE="${FLAP_STRUCTURE:-UL-UM}"
FLAP_CALIBRATION_DATASET="${FLAP_CALIBRATION_DATASET:-wikitext2}"
FLAP_CALIBRATION_SAMPLES="${FLAP_CALIBRATION_SAMPLES:-32}"

for BIAS in true false; do
  echo "==> FLAP-MLP light ablation bias_compensation=${BIAS}"
  RUN_ROOT="$RUN_ROOT" \
  CONFIG_DIR="$CONFIG_DIR" \
  GSM8K_LIMIT="$GSM8K_LIMIT" \
  MATH500_LIMIT="$MATH500_LIMIT" \
  DATASETS="$DATASETS" \
  TAGS="$TAGS" \
  FLAP_METRIC="$FLAP_METRIC" \
  FLAP_STRUCTURE="$FLAP_STRUCTURE" \
  FLAP_CALIBRATION_DATASET="$FLAP_CALIBRATION_DATASET" \
  FLAP_CALIBRATION_SAMPLES="$FLAP_CALIBRATION_SAMPLES" \
  FLAP_BIAS_COMPENSATION="$BIAS" \
  "$ROOT/scripts/14_eval_flap_mlp_qwen3_budget_sweep.sh"
done

"$PYTHON" - <<PY
from pathlib import Path

run_root = Path("$RUN_ROOT")
print(f"FLAP-MLP light ablation complete. Results under: {run_root}")
for path in sorted(run_root.glob("*/01_trajectories.jsonl")):
    print(path.parent)
PY
