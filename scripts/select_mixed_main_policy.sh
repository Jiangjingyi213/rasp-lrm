#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON:-python3}"
ROOT="${ROOT:-runs/08_stage_calibrated_pruning/main_pilot_mixed_reasoning_seed3}"
OUTPUT_DIR="${OUTPUT_DIR:-runs/08_stage_calibrated_pruning/main_pilot_mixed_reasoning_seed3_policy_selection}"
MAIN_ONLY_MAX_DROP="${MAIN_ONLY_MAX_DROP:-0.05}"
MAIN_ONLY_RATIOS="${MAIN_ONLY_RATIOS:-0.10 0.15 0.20}"

read -r -a ratios <<< "${MAIN_ONLY_RATIOS}"

"${PYTHON_BIN}" -m src.stage_calibration.policy_selection \
  --main-only \
  --main-only-max-drop "${MAIN_ONLY_MAX_DROP}" \
  --main-only-candidate-ratios "${ratios[@]}" \
  --roots "${ROOT}" \
  --output-dir "${OUTPUT_DIR}"

echo "Wrote ${OUTPUT_DIR}/policy_selection_main_only.json"
echo "Wrote ${OUTPUT_DIR}/policy_selection_main_only.md"
