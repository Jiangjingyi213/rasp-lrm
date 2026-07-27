#!/usr/bin/env bash
set -euo pipefail

export CONFIG="${CONFIG:-configs/stage_calibrated_pruning/wanda_official_qwen3_1p7b_priority_suite.yaml}"
export RUN_ROOT="${RUN_ROOT:-runs/08_stage_calibrated_pruning/main_pilot_wanda_official_qwen3_1p7b_priority_suite}"
export STAGE_FINAL_METHODS="${STAGE_FINAL_METHODS:-wanda_t30_official}"
export LOG_PREFIX="${LOG_PREFIX:-wanda_official_qwen3_1p7b_priority}"

SOURCE_ROOT="${SOURCE_ROOT:-runs/08_stage_calibrated_pruning/main_pilot_mixed_reasoning_seed3}"
if [[ ! -f "${SOURCE_ROOT}/03_selected/calibration.jsonl" ]]; then
  echo "Missing Wanda calibration file: ${SOURCE_ROOT}/03_selected/calibration.jsonl" >&2
  echo "Set SOURCE_ROOT to an existing mixed pilot run with 03_selected/calibration.jsonl." >&2
  exit 2
fi
export SOURCE_ROOT

exec bash scripts/run_griffin_prompt_priority_suite_gpu.sh
