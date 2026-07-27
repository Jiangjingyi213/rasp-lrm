#!/usr/bin/env bash
set -euo pipefail

export CONFIG="${CONFIG:-configs/stage_calibrated_pruning/shortgpt_qwen3_4b_full.yaml}"
export RUN_ROOT="${RUN_ROOT:-runs/08_stage_calibrated_pruning/main_pilot_shortgpt_qwen3_4b_full}"
export STAGE_FINAL_METHODS="${STAGE_FINAL_METHODS:-shortgpt_t20_matched,shortgpt_t30_matched}"
export RUN_LABEL="${RUN_LABEL:-shortgpt_qwen3_4b_full}"

SOURCE_ROOT="${SOURCE_ROOT:-runs/08_stage_calibrated_pruning/main_pilot_mixed_reasoning_seed3_t30_math_safe_full}"
if [[ ! -f "${SOURCE_ROOT}/03_selected/calibration.jsonl" ]]; then
  echo "Missing ShortGPT calibration file: ${SOURCE_ROOT}/03_selected/calibration.jsonl" >&2
  echo "Set SOURCE_ROOT to an existing mixed pilot run with 03_selected/calibration.jsonl." >&2
  exit 2
fi
export SOURCE_ROOT

exec bash scripts/run_griffin_prompt_matched_full_gpu.sh
