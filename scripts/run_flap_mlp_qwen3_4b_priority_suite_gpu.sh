#!/usr/bin/env bash
set -euo pipefail

export CONFIG="${CONFIG:-configs/stage_calibrated_pruning/flap_mlp_qwen3_4b_priority_suite.yaml}"
export RUN_ROOT="${RUN_ROOT:-runs/08_stage_calibrated_pruning/main_pilot_flap_mlp_qwen3_4b_priority_suite}"
export STAGE_FINAL_METHODS="${STAGE_FINAL_METHODS:-flap_mlp_t30_official}"
export LOG_PREFIX="${LOG_PREFIX:-flap_mlp_qwen3_4b_priority}"

SOURCE_ROOT="${SOURCE_ROOT:-runs/08_stage_calibrated_pruning/main_pilot_mixed_reasoning_seed3}"
if [[ ! -f "${SOURCE_ROOT}/03_selected/calibration.jsonl" ]]; then
  echo "Missing FLAP calibration file: ${SOURCE_ROOT}/03_selected/calibration.jsonl" >&2
  echo "Set SOURCE_ROOT to an existing mixed pilot run with 03_selected/calibration.jsonl." >&2
  exit 2
fi
export SOURCE_ROOT
export BASELINE_LABEL="${BASELINE_LABEL:-FLAP-MLP Qwen3-4B}"
export BASELINE_SCHEMA="${BASELINE_SCHEMA:-flap_mlp_qwen3_4b_priority_suite_aggregate_v1}"

exec bash scripts/run_griffin_prompt_priority_suite_gpu.sh
