#!/usr/bin/env bash
set -euo pipefail

export CONFIG="${CONFIG:-configs/stage_calibrated_pruning/limits_layer_pruning_qwen3_1p7b_full.yaml}"
export RUN_ROOT="${RUN_ROOT:-runs/08_stage_calibrated_pruning/main_pilot_limits_layer_pruning_qwen3_1p7b_full}"
export STAGE_FINAL_METHODS="${STAGE_FINAL_METHODS:-limits_reverse_t30_matched}"
export RUN_LABEL="${RUN_LABEL:-limits_layer_pruning_qwen3_1p7b_full}"

exec bash scripts/run_griffin_prompt_matched_full_gpu.sh
