#!/usr/bin/env bash
set -euo pipefail

export CONFIG="${CONFIG:-configs/stage_calibrated_pruning/mixed_reasoning_qwen3_4b_seed3_t30_math_safe_full.yaml}"
export SOURCE_ROOT="${SOURCE_ROOT:-runs/08_stage_calibrated_pruning/main_pilot_mixed_reasoning_qwen3_4b_seed3}"
export RUN_ROOT="${RUN_ROOT:-runs/08_stage_calibrated_pruning/main_pilot_mixed_reasoning_qwen3_4b_seed3_t30_math_safe_full}"
export STAGE_FINAL_METHODS="${STAGE_FINAL_METHODS:-structured_dense,static_t30_0p37,t30_math_safe}"
export RUN_LABEL="${RUN_LABEL:-qwen3_4b_t30_math_safe_full}"

exec bash scripts/run_t30_math_safe_full_8gpu.sh
