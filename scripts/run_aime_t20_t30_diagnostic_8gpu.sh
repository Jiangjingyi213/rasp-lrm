#!/usr/bin/env bash
set -euo pipefail

export CONFIG="${CONFIG:-configs/stage_calibrated_pruning/mixed_reasoning_seed3_aime_t20_t30_diagnostic.yaml}"
export RUN_ROOT="${RUN_ROOT:-runs/08_stage_calibrated_pruning/main_pilot_mixed_reasoning_seed3_aime_t20_t30_diagnostic}"
export DATASETS_OVERRIDE="${DATASETS_OVERRIDE:-aime2024 aime2025}"
export STAGE_FINAL_METHODS="${STAGE_FINAL_METHODS:-ordinary_dense,structured_dense,static_t20_0p25,t20_math_safe,static_t30_0p37,t30_math_safe}"
export LOG_PREFIX="${LOG_PREFIX:-aime_t20_t30_diagnostic}"

exec bash scripts/run_t30_math_safe_priority_suite_8gpu.sh
