#!/usr/bin/env bash
set -euo pipefail

export CONFIG="${CONFIG:-configs/stage_calibrated_pruning/mixed_reasoning_seed3_t20_math_safe_priority_suite.yaml}"
export RUN_ROOT="${RUN_ROOT:-runs/08_stage_calibrated_pruning/main_pilot_mixed_reasoning_seed3_t20_math_safe_priority_suite}"
export STAGE_FINAL_METHODS="${STAGE_FINAL_METHODS:-structured_dense,static_t20_0p25,t20_math_safe}"
export LOG_PREFIX="${LOG_PREFIX:-t20_priority_suite}"
export DATASETS_OVERRIDE="${DATASETS_OVERRIDE:-amc2023 gpqa_diamond arc_challenge aime2024 aime2025}"

exec bash scripts/run_t30_math_safe_priority_suite_8gpu.sh
