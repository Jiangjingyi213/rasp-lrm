#!/usr/bin/env bash
set -euo pipefail

# Thin wrapper over the priority-suite worker pool. It runs only the final
# 40%+ boundary methods and excludes AIME by default.
export CONFIG="${CONFIG:-configs/stage_calibrated_pruning/mixed_reasoning_seed3_t40_strict_final_suite.yaml}"
export RUN_ROOT="${RUN_ROOT:-runs/08_stage_calibrated_pruning/main_pilot_mixed_reasoning_seed3_t40_strict_final_suite}"
export DATASETS_OVERRIDE="${DATASETS_OVERRIDE:-gsm8k math500 amc2023 gpqa_diamond arc_challenge}"
export STAGE_FINAL_METHODS="${STAGE_FINAL_METHODS:-static_t40_0p48,t40_stage_safe,static_t45_0p52,t45_boundary_safe}"
export LOG_PREFIX="${LOG_PREFIX:-t40_strict_final_suite}"

exec bash scripts/run_t30_math_safe_priority_suite_8gpu.sh
