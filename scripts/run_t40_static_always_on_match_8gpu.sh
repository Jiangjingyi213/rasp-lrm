#!/usr/bin/env bash
set -euo pipefail

# Static-only strict matched baseline. This does not rerun t40_stage_safe.
export CONFIG="${CONFIG:-configs/stage_calibrated_pruning/mixed_reasoning_seed3_t40_strict_final_suite.yaml}"
export RUN_ROOT="${RUN_ROOT:-runs/08_stage_calibrated_pruning/main_pilot_mixed_reasoning_seed3_t40_static_always_on_match}"
export DATASETS_OVERRIDE="${DATASETS_OVERRIDE:-gsm8k math500 amc2023 gpqa_diamond arc_challenge}"
export STAGE_FINAL_METHODS="${STAGE_FINAL_METHODS:-static_t40_always_on_0p47}"
export LOG_PREFIX="${LOG_PREFIX:-t40_static_always_on_match}"

exec bash scripts/run_t30_math_safe_priority_suite_8gpu.sh
