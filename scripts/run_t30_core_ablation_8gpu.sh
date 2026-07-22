#!/usr/bin/env bash
set -euo pipefail

# Core mechanism ablation for the frozen t30_math_safe method.
# This reuses mixed-pilot calibration artifacts and evaluates only 1040
# diagnostic examples by default.
export CONFIG="${CONFIG:-configs/stage_calibrated_pruning/mixed_reasoning_seed3_t30_core_ablation.yaml}"
export RUN_ROOT="${RUN_ROOT:-runs/08_stage_calibrated_pruning/main_pilot_mixed_reasoning_seed3_t30_core_ablation}"
export DATASETS_OVERRIDE="${DATASETS_OVERRIDE:-gsm8k math500 arc_easy arc_challenge bbh_selected}"
export STAGE_FINAL_METHODS="${STAGE_FINAL_METHODS:-structured_dense,static_t30_0p37,t30_math_safe,fixed_global_t30_ratios,fixed_stage_specific_t30_ratios,fixed_shuffled_stage_t30_ratios,dynamic_global_prior_t30,dynamic_shuffled_prior_t30,prior_only_t30,runtime_heavy_t30,no_protected_core_t30,uniform_budget_t30}"
export LOG_PREFIX="${LOG_PREFIX:-t30_core_ablation}"

exec bash scripts/run_t30_math_safe_priority_suite_8gpu.sh
