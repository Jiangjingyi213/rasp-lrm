#!/usr/bin/env bash
set -euo pipefail

# Stage-budget-only ablation for the frozen t30 dynamic pruning recipe.
# This reuses mixed-pilot calibration artifacts and keeps prior/runtime/core/
# refresh settings fixed while changing only per-stage pruning ratios.
export CONFIG="${CONFIG:-configs/stage_calibrated_pruning/mixed_reasoning_seed3_t30_stage_budget_ablation.yaml}"
export RUN_ROOT="${RUN_ROOT:-runs/08_stage_calibrated_pruning/main_pilot_mixed_reasoning_seed3_t30_stage_budget_ablation}"
export DATASETS_OVERRIDE="${DATASETS_OVERRIDE:-gsm8k math500 arc_easy arc_challenge bbh_selected}"
export STAGE_FINAL_METHODS="${STAGE_FINAL_METHODS:-t30_current_budget,budget_uniform_t30,budget_setup_aggressive_t30,budget_reasoning_aggressive_t30,budget_verify_aggressive_t30,budget_final_aggressive_t30,budget_sensitivity_guided_t30}"
export LOG_PREFIX="${LOG_PREFIX:-t30_stage_budget_ablation}"

exec bash scripts/run_t30_math_safe_priority_suite_8gpu.sh
