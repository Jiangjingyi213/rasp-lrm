#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

RUN_ROOT="${RUN_ROOT:-runs/10_stage_budget_output_aware_v1}"
LOG_ROOT="${LOG_ROOT:-logs/10_stage_budget_output_aware_v1}"
SOURCE_ROOT="${SOURCE_ROOT:-runs/08_stage_calibrated_pruning/main_pilot_mixed_reasoning_seed3_t30_math_safe_full}"
CONFIG="${CONFIG:-configs/generated_stage_budget_output_aware_v1/frozen_full.template.yaml}"
METHODS="structured_dense,current_t30_math_safe,dynamic_global_activation_fixed_t30,dynamic_global_output_aware_budgeted"
SEEDS="${STAGE_BUDGET_SEEDS:-3}"

bash scripts/prepare_stage_budget_output_aware_v1.sh

for seed in ${SEEDS}; do
  phase_root="${RUN_ROOT}/05_frozen_full/seed_${seed}"
  CONFIG="${CONFIG}" RUN_ROOT="${phase_root}" SOURCE_ROOT="${SOURCE_ROOT}" LOG_DIR="${LOG_ROOT}/05_frozen_full/seed_${seed}" \
    STAGE_SEED="${seed}" STAGE_FINAL_SEEDS="${seed}" DATASETS_OVERRIDE="gsm8k math500" STAGE_FINAL_EVAL_LIMIT="${STAGE_FINAL_EVAL_LIMIT:--1}" STAGE_FINAL_METHODS="${METHODS}" \
    bash scripts/run_t30_math_safe_priority_suite_8gpu.sh
done
