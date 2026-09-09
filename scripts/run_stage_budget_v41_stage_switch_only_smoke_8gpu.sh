#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

RUN_ROOT="${RUN_ROOT:-runs/10_stage_budget_output_aware_v1}"
LOG_ROOT="${LOG_ROOT:-logs/10_stage_budget_output_aware_v1}"
SOURCE_ROOT="${SOURCE_ROOT:-runs/08_stage_calibrated_pruning/main_pilot_mixed_reasoning_seed3_t30_math_safe_full}"
CONFIG="${CONFIG:-configs/generated_stage_budget_output_aware_v1/budget_v41_stage_switch_only_smoke.yaml}"
METHODS="${STAGE_SWITCH_ONLY_METHODS:-dynamic_global_activation_budgeted_v41_stage_lift_plus_32p,dynamic_global_activation_budgeted_v41_stage_switch_only_plus_32p}"
SEEDS="${STAGE_BUDGET_SEEDS:-3}"
EXPERIMENT_ROOT="${RUN_ROOT}/11_stage_switch_only_ratio"
EXPERIMENT_LOG_ROOT="${LOG_ROOT}/11_stage_switch_only_ratio"

bash scripts/prepare_stage_budget_output_aware_v1.sh
mkdir -p "${EXPERIMENT_ROOT}/00_smoke_32p" "${EXPERIMENT_LOG_ROOT}/00_smoke_32p"

for seed in ${SEEDS}; do
  phase_root="${EXPERIMENT_ROOT}/00_smoke_32p/seed_${seed}"
  CONFIG="${CONFIG}" RUN_ROOT="${phase_root}" SOURCE_ROOT="${SOURCE_ROOT}" LOG_DIR="${EXPERIMENT_LOG_ROOT}/00_smoke_32p/seed_${seed}" \
    STAGE_SEED="${seed}" STAGE_FINAL_SEEDS="${seed}" DATASETS_OVERRIDE="gsm8k math500" \
    STAGE_FINAL_EVAL_LIMIT="${STAGE_BUDGET_V41_SMOKE_LIMIT:-128}" STAGE_FINAL_METHODS="${METHODS}" \
    bash scripts/run_t30_math_safe_priority_suite_8gpu.sh
done

echo "Stage-switch-only ratio ablation completed. Compare the two method summaries under ${EXPERIMENT_ROOT}/00_smoke_32p."
