#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

PYTHON_BIN="${PYTHON:-/home/cike/jjy/envs/rasp_qwen3_eval/bin/python}"
RUN_ROOT="${RUN_ROOT:-runs/10_stage_budget_output_aware_v1}"
LOG_ROOT="${LOG_ROOT:-logs/10_stage_budget_output_aware_v1}"
SOURCE_ROOT="${SOURCE_ROOT:-runs/08_stage_calibrated_pruning/main_pilot_mixed_reasoning_seed3_t30_math_safe_full}"
CONFIG="${CONFIG:-configs/generated_stage_budget_output_aware_v1/budget_v3_smoke.yaml}"
METHODS="dynamic_global_activation_budgeted_v3_calibrated,dynamic_global_activation_budgeted_v3_recovery"
SEEDS="${STAGE_BUDGET_SEEDS:-3}"

bash scripts/prepare_stage_budget_output_aware_v1.sh

for seed in ${SEEDS}; do
  phase_root="${RUN_ROOT}/08_budget_v3_actual_calibrated/00_smoke_actual_pruning/seed_${seed}"
  CONFIG="${CONFIG}" RUN_ROOT="${phase_root}" SOURCE_ROOT="${SOURCE_ROOT}" LOG_DIR="${LOG_ROOT}/08_budget_v3_actual_calibrated/00_smoke_actual_pruning/seed_${seed}" \
    STAGE_SEED="${seed}" STAGE_FINAL_SEEDS="${seed}" DATASETS_OVERRIDE="gsm8k math500" STAGE_FINAL_EVAL_LIMIT="${STAGE_BUDGET_V3_SMOKE_LIMIT:-128}" STAGE_FINAL_METHODS="${METHODS}" \
    bash scripts/run_t30_math_safe_priority_suite_8gpu.sh
done

"${PYTHON_BIN}" -m src.main_select_stage_budget_output_aware \
  --phase a2 \
  --phase-label A3_budget_v3 \
  --selection-mode smoke \
  --phase-root "${RUN_ROOT}/08_budget_v3_actual_calibrated/00_smoke_actual_pruning" \
  --reference-root "${RUN_ROOT}/01_budget_only_dev" \
  --baseline-method dynamic_global_activation_fixed_t30 \
  --candidate-methods dynamic_global_activation_budgeted_v3_calibrated dynamic_global_activation_budgeted_v3_recovery \
  --output "${RUN_ROOT}/08_budget_v3_actual_calibrated/00_smoke_actual_pruning/budget_v3_smoke_selection.json"
