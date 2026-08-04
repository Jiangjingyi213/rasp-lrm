#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

PYTHON_BIN="${PYTHON:-/home/cike/jjy/envs/rasp_qwen3_eval/bin/python}"
RUN_ROOT="${RUN_ROOT:-runs/10_stage_budget_output_aware_v1}"
LOG_ROOT="${LOG_ROOT:-logs/10_stage_budget_output_aware_v1}"
SOURCE_ROOT="${SOURCE_ROOT:-runs/08_stage_calibrated_pruning/main_pilot_mixed_reasoning_seed3_t30_math_safe_full}"
CONFIG="${CONFIG:-configs/generated_stage_budget_output_aware_v1/budget_v41_smoke.yaml}"
METHODS="dynamic_global_activation_budgeted_v41_stage_lift_32p,dynamic_global_activation_budgeted_v41_stage_lift_plus_32p"
SEEDS="${STAGE_BUDGET_SEEDS:-3}"

bash scripts/prepare_stage_budget_output_aware_v1.sh

for seed in ${SEEDS}; do
  phase_root="${RUN_ROOT}/10_budget_v41_stage_lift_32p/00_smoke_32p/seed_${seed}"
  CONFIG="${CONFIG}" RUN_ROOT="${phase_root}" SOURCE_ROOT="${SOURCE_ROOT}" LOG_DIR="${LOG_ROOT}/10_budget_v41_stage_lift_32p/00_smoke_32p/seed_${seed}" \
    STAGE_SEED="${seed}" STAGE_FINAL_SEEDS="${seed}" DATASETS_OVERRIDE="gsm8k math500" STAGE_FINAL_EVAL_LIMIT="${STAGE_BUDGET_V41_SMOKE_LIMIT:-128}" STAGE_FINAL_METHODS="${METHODS}" \
    bash scripts/run_t30_math_safe_priority_suite_8gpu.sh
done

"${PYTHON_BIN}" -m src.main_select_stage_budget_output_aware \
  --phase a2 \
  --phase-label A41_budget_stage_lift_32p \
  --selection-mode smoke \
  --target-min 0.320 \
  --target-max 0.340 \
  --target-center 0.330 \
  --max-fallback-delta 0.02 \
  --max-truncation-delta 0.02 \
  --phase-root "${RUN_ROOT}/10_budget_v41_stage_lift_32p/00_smoke_32p" \
  --reference-root "${RUN_ROOT}/01_budget_only_dev" \
  --baseline-method dynamic_global_activation_fixed_t30 \
  --candidate-methods dynamic_global_activation_budgeted_v41_stage_lift_32p dynamic_global_activation_budgeted_v41_stage_lift_plus_32p \
  --output "${RUN_ROOT}/10_budget_v41_stage_lift_32p/00_smoke_32p/budget_v41_smoke_selection.json"
