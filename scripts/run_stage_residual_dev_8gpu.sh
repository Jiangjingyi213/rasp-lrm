#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

PYTHON_BIN="${PYTHON:-/home/cike/jjy/envs/rasp_qwen3_eval/bin/python}"
CONFIG="${CONFIG:-configs/generated_stage_residual_rasp_v2/stage_residual_dev.yaml}"
RUN_ROOT="${RUN_ROOT:-runs/09_stage_residual_rasp_v2}"
LOG_ROOT="${LOG_ROOT:-logs/09_stage_residual_rasp_v2}"
SOURCE_ROOT="${SOURCE_ROOT:-runs/08_stage_calibrated_pruning/main_pilot_mixed_reasoning_seed3_t30_math_safe_full}"
SEEDS="${STAGE_RESIDUAL_SEEDS:-1 2 3}"
METHODS="structured_dense,strict_static_matched_t30,dynamic_global_activation,dynamic_stage_specific_activation,dynamic_stage_residual_025_activation,dynamic_stage_residual_050_activation,dynamic_shuffled_prior_activation"

bash scripts/prepare_stage_residual_rasp_v2.sh
"${PYTHON_BIN}" -m src.main_stage_residual_evidence_audit \
  --aggregate runs/08_stage_calibrated_pruning/main_pilot_mixed_reasoning_seed3_t30_core_ablation/aggregate_summary.json \
  --output-dir "${RUN_ROOT}/01_existing_evidence_audit"

for seed in ${SEEDS}; do
  seed_root="${RUN_ROOT}/02_stage_residual_dev/seed_${seed}"
  if [[ ! -f "${seed_root}/04_masks/stage_residual_bank_manifest.json" ]]; then
    STAGE_SEED="${seed}" STAGE_WORKFLOW_ROOT="${seed_root}" "${PYTHON_BIN}" -m src.main_stage_calibrated_pruning --config "${CONFIG}" --profile pilot --stage preflight --force
    "${PYTHON_BIN}" -m src.main_prepare_stage_residual_bank --source-root "${SOURCE_ROOT}" --target-root "${seed_root}"
  fi
  CONFIG="${CONFIG}" RUN_ROOT="${seed_root}" SOURCE_ROOT="${seed_root}" LOG_DIR="${LOG_ROOT}/02_dev/seed_${seed}" \
    STAGE_SEED="${seed}" DATASETS_OVERRIDE="gsm8k math500" STAGE_FINAL_EVAL_LIMIT=520 STAGE_FINAL_METHODS="${METHODS}" \
    bash scripts/run_t30_math_safe_priority_suite_8gpu.sh
done

"${PYTHON_BIN}" -m src.main_select_stage_residual_policy \
  --phase-root "${RUN_ROOT}/02_stage_residual_dev" \
  --output "${RUN_ROOT}/02_stage_residual_dev/residual_selection.json"
printf '{"status":"completed","artifact":"residual_selection.json"}\n' > "${RUN_ROOT}/02_stage_residual_dev/status.json"
