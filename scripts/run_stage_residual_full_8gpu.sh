#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

PYTHON_BIN="${PYTHON:-/home/cike/jjy/envs/rasp_qwen3_eval/bin/python}"
RUN_ROOT="${RUN_ROOT:-runs/09_stage_residual_rasp_v2}"
LOG_ROOT="${LOG_ROOT:-logs/09_stage_residual_rasp_v2}"
SOURCE_ROOT="${SOURCE_ROOT:-runs/08_stage_calibrated_pruning/main_pilot_mixed_reasoning_seed3_t30_math_safe_full}"
CONFIG="${CONFIG:-configs/generated_stage_residual_rasp_v2/stage_residual_output_aware_frozen.yaml}"
SEEDS="${STAGE_RESIDUAL_SEEDS:-1 2 3}"
SELECTION="${RUN_ROOT}/03_output_aware_continuity_dev/final_method_selection.json"

selected_method="$(${PYTHON_BIN} -c 'import json, sys; row=json.load(open(sys.argv[1], encoding="utf-8"))["selection"]; assert row["status"] == "passed", row; print(row["method"])' "${SELECTION}")"
methods="structured_dense,strict_static_matched_t30,current_t30_math_safe_activation,dynamic_global_activation,${selected_method}"

bash scripts/prepare_stage_residual_rasp_v2.sh
for seed in ${SEEDS}; do
  seed_root="${RUN_ROOT}/04_frozen_full_qwen3_1p7b/seed_${seed}"
  if [[ ! -f "${seed_root}/04_masks/stage_residual_bank_manifest.json" ]]; then
    STAGE_SEED="${seed}" STAGE_WORKFLOW_ROOT="${seed_root}" "${PYTHON_BIN}" -m src.main_stage_calibrated_pruning --config "${CONFIG}" --profile pilot --stage preflight --force
    "${PYTHON_BIN}" -m src.main_prepare_stage_residual_bank --source-root "${SOURCE_ROOT}" --target-root "${seed_root}"
  fi
  CONFIG="${CONFIG}" RUN_ROOT="${seed_root}" SOURCE_ROOT="${seed_root}" LOG_DIR="${LOG_ROOT}/04_full/seed_${seed}" \
    STAGE_SEED="${seed}" DATASETS_OVERRIDE="gsm8k math500" STAGE_FINAL_EVAL_LIMIT=-1 STAGE_FINAL_METHODS="${methods}" \
    bash scripts/run_t30_math_safe_priority_suite_8gpu.sh
done
printf '{"status":"completed","selected_method":"%s"}\n' "${selected_method}" > "${RUN_ROOT}/04_frozen_full_qwen3_1p7b/status.json"
