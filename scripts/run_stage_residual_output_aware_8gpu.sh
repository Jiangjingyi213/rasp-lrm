#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

PYTHON_BIN="${PYTHON:-/home/cike/jjy/envs/rasp_qwen3_eval/bin/python}"
RUN_ROOT="${RUN_ROOT:-runs/09_stage_residual_rasp_v2}"
LOG_ROOT="${LOG_ROOT:-logs/09_stage_residual_rasp_v2}"
SOURCE_ROOT="${SOURCE_ROOT:-runs/08_stage_calibrated_pruning/main_pilot_mixed_reasoning_seed3_t30_math_safe_full}"
SEEDS="${STAGE_RESIDUAL_SEEDS:-3}"
CONFIG="configs/generated_stage_residual_rasp_v2/stage_residual_output_aware_frozen.yaml"
METHODS="selected_residual_activation,selected_residual_output_aware,selected_residual_output_aware_continuity,dynamic_global_output_aware"

bash scripts/prepare_stage_residual_rasp_v2.sh
"${PYTHON_BIN}" -m src.main_render_stage_residual_config \
  --template configs/generated_stage_residual_rasp_v2/stage_residual_output_aware.template.yaml \
  --selection "${RUN_ROOT}/02_stage_residual_dev/residual_selection.json" \
  --output "${CONFIG}"

for seed in ${SEEDS}; do
  seed_root="${RUN_ROOT}/03_output_aware_continuity_dev/seed_${seed}"
  if [[ ! -f "${seed_root}/04_masks/stage_residual_bank_manifest.json" ]]; then
    STAGE_SEED="${seed}" STAGE_WORKFLOW_ROOT="${seed_root}" "${PYTHON_BIN}" -m src.main_stage_calibrated_pruning --config "${CONFIG}" --profile pilot --stage preflight --force
    "${PYTHON_BIN}" -m src.main_prepare_stage_residual_bank --source-root "${SOURCE_ROOT}" --target-root "${seed_root}"
  fi
  CONFIG="${CONFIG}" RUN_ROOT="${seed_root}" SOURCE_ROOT="${seed_root}" LOG_DIR="${LOG_ROOT}/03_dev/seed_${seed}" \
    STAGE_SEED="${seed}" STAGE_FINAL_SEEDS="${seed}" DATASETS_OVERRIDE="gsm8k math500" STAGE_FINAL_EVAL_LIMIT=520 STAGE_FINAL_METHODS="${METHODS}" \
    bash scripts/run_t30_math_safe_priority_suite_8gpu.sh
done

"${PYTHON_BIN}" -m src.main_select_stage_residual_policy \
  --phase-root "${RUN_ROOT}/03_output_aware_continuity_dev" \
  --baseline selected_residual_activation \
  --candidates selected_residual_output_aware selected_residual_output_aware_continuity \
  --output "${RUN_ROOT}/03_output_aware_continuity_dev/final_method_selection.json"
printf '{"status":"completed","artifact":"final_method_selection.json"}\n' > "${RUN_ROOT}/03_output_aware_continuity_dev/status.json"
