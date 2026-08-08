#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

PYTHON_BIN="${PYTHON:-/home/cike/jjy/envs/rasp_qwen3_eval/bin/python}"
RUN_ROOT="${RUN_ROOT:-runs/10_stage_budget_output_aware_v1}"
LOG_ROOT="${LOG_ROOT:-logs/10_stage_budget_output_aware_v1}"
SOURCE_ROOT="${SOURCE_ROOT:-runs/08_stage_calibrated_pruning/main_pilot_mixed_reasoning_qwen3_4b_seed3_t30_math_safe_full}"
FULL_CONFIG="${FULL_CONFIG:-configs/generated_stage_budget_output_aware_v1/budget_v41_qwen3_4b_full.template.yaml}"
PRIORITY_CONFIG="${PRIORITY_CONFIG:-configs/generated_stage_budget_output_aware_v1/budget_v41_qwen3_4b_priority_suite.template.yaml}"
SEED="${STAGE_SEED:-3}"
METHODS="${STAGE_FINAL_METHODS:-dynamic_global_activation_fixed_t30,dynamic_global_activation_budgeted_v41_stage_lift_plus_32p}"

for artifact_dir in 03_selected 04_masks; do
  if [[ ! -d "${SOURCE_ROOT}/${artifact_dir}" ]]; then
    echo "Missing Qwen3-4B reusable artifact: ${SOURCE_ROOT}/${artifact_dir}" >&2
    echo "Set SOURCE_ROOT to the existing Qwen3-4B pilot run with 03_selected and 04_masks." >&2
    exit 2
  fi
done

bash scripts/prepare_stage_budget_output_aware_v1.sh

transfer_root="${RUN_ROOT}/10_budget_v41_stage_lift_32p/04_qwen3_4b_transfer"
transfer_log_root="${LOG_ROOT}/10_budget_v41_stage_lift_32p/04_qwen3_4b_transfer"
mkdir -p "${transfer_root}" "${transfer_log_root}"

full_root="${transfer_root}/01_full_seed_${SEED}"
full_log_dir="${transfer_log_root}/01_full_seed_${SEED}"
mkdir -p "${full_root}" "${full_log_dir}"
cat > "${full_root}/status.json" <<EOF
{
  "status": "running",
  "mode": "qwen3_4b_transfer_full",
  "seed": ${SEED},
  "datasets": ["gsm8k", "math500"],
  "methods": ["dynamic_global_activation_fixed_t30", "dynamic_global_activation_budgeted_v41_stage_lift_plus_32p"]
}
EOF

CONFIG="${FULL_CONFIG}" RUN_ROOT="${full_root}" SOURCE_ROOT="${SOURCE_ROOT}" LOG_DIR="${full_log_dir}" \
  STAGE_SEED="${SEED}" STAGE_FINAL_SEEDS="${SEED}" DATASETS_OVERRIDE="gsm8k math500" \
  STAGE_FINAL_EVAL_LIMIT="${STAGE_FINAL_EVAL_LIMIT:--1}" STAGE_FINAL_METHODS="${METHODS}" \
  bash scripts/run_t30_math_safe_priority_suite_8gpu.sh

cat > "${full_root}/status.json" <<EOF
{
  "status": "completed",
  "mode": "qwen3_4b_transfer_full",
  "seed": ${SEED},
  "datasets": ["gsm8k", "math500"],
  "methods": ["dynamic_global_activation_fixed_t30", "dynamic_global_activation_budgeted_v41_stage_lift_plus_32p"]
}
EOF

priority_root="${transfer_root}/02_priority_suite_seed_${SEED}"
priority_log_dir="${transfer_log_root}/02_priority_suite_seed_${SEED}"
mkdir -p "${priority_root}" "${priority_log_dir}"
cat > "${priority_root}/status.json" <<EOF
{
  "status": "running",
  "mode": "qwen3_4b_transfer_priority_suite",
  "seed": ${SEED},
  "datasets": ["aime2024", "aime2025", "amc2023", "gpqa_diamond", "arc_challenge"],
  "methods": ["dynamic_global_activation_fixed_t30", "dynamic_global_activation_budgeted_v41_stage_lift_plus_32p"]
}
EOF

CONFIG="${PRIORITY_CONFIG}" RUN_ROOT="${priority_root}" SOURCE_ROOT="${SOURCE_ROOT}" LOG_DIR="${priority_log_dir}" \
  STAGE_SEED="${SEED}" STAGE_FINAL_SEEDS="${SEED}" DATASETS_OVERRIDE="aime2024 aime2025 amc2023 gpqa_diamond arc_challenge" \
  STAGE_FINAL_EVAL_LIMIT="${STAGE_PRIORITY_EVAL_LIMIT:--1}" STAGE_FINAL_METHODS="${METHODS}" \
  bash scripts/run_t30_math_safe_priority_suite_8gpu.sh

cat > "${priority_root}/status.json" <<EOF
{
  "status": "completed",
  "mode": "qwen3_4b_transfer_priority_suite",
  "seed": ${SEED},
  "datasets": ["aime2024", "aime2025", "amc2023", "gpqa_diamond", "arc_challenge"],
  "methods": ["dynamic_global_activation_fixed_t30", "dynamic_global_activation_budgeted_v41_stage_lift_plus_32p"]
}
EOF
