#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

PYTHON_BIN="${PYTHON:-/home/cike/jjy/envs/rasp_qwen3_eval/bin/python}"
HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
LOG_DIR="${LOG_DIR:-logs}"
FINAL_METHODS="${STAGE_FINAL_METHODS:-fixed_global_t30_ratios}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
SEED="${STAGE_SEED:-3}"

SOURCE_ROOT="${SOURCE_ROOT:-runs/08_stage_calibrated_pruning/main_pilot_mixed_reasoning_seed3}"
PRIORITY_CONFIG="${PRIORITY_CONFIG:-configs/stage_calibrated_pruning/fixed_global_t30_baseline_qwen3_1p7b_priority_suite.yaml}"
FULL_CONFIG="${FULL_CONFIG:-configs/stage_calibrated_pruning/fixed_global_t30_baseline_qwen3_1p7b_full.yaml}"
PRIORITY_ROOT="${PRIORITY_ROOT:-runs/08_stage_calibrated_pruning/main_pilot_fixed_global_t30_baseline_qwen3_1p7b_priority_suite}"
FULL_ROOT="${FULL_ROOT:-runs/08_stage_calibrated_pruning/main_pilot_fixed_global_t30_baseline_qwen3_1p7b_full}"

for artifact_dir in 03_selected 04_masks; do
  if [[ ! -d "${SOURCE_ROOT}/${artifact_dir}" ]]; then
    echo "Missing reusable artifact: ${SOURCE_ROOT}/${artifact_dir}" >&2
    echo "Set SOURCE_ROOT to an existing mixed pilot run that contains 03_selected and 04_masks." >&2
    exit 2
  fi
done

mkdir -p "${LOG_DIR}"
export SOURCE_ROOT

echo "==== START Fixed-Global t30 priority suite: ${FINAL_METHODS} ===="
PYTHON="${PYTHON_BIN}" \
CONFIG="${PRIORITY_CONFIG}" \
RUN_ROOT="${PRIORITY_ROOT}" \
HF_ENDPOINT="${HF_ENDPOINT}" \
STAGE_SEED="${SEED}" \
STAGE_FINAL_METHODS="${FINAL_METHODS}" \
LOG_PREFIX="fixed_global_t30_qwen3_1p7b_priority" \
SKIP_EXISTING="${SKIP_EXISTING}" \
bash scripts/run_t30_math_safe_priority_suite_8gpu.sh

echo "==== START Fixed-Global t30 full math: ${FINAL_METHODS} ===="
PYTHON="${PYTHON_BIN}" \
CONFIG="${FULL_CONFIG}" \
RUN_ROOT="${FULL_ROOT}" \
HF_ENDPOINT="${HF_ENDPOINT}" \
STAGE_SEED="${SEED}" \
STAGE_FINAL_METHODS="${FINAL_METHODS}" \
RUN_LABEL="fixed_global_t30_qwen3_1p7b_full" \
SKIP_EXISTING="${SKIP_EXISTING}" \
bash scripts/run_t30_math_safe_full_8gpu.sh

echo "==== ALL DONE Fixed-Global t30 baseline ===="
echo "Priority summary: ${PRIORITY_ROOT}/aggregate_summary.md"
echo "Full summary: ${FULL_ROOT}/06_final/summary.json"
