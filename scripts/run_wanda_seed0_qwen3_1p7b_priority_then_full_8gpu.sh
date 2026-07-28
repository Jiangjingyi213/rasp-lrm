#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

PYTHON_BIN="${PYTHON:-/home/cike/jjy/envs/rasp_qwen3_eval/bin/python}"
HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
LOG_DIR="${LOG_DIR:-logs/wanda_seed0_t30}"
FINAL_METHODS="${STAGE_FINAL_METHODS:-wanda_t30_official}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
SEED="${STAGE_SEED:-0}"

SOURCE_ROOT="${SOURCE_ROOT:-runs/08_stage_calibrated_pruning/main_pilot_mixed_reasoning_seed3}"
PRIORITY_CONFIG="${PRIORITY_CONFIG:-configs/stage_calibrated_pruning/wanda_official_seed0_qwen3_1p7b_priority_suite.yaml}"
FULL_CONFIG="${FULL_CONFIG:-configs/stage_calibrated_pruning/wanda_official_seed0_qwen3_1p7b_full.yaml}"
PRIORITY_ROOT="${PRIORITY_ROOT:-runs/08_stage_calibrated_pruning/main_pilot_wanda_official_seed0_qwen3_1p7b_priority_suite}"
FULL_ROOT="${FULL_ROOT:-runs/08_stage_calibrated_pruning/main_pilot_wanda_official_seed0_qwen3_1p7b_full}"

if [[ "${FINAL_METHODS}" == *","* ]]; then
  echo "Wanda official evaluation supports one method per run; got STAGE_FINAL_METHODS=${FINAL_METHODS}" >&2
  echo "Run this script once with wanda_t30_official and once with wanda_t20_official." >&2
  exit 2
fi

if [[ ! -f "${SOURCE_ROOT}/03_selected/calibration.jsonl" ]]; then
  echo "Missing mixed calibration file: ${SOURCE_ROOT}/03_selected/calibration.jsonl" >&2
  echo "Set SOURCE_ROOT to an existing mixed pilot run with 03_selected/calibration.jsonl." >&2
  exit 2
fi

mkdir -p "${LOG_DIR}"
export SOURCE_ROOT

echo "==== START Wanda seed0 priority suite: ${FINAL_METHODS} ===="
PYTHON="${PYTHON_BIN}" \
CONFIG="${PRIORITY_CONFIG}" \
RUN_ROOT="${PRIORITY_ROOT}" \
HF_ENDPOINT="${HF_ENDPOINT}" \
STAGE_SEED="${SEED}" \
STAGE_FINAL_METHODS="${FINAL_METHODS}" \
LOG_DIR="${LOG_DIR}" \
LOG_PREFIX="wanda_seed0_qwen3_1p7b_priority" \
BASELINE_LABEL="Wanda Seed0" \
BASELINE_SCHEMA="wanda_seed0_priority_suite_aggregate_v1" \
SKIP_EXISTING="${SKIP_EXISTING}" \
bash scripts/run_griffin_prompt_priority_suite_gpu.sh

echo "==== START Wanda seed0 full math: ${FINAL_METHODS} ===="
PYTHON="${PYTHON_BIN}" \
CONFIG="${FULL_CONFIG}" \
RUN_ROOT="${FULL_ROOT}" \
HF_ENDPOINT="${HF_ENDPOINT}" \
STAGE_SEED="${SEED}" \
STAGE_FINAL_METHODS="${FINAL_METHODS}" \
LOG_DIR="${LOG_DIR}" \
RUN_LABEL="wanda_seed0_qwen3_1p7b_full" \
SKIP_EXISTING="${SKIP_EXISTING}" \
bash scripts/run_griffin_prompt_matched_full_gpu.sh

echo "==== ALL DONE Wanda seed0 ${FINAL_METHODS} ===="
echo "Priority summary: ${PRIORITY_ROOT}/aggregate_summary.md"
echo "Full summary: ${FULL_ROOT}/06_final/summary.json"
