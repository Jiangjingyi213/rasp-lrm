#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

PYTHON_BIN="${PYTHON:-/home/cike/jjy/envs/rasp_qwen3_eval/bin/python}"
LOG_DIR="${LOG_DIR:-logs}"
HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
FINAL_METHODS="${STAGE_FINAL_METHODS:-wanda_c4_seed0_t30}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
SEED="${STAGE_SEED:-0}"

C4_CALIBRATION_PATH="${C4_CALIBRATION_PATH:-runs/08_stage_calibrated_pruning/wanda_c4_seed0_calibration/c4_128_seed0.jsonl}"
C4_CALIBRATION_SAMPLES="${C4_CALIBRATION_SAMPLES:-128}"
C4_CALIBRATION_MIN_CHARS="${C4_CALIBRATION_MIN_CHARS:-64}"
C4_CALIBRATION_BUFFER_SIZE="${C4_CALIBRATION_BUFFER_SIZE:-10000}"

PRIORITY_CONFIG="${PRIORITY_CONFIG:-configs/stage_calibrated_pruning/wanda_official_c4_seed0_qwen3_1p7b_priority_suite.yaml}"
FULL_CONFIG="${FULL_CONFIG:-configs/stage_calibrated_pruning/wanda_official_c4_seed0_qwen3_1p7b_full.yaml}"
PRIORITY_ROOT="${PRIORITY_ROOT:-runs/08_stage_calibrated_pruning/main_pilot_wanda_official_c4_seed0_qwen3_1p7b_priority_suite}"
FULL_ROOT="${FULL_ROOT:-runs/08_stage_calibrated_pruning/main_pilot_wanda_official_c4_seed0_qwen3_1p7b_full}"

if [[ "${FINAL_METHODS}" == *","* ]]; then
  echo "Wanda official evaluation supports one method per run; got STAGE_FINAL_METHODS=${FINAL_METHODS}" >&2
  echo "Run this script once with wanda_c4_seed0_t30 and once with wanda_c4_seed0_t20." >&2
  exit 2
fi

mkdir -p "${LOG_DIR}" "$(dirname "${C4_CALIBRATION_PATH}")"

needs_c4=1
if [[ -f "${C4_CALIBRATION_PATH}" ]]; then
  line_count="$(wc -l < "${C4_CALIBRATION_PATH}" | tr -d ' ')"
  if [[ "${line_count}" == "${C4_CALIBRATION_SAMPLES}" ]]; then
    needs_c4=0
    echo "SKIP C4 calibration artifact; existing ${C4_CALIBRATION_PATH} has ${line_count} rows."
  else
    echo "Regenerating C4 calibration artifact; ${C4_CALIBRATION_PATH} has ${line_count} rows, expected ${C4_CALIBRATION_SAMPLES}."
  fi
fi

if [[ "${needs_c4}" == "1" ]]; then
  echo "START build C4 calibration artifact: ${C4_CALIBRATION_PATH}"
  HF_ENDPOINT="${HF_ENDPOINT}" \
  HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}" \
  HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-120}" \
  HF_HUB_ETAG_TIMEOUT="${HF_HUB_ETAG_TIMEOUT:-120}" \
  "${PYTHON_BIN}" -m src.data.build_c4_calibration \
    --output "${C4_CALIBRATION_PATH}" \
    --samples "${C4_CALIBRATION_SAMPLES}" \
    --seed "${SEED}" \
    --min-chars "${C4_CALIBRATION_MIN_CHARS}" \
    --buffer-size "${C4_CALIBRATION_BUFFER_SIZE}"
  echo "DONE build C4 calibration artifact"
fi

echo "==== START Wanda-C4 Seed0 priority suite: ${FINAL_METHODS} ===="
PYTHON="${PYTHON_BIN}" \
CONFIG="${PRIORITY_CONFIG}" \
RUN_ROOT="${PRIORITY_ROOT}" \
HF_ENDPOINT="${HF_ENDPOINT}" \
STAGE_SEED="${SEED}" \
STAGE_FINAL_METHODS="${FINAL_METHODS}" \
LOG_PREFIX="wanda_c4_seed0_qwen3_1p7b_priority" \
BASELINE_LABEL="Wanda-C4 Seed0" \
BASELINE_SCHEMA="wanda_c4_seed0_priority_suite_aggregate_v1" \
SKIP_EXISTING="${SKIP_EXISTING}" \
bash scripts/run_griffin_prompt_priority_suite_gpu.sh

echo "==== START Wanda-C4 Seed0 full math: ${FINAL_METHODS} ===="
PYTHON="${PYTHON_BIN}" \
CONFIG="${FULL_CONFIG}" \
RUN_ROOT="${FULL_ROOT}" \
HF_ENDPOINT="${HF_ENDPOINT}" \
STAGE_SEED="${SEED}" \
STAGE_FINAL_METHODS="${FINAL_METHODS}" \
RUN_LABEL="wanda_c4_seed0_qwen3_1p7b_full" \
SKIP_EXISTING="${SKIP_EXISTING}" \
bash scripts/run_griffin_prompt_matched_full_gpu.sh

echo "==== ALL DONE Wanda-C4 Seed0 ${FINAL_METHODS} ===="
echo "Priority summary: ${PRIORITY_ROOT}/aggregate_summary.md"
echo "Full summary: ${FULL_ROOT}/06_final/summary.json"
