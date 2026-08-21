#!/usr/bin/env bash
set -euo pipefail

unset HF_DATASETS_OFFLINE
unset TRANSFORMERS_OFFLINE
unset HF_HUB_OFFLINE

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

PYTHON_BIN="${PYTHON:-/home/cike/jjy/envs/rasp_qwen3_eval/bin/python}"
HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
SEED="${STAGE_SEED:-3}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
METHOD="shortgpt_t20_matched"

FULL_CONFIG="${FULL_CONFIG:-configs/generated_additional_baselines/shortgpt_t20_qwen3_1p7b_full.yaml}"
PRIORITY_CONFIG="${PRIORITY_CONFIG:-configs/generated_additional_baselines/shortgpt_t20_qwen3_1p7b_priority_suite.yaml}"
RUN_ROOT="${RUN_ROOT:-runs/12_additional_baselines/02_shortgpt_t20}"
LOG_ROOT="${LOG_ROOT:-logs/12_additional_baselines/02_shortgpt_t20}"
SOURCE_ROOT="${SOURCE_ROOT:-runs/08_stage_calibrated_pruning/main_pilot_mixed_reasoning_seed3}"

FULL_ROOT="${FULL_ROOT:-${RUN_ROOT}/01_full_qwen3_1p7b}"
PRIORITY_ROOT="${PRIORITY_ROOT:-${RUN_ROOT}/02_priority_suite_qwen3_1p7b}"
FULL_LOG_DIR="${FULL_LOG_DIR:-${LOG_ROOT}/01_full_qwen3_1p7b}"
PRIORITY_LOG_DIR="${PRIORITY_LOG_DIR:-${LOG_ROOT}/02_priority_suite_qwen3_1p7b}"
PRUNED_LAYERS_PATH="${PRUNED_LAYERS_PATH:-${RUN_ROOT}/00_preflight/shortgpt_t20_pruned_layers.json}"

if [[ ! -f "${SOURCE_ROOT}/03_selected/calibration.jsonl" ]]; then
  echo "Missing ShortGPT calibration file: ${SOURCE_ROOT}/03_selected/calibration.jsonl" >&2
  echo "Set SOURCE_ROOT to an existing mixed pilot run with 03_selected/calibration.jsonl." >&2
  exit 2
fi

mkdir -p \
  "${RUN_ROOT}/00_preflight" \
  "${FULL_ROOT}" \
  "${PRIORITY_ROOT}" \
  "${LOG_ROOT}/00_preflight" \
  "${FULL_LOG_DIR}" \
  "${PRIORITY_LOG_DIR}"

echo "==== ShortGPT T20 Qwen3-1.7B full + priority baseline ===="
echo "method=${METHOD}"
echo "source_root=${SOURCE_ROOT}"
echo "pruned_layers_path=${PRUNED_LAYERS_PATH}"
echo "full_root=${FULL_ROOT}"
echo "priority_root=${PRIORITY_ROOT}"
echo "gpus=${FINAL_GPUS:-0 1 2 3 4 5 6 7}"
echo "priority_datasets=${DATASETS_OVERRIDE:-amc2023 gpqa_diamond arc_challenge}"

if [[ "${FORCE_SHORTGPT_PRECOMPUTE:-0}" == "1" || ! -f "${PRUNED_LAYERS_PATH}" ]]; then
  echo "==== START ShortGPT T20 Block Influence precompute ===="
  CUDA_VISIBLE_DEVICES="${SHORTGPT_PRECOMPUTE_GPU:-0}" \
  HF_ENDPOINT="${HF_ENDPOINT}" \
  HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}" \
  HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-120}" \
  HF_HUB_ETAG_TIMEOUT="${HF_HUB_ETAG_TIMEOUT:-120}" \
  SOURCE_ROOT="${SOURCE_ROOT}" \
  "${PYTHON_BIN}" -m src.baselines.prepare_shortgpt_pruned_layers \
    --config "${FULL_CONFIG}" \
    --method "${METHOD}" \
    --output "${PRUNED_LAYERS_PATH}" \
    > "${LOG_ROOT}/00_preflight/shortgpt_t20_precompute.log" 2>&1
  echo "DONE ShortGPT T20 precompute: ${PRUNED_LAYERS_PATH}"
else
  echo "SKIP ShortGPT T20 precompute; existing ${PRUNED_LAYERS_PATH}"
fi

echo "==== START ShortGPT T20 full GSM8K + MATH500 ===="
PYTHON="${PYTHON_BIN}" \
CONFIG="${FULL_CONFIG}" \
RUN_ROOT="${FULL_ROOT}" \
SOURCE_ROOT="${SOURCE_ROOT}" \
HF_ENDPOINT="${HF_ENDPOINT}" \
STAGE_SEED="${SEED}" \
STAGE_FINAL_SEEDS="${SEED}" \
STAGE_FINAL_METHODS="${METHOD}" \
STAGE_FINAL_EVAL_LIMIT="${STAGE_FINAL_EVAL_LIMIT:--1}" \
SKIP_EXISTING="${SKIP_EXISTING}" \
LOG_DIR="${FULL_LOG_DIR}" \
RUN_LABEL="shortgpt_t20_qwen3_1p7b_full" \
bash scripts/run_griffin_prompt_matched_full_gpu.sh

echo "==== START ShortGPT T20 priority suite ===="
PYTHON="${PYTHON_BIN}" \
CONFIG="${PRIORITY_CONFIG}" \
RUN_ROOT="${PRIORITY_ROOT}" \
SOURCE_ROOT="${SOURCE_ROOT}" \
HF_ENDPOINT="${HF_ENDPOINT}" \
STAGE_SEED="${SEED}" \
STAGE_FINAL_SEEDS="${SEED}" \
STAGE_FINAL_METHODS="${METHOD}" \
STAGE_FINAL_EVAL_LIMIT="${STAGE_PRIORITY_EVAL_LIMIT:--1}" \
SKIP_EXISTING="${SKIP_EXISTING}" \
LOG_DIR="${PRIORITY_LOG_DIR}" \
LOG_PREFIX="shortgpt_t20_qwen3_1p7b_priority" \
BASELINE_LABEL="ShortGPT T20" \
BASELINE_SCHEMA="shortgpt_t20_priority_suite_aggregate_v1" \
DATASETS_OVERRIDE="${DATASETS_OVERRIDE:-amc2023 gpqa_diamond arc_challenge}" \
bash scripts/run_griffin_prompt_priority_suite_gpu.sh

echo "==== ALL DONE ShortGPT T20 Qwen3-1.7B ===="
echo "Full summary: ${FULL_ROOT}/06_final/summary.json"
echo "Priority summary: ${PRIORITY_ROOT}/aggregate_summary.md"

