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

CONFIG="${CONFIG:-configs/generated_additional_baselines/shortgpt_t20_qwen3_1p7b_priority_suite.yaml}"
FULL_CONFIG="${FULL_CONFIG:-configs/generated_additional_baselines/shortgpt_t20_qwen3_1p7b_full.yaml}"
RUN_ROOT="${RUN_ROOT:-runs/12_additional_baselines/02_shortgpt_t20}"
LOG_ROOT="${LOG_ROOT:-logs/12_additional_baselines/02_shortgpt_t20}"
SOURCE_ROOT="${SOURCE_ROOT:-runs/08_stage_calibrated_pruning/main_pilot_mixed_reasoning_seed3}"
PRIORITY_ROOT="${PRIORITY_ROOT:-${RUN_ROOT}/02_priority_suite_qwen3_1p7b}"
PRIORITY_LOG_DIR="${PRIORITY_LOG_DIR:-${LOG_ROOT}/02_priority_suite_qwen3_1p7b}"
PRUNED_LAYERS_PATH="${PRUNED_LAYERS_PATH:-${RUN_ROOT}/00_preflight/shortgpt_t20_pruned_layers.json}"

if [[ ! -f "${SOURCE_ROOT}/03_selected/calibration.jsonl" ]]; then
  echo "Missing ShortGPT calibration file: ${SOURCE_ROOT}/03_selected/calibration.jsonl" >&2
  echo "Set SOURCE_ROOT to an existing mixed pilot run with 03_selected/calibration.jsonl." >&2
  exit 2
fi

mkdir -p "${RUN_ROOT}/00_preflight" "${PRIORITY_ROOT}" "${LOG_ROOT}/00_preflight" "${PRIORITY_LOG_DIR}"

echo "==== ShortGPT T20 Qwen3-1.7B priority-only baseline ===="
echo "method=${METHOD}"
echo "source_root=${SOURCE_ROOT}"
echo "pruned_layers_path=${PRUNED_LAYERS_PATH}"
echo "priority_root=${PRIORITY_ROOT}"
echo "gpus=${FINAL_GPUS:-0 1 2 3 4 5 6 7}"
echo "datasets=${DATASETS_OVERRIDE:-amc2023 gpqa_diamond arc_challenge}"

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

echo "==== START ShortGPT T20 priority suite ===="
PYTHON="${PYTHON_BIN}" \
CONFIG="${CONFIG}" \
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

echo "==== ALL DONE ShortGPT T20 Qwen3-1.7B priority ===="
echo "Priority summary: ${PRIORITY_ROOT}/aggregate_summary.md"

