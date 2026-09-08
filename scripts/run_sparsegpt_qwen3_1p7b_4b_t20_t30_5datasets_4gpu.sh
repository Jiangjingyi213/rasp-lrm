#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

PYTHON_BIN="${PYTHON:-/home/cike/jjy/envs/rasp_qwen3_eval/bin/python}"
GPU_LIST="${FINAL_GPUS:-0 1 2 3}"
read -r -a GPUS <<< "${GPU_LIST}"
SHARD_COUNT="${STAGE_FINAL_SHARD_COUNT:-${#GPUS[@]}}"
PREPARE_VISIBLE="${PREPARE_CUDA_VISIBLE_DEVICES:-$(printf "%s," "${GPUS[@]}" | sed 's/,$//')}"
LOG_DIR="${LOG_DIR:-logs/12_additional_baselines/05_sparsegpt}"
RUN_BASE="${RUN_BASE:-runs/12_additional_baselines/05_sparsegpt}"
PROFILE="${PROFILE:-pilot}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
SPARSEGPT_BATCH="${SPARSEGPT_CALIBRATION_BATCH_SIZE:-1}"
METHODS="${SPARSEGPT_METHODS:-sparsegpt_t20_official sparsegpt_t30_official}"

mkdir -p "${LOG_DIR}" "${RUN_BASE}/artifacts/qwen3_1p7b" "${RUN_BASE}/artifacts/qwen3_4b"

method_label() {
  case "$1" in
    sparsegpt_t20_official) printf "t20p22" ;;
    sparsegpt_t30_official) printf "t30p35" ;;
    *) printf "%s" "$1" | tr -c 'A-Za-z0-9_-' '_' ;;
  esac
}

run_sparsegpt_pair() {
  local model_tag="$1"
  local full_script="$2"
  local priority_script="$3"
  local method="$4"
  local label
  label="$(method_label "${method}")"

  echo "START ${model_tag} ${method} full: GSM8K + Math500"
  PYTHON="${PYTHON_BIN}" \
  FINAL_GPUS="${GPU_LIST}" \
  STAGE_FINAL_SHARD_COUNT="${SHARD_COUNT}" \
  PREPARE_CUDA_VISIBLE_DEVICES="${PREPARE_VISIBLE}" \
  STAGE_FINAL_METHODS="${method}" \
  SPARSEGPT_CALIBRATION_BATCH_SIZE="${SPARSEGPT_BATCH}" \
  SPARSEGPT_ARTIFACT_ROOT="${RUN_BASE}/artifacts/${model_tag}" \
  RUN_ROOT="${RUN_BASE}/${model_tag}_${label}_full" \
  LOG_DIR="${LOG_DIR}" \
  RUN_LABEL="sparsegpt_${model_tag}_${label}_full" \
  PROFILE="${PROFILE}" \
  SKIP_EXISTING="${SKIP_EXISTING}" \
  bash "${full_script}"
  echo "DONE ${model_tag} ${method} full"

  echo "START ${model_tag} ${method} priority suite: AMC2023 + GPQA-Diamond + ARC-Challenge"
  PYTHON="${PYTHON_BIN}" \
  FINAL_GPUS="${GPU_LIST}" \
  STAGE_FINAL_SHARD_COUNT="${SHARD_COUNT}" \
  PREPARE_CUDA_VISIBLE_DEVICES="${PREPARE_VISIBLE}" \
  STAGE_FINAL_METHODS="${method}" \
  SPARSEGPT_CALIBRATION_BATCH_SIZE="${SPARSEGPT_BATCH}" \
  SPARSEGPT_ARTIFACT_ROOT="${RUN_BASE}/artifacts/${model_tag}" \
  RUN_ROOT="${RUN_BASE}/${model_tag}_${label}_priority_suite" \
  LOG_DIR="${LOG_DIR}" \
  LOG_PREFIX="sparsegpt_${model_tag}_${label}_priority" \
  PROFILE="${PROFILE}" \
  SKIP_EXISTING="${SKIP_EXISTING}" \
  bash "${priority_script}"
  echo "DONE ${model_tag} ${method} priority suite"
}

for method in ${METHODS}; do
  run_sparsegpt_pair \
    "qwen3_1p7b" \
    "scripts/run_sparsegpt_official_qwen3_1p7b_full_8gpu.sh" \
    "scripts/run_sparsegpt_official_qwen3_1p7b_priority_suite_8gpu.sh" \
    "${method}"
done

for method in ${METHODS}; do
  run_sparsegpt_pair \
    "qwen3_4b" \
    "scripts/run_sparsegpt_official_qwen3_4b_full_gpu.sh" \
    "scripts/run_sparsegpt_official_qwen3_4b_priority_suite_gpu.sh" \
    "${method}"
done

echo "ALL DONE: SparseGPT matrix under ${RUN_BASE}"
