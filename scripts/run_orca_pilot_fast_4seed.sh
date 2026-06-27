#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON:-/home/cike/jjy/envs/rasp_qwen3_eval/bin/python}"
CONFIG_PATH="${CONFIG:-configs/stage_calibrated_pruning/orca_math_pilot_fast.yaml}"
PROFILE="${PROFILE:-pilot}"
HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
LOG_DIR="${LOG_DIR:-logs}"

mkdir -p "${LOG_DIR}"

read -r -a seeds <<< "${PILOT_SEEDS:-1 2 3 4}"
read -r -a gpus <<< "${PILOT_GPUS:-0 1 2 3}"

if [[ "${#gpus[@]}" -lt "${#seeds[@]}" ]]; then
  echo "Need at least ${#seeds[@]} GPU ids, got ${#gpus[@]}: ${gpus[*]}" >&2
  exit 2
fi

for index in "${!seeds[@]}"; do
  seed="${seeds[$index]}"
  gpu="${gpus[$index]}"
  log_path="${LOG_DIR}/stage_pilot_fast_seed${seed}_gpu${gpu}.log"
  echo "Launching pilot-fast seed=${seed} on GPU ${gpu}; log=${log_path}"
  CUDA_VISIBLE_DEVICES="${gpu}" \
  HF_ENDPOINT="${HF_ENDPOINT}" \
  HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}" \
  HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-120}" \
  HF_HUB_ETAG_TIMEOUT="${HF_HUB_ETAG_TIMEOUT:-120}" \
  CONFIG="${CONFIG_PATH}" \
  PROFILE="${PROFILE}" \
  STAGE_SEED="${seed}" \
  PYTHON="${PYTHON_BIN}" \
  nohup bash scripts/run_stage_calibrated_pruning.sh > "${log_path}" 2>&1 &
done

echo "Orca pilot-fast jobs launched."
echo "Monitor with: tail -f ${LOG_DIR}/stage_pilot_fast_seed*_gpu*.log"
