#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON:-python3}"
PROFILE="${PROFILE:-smoke}"
HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
LOG_DIR="${LOG_DIR:-logs}"

mkdir -p "${LOG_DIR}"

configs=(
  "configs/stage_calibrated_pruning/orca_math_smoke_plus_r025.yaml"
  "configs/stage_calibrated_pruning/orca_math_smoke_plus_r030.yaml"
  "configs/stage_calibrated_pruning/orca_math_smoke_plus_r035.yaml"
  "configs/stage_calibrated_pruning/orca_math_smoke_plus_r040.yaml"
  "configs/stage_calibrated_pruning/orca_math_smoke_plus_r050.yaml"
)

read -r -a gpus <<< "${BOUNDARY_GPUS:-0 1 2 3 4}"

if [[ "${#gpus[@]}" -lt "${#configs[@]}" ]]; then
  echo "Need at least ${#configs[@]} GPU ids, got ${#gpus[@]}: ${gpus[*]}" >&2
  exit 2
fi

for index in "${!configs[@]}"; do
  config="${configs[$index]}"
  gpu="${gpus[$index]}"
  tag="$(basename "${config}" .yaml | sed 's/orca_math_smoke_plus_//')"
  log_path="${LOG_DIR}/stage_boundary_${tag}_gpu${gpu}.log"
  echo "Launching ${config} on GPU ${gpu}; log=${log_path}"
  CUDA_VISIBLE_DEVICES="${gpu}" \
  HF_ENDPOINT="${HF_ENDPOINT}" \
  HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}" \
  HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-120}" \
  HF_HUB_ETAG_TIMEOUT="${HF_HUB_ETAG_TIMEOUT:-120}" \
  CONFIG="${config}" \
  PROFILE="${PROFILE}" \
  PYTHON="${PYTHON_BIN}" \
  nohup bash scripts/run_stage_calibrated_pruning.sh > "${log_path}" 2>&1 &
done

echo "Boundary smoke_plus jobs launched."
echo "Monitor with: tail -f ${LOG_DIR}/stage_boundary_r0*_gpu*.log"
