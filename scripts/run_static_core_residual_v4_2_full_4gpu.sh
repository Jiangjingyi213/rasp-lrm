#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

PYTHON_BIN="${PYTHON:-/home/cike/jjy/envs/rasp_qwen3_eval/bin/python}"
CONFIG_PATH="${CONFIG:-configs/stage_calibrated_pruning/mixed_reasoning_seed3_static_core_residual_v4_2_full.yaml}"
PROFILE="${PROFILE:-pilot}"
SOURCE_ROOT="${SOURCE_ROOT:-runs/08_stage_calibrated_pruning/main_pilot_mixed_reasoning_seed3_static_core_residual_v4_2}"
RUN_ROOT="${RUN_ROOT:-runs/08_stage_calibrated_pruning/main_pilot_mixed_reasoning_seed3_static_core_residual_v4_2_full}"
FINAL_EVAL_LIMIT="${STAGE_FINAL_EVAL_LIMIT:--1}"
FINAL_METHODS="${STAGE_FINAL_METHODS:-structured_dense,static_core_residual_stage_dynamic,static_matched_global}"
SHARD_COUNT="${STAGE_FINAL_SHARD_COUNT:-4}"
LOG_DIR="${LOG_DIR:-logs}"
HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

read -r -a GPUS <<< "${FINAL_GPUS:-0 1 2 3}"
if [[ "${#GPUS[@]}" -lt "${SHARD_COUNT}" ]]; then
  echo "Need at least ${SHARD_COUNT} GPU ids, got ${#GPUS[@]}: ${GPUS[*]}" >&2
  exit 2
fi

for artifact_dir in 03_selected 04_masks 05_dev; do
  if [[ ! -d "${SOURCE_ROOT}/${artifact_dir}" ]]; then
    echo "Missing reusable artifact: ${SOURCE_ROOT}/${artifact_dir}" >&2
    exit 2
  fi
done

mkdir -p "${LOG_DIR}" "${RUN_ROOT}"

echo "START preflight for ${RUN_ROOT}"
HF_ENDPOINT="${HF_ENDPOINT}" \
HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}" \
HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-120}" \
HF_HUB_ETAG_TIMEOUT="${HF_HUB_ETAG_TIMEOUT:-120}" \
"${PYTHON_BIN}" -m src.main_stage_calibrated_pruning \
  --config "${CONFIG_PATH}" \
  --profile "${PROFILE}" \
  --stage preflight \
  --force
echo "DONE preflight"

for artifact_dir in 03_selected 04_masks 05_dev; do
  if [[ ! -d "${RUN_ROOT}/${artifact_dir}" ]]; then
    echo "Reusing ${SOURCE_ROOT}/${artifact_dir} -> ${RUN_ROOT}/${artifact_dir}"
    cp -a "${SOURCE_ROOT}/${artifact_dir}" "${RUN_ROOT}/"
  else
    echo "Keeping existing ${RUN_ROOT}/${artifact_dir}"
  fi
done

POLICY_SELECTION="${RUN_ROOT}/05_dev/adaptive_griffin_policy_selection.json"
if [[ ! -f "${POLICY_SELECTION}" ]]; then
  echo "Missing policy selection: ${POLICY_SELECTION}" >&2
  exit 1
fi

echo "START full sharded final; limit=${FINAL_EVAL_LIMIT}; methods=${FINAL_METHODS}"
pids=()
for shard_index in $(seq 0 $((SHARD_COUNT - 1))); do
  gpu="${GPUS[$shard_index]}"
  log_path="${LOG_DIR}/v4_2_full_shard${shard_index}_of${SHARD_COUNT}_gpu${gpu}.log"
  echo "Launching full shard ${shard_index}/${SHARD_COUNT} on GPU ${gpu}; log=${log_path}"
  CUDA_VISIBLE_DEVICES="${gpu}" \
  HF_ENDPOINT="${HF_ENDPOINT}" \
  HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}" \
  HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-120}" \
  HF_HUB_ETAG_TIMEOUT="${HF_HUB_ETAG_TIMEOUT:-120}" \
  STAGE_POLICY_SELECTION="${POLICY_SELECTION}" \
  STAGE_FINAL_EVAL_LIMIT="${FINAL_EVAL_LIMIT}" \
  STAGE_FINAL_METHODS="${FINAL_METHODS}" \
  STAGE_FINAL_SHARD_INDEX="${shard_index}" \
  STAGE_FINAL_SHARD_COUNT="${SHARD_COUNT}" \
  "${PYTHON_BIN}" -m src.main_stage_calibrated_pruning \
    --config "${CONFIG_PATH}" \
    --profile "${PROFILE}" \
    --stage evaluate_final \
    --force \
    > "${log_path}" 2>&1 &
  pids+=("$!")
done

failed=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    failed=1
  fi
done
if [[ "${failed}" -ne 0 ]]; then
  echo "At least one full shard failed. Check logs under ${LOG_DIR}." >&2
  exit 1
fi

echo "START merge_final_shards"
HF_ENDPOINT="${HF_ENDPOINT}" \
STAGE_POLICY_SELECTION="${POLICY_SELECTION}" \
STAGE_FINAL_EVAL_LIMIT="${FINAL_EVAL_LIMIT}" \
STAGE_FINAL_SHARD_COUNT="${SHARD_COUNT}" \
"${PYTHON_BIN}" -m src.main_stage_calibrated_pruning \
  --config "${CONFIG_PATH}" \
  --profile "${PROFILE}" \
  --stage merge_final_shards \
  --force
echo "DONE merge_final_shards"

echo "ALL DONE: ${RUN_ROOT}/06_final/summary.json"
