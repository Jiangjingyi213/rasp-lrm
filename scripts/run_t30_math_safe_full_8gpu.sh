#!/usr/bin/env bash
set -euo pipefail

unset HF_DATASETS_OFFLINE
unset TRANSFORMERS_OFFLINE
unset HF_HUB_OFFLINE

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

PYTHON_BIN="${PYTHON:-/home/cike/jjy/envs/rasp_qwen3_eval/bin/python}"
CONFIG_PATH="${CONFIG:-configs/stage_calibrated_pruning/mixed_reasoning_seed3_t30_math_safe_full.yaml}"
PROFILE="${PROFILE:-pilot}"
SOURCE_ROOT="${SOURCE_ROOT:-runs/08_stage_calibrated_pruning/main_pilot_mixed_reasoning_seed3}"
RUN_ROOT="${RUN_ROOT:-runs/08_stage_calibrated_pruning/main_pilot_mixed_reasoning_seed3_t30_math_safe_full}"
FINAL_METHODS="${STAGE_FINAL_METHODS:-structured_dense,static_t30_0p37,t30_math_safe}"
FINAL_EVAL_LIMIT="${STAGE_FINAL_EVAL_LIMIT:--1}"
LOG_DIR="${LOG_DIR:-logs}"
HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
SEED="${STAGE_SEED:-3}"
RUN_LABEL="${RUN_LABEL:-t30_math_safe_full}"
SKIP_EXISTING="${SKIP_EXISTING:-0}"

read -r -a GPUS <<< "${FINAL_GPUS:-0 1 2 3 4 5 6 7}"
SHARD_COUNT="${STAGE_FINAL_SHARD_COUNT:-${#GPUS[@]}}"
if [[ "${#GPUS[@]}" -lt "${SHARD_COUNT}" ]]; then
  echo "Need at least ${SHARD_COUNT} GPU ids, got ${#GPUS[@]}: ${GPUS[*]}" >&2
  exit 2
fi

for artifact_dir in 03_selected 04_masks; do
  if [[ ! -d "${SOURCE_ROOT}/${artifact_dir}" ]]; then
    echo "Missing reusable artifact: ${SOURCE_ROOT}/${artifact_dir}" >&2
    echo "Set SOURCE_ROOT to an existing mixed pilot run that contains 03_selected and 04_masks." >&2
    exit 2
  fi
done

mkdir -p "${LOG_DIR}" "${RUN_ROOT}"

echo "START preflight for ${RUN_ROOT}"
HF_ENDPOINT="${HF_ENDPOINT}" \
HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}" \
HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-120}" \
HF_HUB_ETAG_TIMEOUT="${HF_HUB_ETAG_TIMEOUT:-120}" \
STAGE_SEED="${SEED}" \
STAGE_WORKFLOW_ROOT="${RUN_ROOT}" \
"${PYTHON_BIN}" -m src.main_stage_calibrated_pruning \
  --config "${CONFIG_PATH}" \
  --profile "${PROFILE}" \
  --stage preflight \
  --force
echo "DONE preflight"

for artifact_dir in 03_selected 04_masks; do
  if [[ ! -d "${RUN_ROOT}/${artifact_dir}" ]]; then
    echo "Reusing ${SOURCE_ROOT}/${artifact_dir} -> ${RUN_ROOT}/${artifact_dir}"
    cp -a "${SOURCE_ROOT}/${artifact_dir}" "${RUN_ROOT}/"
  else
    echo "Keeping existing ${RUN_ROOT}/${artifact_dir}"
  fi
done

echo "START ${RUN_LABEL} sharded final; methods=${FINAL_METHODS}; shards=${SHARD_COUNT}; global_limit=${FINAL_EVAL_LIMIT}"
pids=()
for shard_index in $(seq 0 $((SHARD_COUNT - 1))); do
  gpu="${GPUS[$shard_index]}"
  shard_summary="$(printf "%s/06_final/summary_shard_%05d_of_%05d.json" "${RUN_ROOT}" "${shard_index}" "${SHARD_COUNT}")"
  if [[ "${SKIP_EXISTING}" == "1" && -f "${shard_summary}" ]]; then
    echo "SKIP shard ${shard_index}/${SHARD_COUNT}; existing ${shard_summary}"
    continue
  fi
  log_path="${LOG_DIR}/${RUN_LABEL}_seed${SEED}_shard${shard_index}_of${SHARD_COUNT}_gpu${gpu}.log"
  echo "Launching shard ${shard_index}/${SHARD_COUNT} on GPU ${gpu}; log=${log_path}"
  CUDA_VISIBLE_DEVICES="${gpu}" \
  HF_ENDPOINT="${HF_ENDPOINT}" \
  HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}" \
  HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-120}" \
  HF_HUB_ETAG_TIMEOUT="${HF_HUB_ETAG_TIMEOUT:-120}" \
  STAGE_SEED="${SEED}" \
  STAGE_WORKFLOW_ROOT="${RUN_ROOT}" \
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
  echo "At least one ${RUN_LABEL} shard failed. Check logs under ${LOG_DIR}." >&2
  exit 1
fi

echo "START merge_final_shards"
HF_ENDPOINT="${HF_ENDPOINT}" \
STAGE_SEED="${SEED}" \
STAGE_WORKFLOW_ROOT="${RUN_ROOT}" \
STAGE_FINAL_EVAL_LIMIT="${FINAL_EVAL_LIMIT}" \
STAGE_FINAL_SHARD_COUNT="${SHARD_COUNT}" \
"${PYTHON_BIN}" -m src.main_stage_calibrated_pruning \
  --config "${CONFIG_PATH}" \
  --profile "${PROFILE}" \
  --stage merge_final_shards \
  --force
echo "DONE merge_final_shards"

echo "ALL DONE: ${RUN_ROOT}/06_final/summary.json"
