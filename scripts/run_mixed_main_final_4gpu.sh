#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

PYTHON_BIN="${PYTHON:-/home/cike/jjy/envs/rasp_qwen3_eval/bin/python}"
CONFIG_PATH="${CONFIG:-configs/stage_calibrated_pruning/mixed_reasoning_seed3.yaml}"
PROFILE="${PROFILE:-pilot}"
POLICY_SELECTION="${STAGE_POLICY_SELECTION:-runs/08_stage_calibrated_pruning/main_pilot_mixed_reasoning_seed3_policy_selection/policy_selection_main_only.json}"
FINAL_EVAL_LIMIT="${STAGE_FINAL_EVAL_LIMIT:--1}"
FINAL_METHODS="${STAGE_FINAL_METHODS:-structured_dense,dynamic_stage_main,static_matched_global}"
SHARD_COUNT="${STAGE_FINAL_SHARD_COUNT:-4}"
LOG_DIR="${LOG_DIR:-logs}"
HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
SEED="${STAGE_SEED:-3}"

read -r -a GPUS <<< "${FINAL_GPUS:-0 1 2 3}"
if [[ "${#GPUS[@]}" -lt "${SHARD_COUNT}" ]]; then
  echo "Need at least ${SHARD_COUNT} GPU ids, got ${#GPUS[@]}: ${GPUS[*]}" >&2
  exit 2
fi

if [[ ! -f "${POLICY_SELECTION}" ]]; then
  echo "Missing policy selection: ${POLICY_SELECTION}" >&2
  echo "Run scripts/select_mixed_main_policy.sh before final evaluation." >&2
  exit 2
fi

mkdir -p "${LOG_DIR}"

pids=()
for shard_index in $(seq 0 $((SHARD_COUNT - 1))); do
  gpu="${GPUS[$shard_index]}"
  log_path="${LOG_DIR}/mixed_final_seed${SEED}_shard${shard_index}_of${SHARD_COUNT}_gpu${gpu}.log"
  echo "Launching mixed final shard ${shard_index}/${SHARD_COUNT} on GPU ${gpu}; log=${log_path}"
  CUDA_VISIBLE_DEVICES="${gpu}" \
  HF_ENDPOINT="${HF_ENDPOINT}" \
  HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}" \
  HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-120}" \
  HF_HUB_ETAG_TIMEOUT="${HF_HUB_ETAG_TIMEOUT:-120}" \
  STAGE_SEED="${SEED}" \
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
  echo "At least one final shard failed. Check logs under ${LOG_DIR}." >&2
  exit 1
fi

echo "All mixed final shards completed; merging."
HF_ENDPOINT="${HF_ENDPOINT}" \
STAGE_SEED="${SEED}" \
STAGE_POLICY_SELECTION="${POLICY_SELECTION}" \
STAGE_FINAL_EVAL_LIMIT="${FINAL_EVAL_LIMIT}" \
STAGE_FINAL_SHARD_COUNT="${SHARD_COUNT}" \
"${PYTHON_BIN}" -m src.main_stage_calibrated_pruning \
  --config "${CONFIG_PATH}" \
  --profile "${PROFILE}" \
  --stage merge_final_shards \
  --force

echo "Merged final summary written."
