#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

PYTHON="${PYTHON:-/home/cike/jjy/envs/rasp_qwen3_eval/bin/python}"
CONFIG="${CONFIG:-configs/stage_calibrated_pruning/mixed_reasoning_seed3.yaml}"
PROFILE="${PROFILE:-smoke}"
GPU_IDS="${GPU_IDS:-0 1 2 3}"
read -r -a GPUS <<< "${GPU_IDS}"
SHARD_COUNT="${STAGE_GENERATE_SHARD_COUNT:-${#GPUS[@]}}"
LOG_DIR="${LOG_DIR:-logs}"

if [[ "${#GPUS[@]}" -lt "${SHARD_COUNT}" ]]; then
  echo "Need at least ${SHARD_COUNT} GPU ids, got ${#GPUS[@]}: ${GPUS[*]}" >&2
  exit 2
fi

mkdir -p "${LOG_DIR}"

run_stage() {
  local stage="$1"
  shift || true
  echo "START stage=${stage} profile=${PROFILE}"
  "${PYTHON}" -m src.main_stage_calibrated_pruning \
      --config "${CONFIG}" \
      --profile "${PROFILE}" \
      --stage "${stage}" \
      "$@" || {
    local status=$?
    echo "FAILED stage=${stage}" >&2
    return "${status}"
  }
  echo "DONE stage=${stage}"
}

finalize_partial() {
  "${PYTHON}" -m src.main_stage_calibrated_pruning \
    --config "${CONFIG}" \
    --profile "${PROFILE}" \
    --stage summarize \
    --force >/dev/null 2>&1 || true
}

run_generate_shards() {
  local pids=()
  for shard_index in $(seq 0 $((SHARD_COUNT - 1))); do
    local gpu="${GPUS[$shard_index]}"
    local log_path="${LOG_DIR}/stage_generate_${PROFILE}_shard${shard_index}_of${SHARD_COUNT}_gpu${gpu}.log"
    echo "Launching generate shard ${shard_index}/${SHARD_COUNT} on GPU ${gpu}; log=${log_path}"
    (
      CUDA_VISIBLE_DEVICES="${gpu}" \
      STAGE_GENERATE_SHARD_INDEX="${shard_index}" \
      STAGE_GENERATE_SHARD_COUNT="${SHARD_COUNT}" \
      "${PYTHON}" -m src.main_stage_calibrated_pruning \
        --config "${CONFIG}" \
        --profile "${PROFILE}" \
        --stage generate_trajectories
    ) >"${log_path}" 2>&1 &
    pids+=("$!")
  done

  local failed=0
  for pid in "${pids[@]}"; do
    if ! wait "${pid}"; then
      failed=1
    fi
  done
  if [[ "${failed}" -ne 0 ]]; then
    echo "At least one generate shard failed. Check logs under ${LOG_DIR}." >&2
    return 1
  fi
}

trap finalize_partial ERR

run_stage preflight

while true; do
  run_stage build_pool
  run_generate_shards
  STAGE_GENERATE_SHARD_COUNT="${SHARD_COUNT}" run_stage merge_trajectory_shards --force
  if run_stage select_trajectories; then
    break
  else
    status=$?
  fi
  if [[ "${PROFILE}" != "formal" || "${status}" -ne 42 ]]; then
    exit "${status}"
  fi
  echo "Formal selection requested more candidate problems; expanding and regenerating missing shards."
done

for stage in calibrate_masks validate_masks evaluate_dev evaluate_final summarize; do
  run_stage "${stage}"
done
