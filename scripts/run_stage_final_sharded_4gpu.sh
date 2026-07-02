#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON:-/home/cike/jjy/envs/rasp_qwen3_eval/bin/python}"
CONFIG_PATH="${CONFIG:-configs/stage_calibrated_pruning/orca_math_pilot_fast.yaml}"
PROFILE="${PROFILE:-pilot}"
POLICY_SELECTION="${STAGE_POLICY_SELECTION:-runs/08_stage_calibrated_pruning/orca_pilot_fast_policy_selection_labels_v4/policy_selection.json}"
FINAL_EVAL_LIMIT="${STAGE_FINAL_EVAL_LIMIT:-200}"
FINAL_METHODS="${STAGE_FINAL_METHODS:-ordinary_dense,structured_dense,conservative_stage_specific_0p10,main_dynamic_stage_specific_0p20,stage_budget_balanced,aggressive_stage_balanced_global_0p35,shuffled_control_shuffled_stage_0p10}"
SHARD_COUNT="${STAGE_FINAL_SHARD_COUNT:-4}"
LOG_DIR="${LOG_DIR:-logs}"
HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
SEED="${STAGE_SEED:-3}"

read -r -a gpus <<< "${FINAL_GPUS:-0 1 2 3}"
if [[ "${#gpus[@]}" -lt "${SHARD_COUNT}" ]]; then
  echo "Need at least ${SHARD_COUNT} GPU ids, got ${#gpus[@]}: ${gpus[*]}" >&2
  exit 2
fi

mkdir -p "${LOG_DIR}"

pids=()
for shard_index in $(seq 0 $((SHARD_COUNT - 1))); do
  gpu="${gpus[$shard_index]}"
  log_path="${LOG_DIR}/stage_final_seed${SEED}_shard${shard_index}_of${SHARD_COUNT}_gpu${gpu}.log"
  echo "Launching final shard ${shard_index}/${SHARD_COUNT} on GPU ${gpu}; log=${log_path}"
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

for pid in "${pids[@]}"; do
  wait "${pid}"
done

echo "All final shards completed; merging."
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
