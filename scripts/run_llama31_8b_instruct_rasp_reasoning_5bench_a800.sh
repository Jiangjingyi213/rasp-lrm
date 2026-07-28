#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

PYTHON_BIN="${PYTHON:-$(command -v python || command -v python3 || true)}"
CONFIG_PATH="${CONFIG:-configs/stage_calibrated_pruning/llama31_8b_instruct_rasp_reasoning_5bench.yaml}"
LOG_DIR="${LOG_DIR:-logs}"
HF_ENDPOINT="${HF_ENDPOINT:-https://huggingface.co}"
SEED="${STAGE_SEED:-3}"
MODEL_NAME_OR_PATH="${STAGE_MODEL_NAME_OR_PATH:-meta-llama/Llama-3.1-8B-Instruct}"

RUN_MODEL_PROBE="${RUN_MODEL_PROBE:-1}"
RUN_PRIOR_FIRST="${RUN_PRIOR_FIRST:-1}"
RUN_FULL_AFTER_PRIOR="${RUN_FULL_AFTER_PRIOR:-1}"
RUN_EVALUATE_DEV="${RUN_EVALUATE_DEV:-0}"

PRIOR_PROFILE="${PRIOR_PROFILE:-smoke}"
FULL_PROFILE="${FULL_PROFILE:-pilot}"
PRIOR_EVAL_LIMIT="${PRIOR_EVAL_LIMIT:-3}"
FULL_EVAL_LIMIT="${FULL_EVAL_LIMIT:--1}"

GENERATE_GPUS="${GENERATE_GPUS:-0}"
FINAL_GPUS="${FINAL_GPUS:-0}"
read -r -a GEN_GPU_LIST <<< "${GENERATE_GPUS}"
read -r -a FINAL_GPU_LIST <<< "${FINAL_GPUS}"
GENERATE_SHARD_COUNT="${STAGE_GENERATE_SHARD_COUNT:-${#GEN_GPU_LIST[@]}}"
FINAL_SHARD_COUNT="${STAGE_FINAL_SHARD_COUNT:-${#FINAL_GPU_LIST[@]}}"

if [[ -z "${PYTHON_BIN}" ]]; then
  echo "Could not locate python. Set PYTHON=/path/to/python." >&2
  exit 2
fi
if [[ "${#GEN_GPU_LIST[@]}" -lt "${GENERATE_SHARD_COUNT}" ]]; then
  echo "Need at least ${GENERATE_SHARD_COUNT} generate GPU ids, got ${GEN_GPU_LIST[*]}" >&2
  exit 2
fi
if [[ "${#FINAL_GPU_LIST[@]}" -lt "${FINAL_SHARD_COUNT}" ]]; then
  echo "Need at least ${FINAL_SHARD_COUNT} final GPU ids, got ${FINAL_GPU_LIST[*]}" >&2
  exit 2
fi

mkdir -p "${LOG_DIR}"

common_env() {
  export HF_ENDPOINT
  export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
  export HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-120}"
  export HF_HUB_ETAG_TIMEOUT="${HF_HUB_ETAG_TIMEOUT:-120}"
  export STAGE_SEED="${SEED}"
  export STAGE_MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH}"
}

model_probe() {
  echo "START model probe for ${MODEL_NAME_OR_PATH}"
  if [[ "${MODEL_NAME_OR_PATH}" != /* && -z "${HF_TOKEN:-}" && -z "${HUGGING_FACE_HUB_TOKEN:-}" ]]; then
    echo "WARNING: HF_TOKEN is empty. Remote gated models will fail unless the token is already configured in the HF cache." >&2
  fi
  common_env
  "${PYTHON_BIN}" - <<'PY'
import os
from transformers import AutoConfig, AutoTokenizer

model_name = os.environ["STAGE_MODEL_NAME_OR_PATH"]
cfg = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
print({"model_type": cfg.model_type, "num_hidden_layers": cfg.num_hidden_layers, "has_chat_template": bool(tok.chat_template)})
PY
  echo "DONE model probe"
}

run_stage() {
  local profile="$1"
  local stage="$2"
  shift 2 || true
  echo "START profile=${profile} stage=${stage}"
  common_env
  "${PYTHON_BIN}" -m src.main_stage_calibrated_pruning \
    --config "${CONFIG_PATH}" \
    --profile "${profile}" \
    --stage "${stage}" \
    "$@"
  echo "DONE profile=${profile} stage=${stage}"
}

wait_all() {
  local failed=0
  for pid in "$@"; do
    if ! wait "${pid}"; then
      failed=1
    fi
  done
  if [[ "${failed}" -ne 0 ]]; then
    echo "At least one worker failed. Check logs under ${LOG_DIR}." >&2
    exit 1
  fi
}

run_generate_shards() {
  local profile="$1"
  local pids=()
  for shard_index in $(seq 0 $((GENERATE_SHARD_COUNT - 1))); do
    local gpu="${GEN_GPU_LIST[$shard_index]}"
    local log_path="${LOG_DIR}/llama31_8b_instruct_5bench_${profile}_generate_shard${shard_index}_of${GENERATE_SHARD_COUNT}_gpu${gpu}.log"
    echo "Launching trajectory shard ${shard_index}/${GENERATE_SHARD_COUNT} on GPU ${gpu}; log=${log_path}"
    (
      common_env
      CUDA_VISIBLE_DEVICES="${gpu}" \
      STAGE_GENERATE_SHARD_INDEX="${shard_index}" \
      STAGE_GENERATE_SHARD_COUNT="${GENERATE_SHARD_COUNT}" \
      "${PYTHON_BIN}" -m src.main_stage_calibrated_pruning \
        --config "${CONFIG_PATH}" \
        --profile "${profile}" \
        --stage generate_trajectories
    ) > "${log_path}" 2>&1 &
    pids+=("$!")
  done
  wait_all "${pids[@]}"
}

run_final_shards() {
  local profile="$1"
  local limit="$2"
  local pids=()
  for shard_index in $(seq 0 $((FINAL_SHARD_COUNT - 1))); do
    local gpu="${FINAL_GPU_LIST[$shard_index]}"
    local log_path="${LOG_DIR}/llama31_8b_instruct_5bench_${profile}_final_shard${shard_index}_of${FINAL_SHARD_COUNT}_gpu${gpu}.log"
    echo "Launching final shard ${shard_index}/${FINAL_SHARD_COUNT} on GPU ${gpu}; log=${log_path}"
    (
      common_env
      CUDA_VISIBLE_DEVICES="${gpu}" \
      STAGE_FINAL_EVAL_LIMIT="${limit}" \
      STAGE_FINAL_SHARD_INDEX="${shard_index}" \
      STAGE_FINAL_SHARD_COUNT="${FINAL_SHARD_COUNT}" \
      "${PYTHON_BIN}" -m src.main_stage_calibrated_pruning \
        --config "${CONFIG_PATH}" \
        --profile "${profile}" \
        --stage evaluate_final \
        --force
    ) > "${log_path}" 2>&1 &
    pids+=("$!")
  done
  wait_all "${pids[@]}"
  common_env
  STAGE_FINAL_EVAL_LIMIT="${limit}" \
  STAGE_FINAL_SHARD_COUNT="${FINAL_SHARD_COUNT}" \
  run_stage "${profile}" merge_final_shards --force
}

run_profile() {
  local profile="$1"
  local final_limit="$2"
  run_stage "${profile}" preflight --force
  run_stage "${profile}" build_pool
  run_generate_shards "${profile}"
  STAGE_GENERATE_SHARD_COUNT="${GENERATE_SHARD_COUNT}" run_stage "${profile}" merge_trajectory_shards --force
  run_stage "${profile}" select_trajectories
  run_stage "${profile}" calibrate_masks
  run_stage "${profile}" validate_masks
  if [[ "${RUN_EVALUATE_DEV}" == "1" ]]; then
    run_stage "${profile}" evaluate_dev
  else
    echo "SKIP profile=${profile} stage=evaluate_dev because RUN_EVALUATE_DEV=${RUN_EVALUATE_DEV}"
  fi
  run_final_shards "${profile}" "${final_limit}"
  run_stage "${profile}" summarize --force
}

if [[ "${RUN_MODEL_PROBE}" == "1" ]]; then
  model_probe
fi

if [[ "${RUN_PRIOR_FIRST}" == "1" ]]; then
  echo "START Llama-3.1-8B-Instruct RASP five-benchmark prior/smoke"
  run_profile "${PRIOR_PROFILE}" "${PRIOR_EVAL_LIMIT}"
  echo "DONE Llama-3.1-8B-Instruct RASP five-benchmark prior/smoke"
fi

if [[ "${RUN_FULL_AFTER_PRIOR}" == "1" ]]; then
  echo "START Llama-3.1-8B-Instruct RASP five-benchmark pilot/full"
  run_profile "${FULL_PROFILE}" "${FULL_EVAL_LIMIT}"
  echo "DONE Llama-3.1-8B-Instruct RASP five-benchmark pilot/full"
fi
