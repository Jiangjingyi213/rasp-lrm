#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

export PYTHON="${PYTHON:-/home/cike/jjy/envs/rasp_qwen3_eval/bin/python}"
export STAGE_MODEL_NAME_OR_PATH="${STAGE_MODEL_NAME_OR_PATH:-/home/cike/models/unsloth_Llama-3.1-8B-Instruct}"
export STAGE_MODEL_DTYPE="${STAGE_MODEL_DTYPE:-float16}"
export STAGE_FINAL_METHODS="${STAGE_FINAL_METHODS:-ordinary_dense,structured_dense,static_t40_0p48,rasp_t40_safe_under}"

# Important: keep commas. This launches one worker that sees all seven P100s,
# letting Transformers device_map=auto shard the 8B model across GPUs.
export GENERATE_GPUS="${GENERATE_GPUS:-0,1,2,3,4,5,6}"
export FINAL_GPUS="${FINAL_GPUS:-0,1,2,3,4,5,6}"
export STAGE_GENERATE_SHARD_COUNT="${STAGE_GENERATE_SHARD_COUNT:-1}"
export STAGE_FINAL_SHARD_COUNT="${STAGE_FINAL_SHARD_COUNT:-1}"
export PRIOR_PROFILE="${PRIOR_PROFILE:-smoke}"
export FULL_PROFILE="${FULL_PROFILE:-pilot}"

exec bash scripts/run_llama31_8b_instruct_rasp_reasoning_5bench_a800.sh "$@"
