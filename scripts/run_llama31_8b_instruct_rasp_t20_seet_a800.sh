#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

export PYTHON="${PYTHON:-/root/jjy/envs/rasp_qwen3_eval/bin/python}"
export STAGE_MODEL_NAME_OR_PATH="${STAGE_MODEL_NAME_OR_PATH:-/root/autodl-tmp/models/unsloth_Llama-3.1-8B-Instruct}"
export STAGE_MODEL_DTYPE="${STAGE_MODEL_DTYPE:-bfloat16}"
export STAGE_FINAL_METHODS="${STAGE_FINAL_METHODS:-ordinary_dense,structured_dense,static_t20_0p25,rasp_t20_safe_under}"
export GENERATE_GPUS="${GENERATE_GPUS:-0}"
export FINAL_GPUS="${FINAL_GPUS:-0}"
export PRIOR_PROFILE="${PRIOR_PROFILE:-smoke}"
export FULL_PROFILE="${FULL_PROFILE:-pilot}"

exec bash scripts/run_llama31_8b_instruct_rasp_reasoning_5bench_a800.sh "$@"
