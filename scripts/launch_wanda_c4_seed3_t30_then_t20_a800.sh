#!/usr/bin/env bash
set -euo pipefail

unset CUDA_VISIBLE_DEVICES
unset STAGE_MODEL_NAME_OR_PATH
unset STAGE_MODEL_DTYPE
unset STAGE_WORKFLOW_ROOT
unset STAGE_FINAL_DATASET_NAME
unset STAGE_FINAL_METHODS
unset DATASETS_OVERRIDE
unset HF_DATASETS_OFFLINE
unset TRANSFORMERS_OFFLINE
unset HF_HUB_OFFLINE

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"

export PYTHON="${PYTHON:-/root/jjy/envs/rasp_qwen3_eval/bin/python}"
export PYTHONPATH="${PYTHONPATH:-$(pwd)}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HOME="${HF_HOME:-/root/autodl-tmp/rasp_cache/hf_home}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-/root/autodl-tmp/rasp_cache/hf_cache}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-/root/autodl-tmp/rasp_cache/hf_cache}"
export TORCH_HOME="${TORCH_HOME:-/root/autodl-tmp/rasp_cache/torch_cache}"
export TMPDIR="${TMPDIR:-/root/autodl-tmp/rasp_cache/tmp}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-300}"
export HF_HUB_ETAG_TIMEOUT="${HF_HUB_ETAG_TIMEOUT:-300}"
export C4_DOWNLOAD_TIMEOUT="${C4_DOWNLOAD_TIMEOUT:-300}"
export C4_SOURCE_MODE="${C4_SOURCE_MODE:-direct}"

export FINAL_GPUS="${FINAL_GPUS:-0 0}"
export STAGE_FINAL_SHARD_COUNT="${STAGE_FINAL_SHARD_COUNT:-2}"
export STAGE_SEED="${STAGE_SEED:-3}"
export SKIP_EXISTING="${SKIP_EXISTING:-1}"

mkdir -p logs/wanda_c4_seed3_t30 logs/wanda_c4_seed3_t20 "${TMPDIR}"

echo "==== START Wanda-C4 seed3 t30: priority then full ===="
STAGE_FINAL_METHODS="wanda_c4_seed3_t30" \
LOG_DIR="logs/wanda_c4_seed3_t30" \
bash scripts/run_wanda_c4_seed3_qwen3_1p7b_priority_then_full_8gpu.sh

echo "==== START Wanda-C4 seed3 t20: priority then full ===="
STAGE_FINAL_METHODS="wanda_c4_seed3_t20" \
LOG_DIR="logs/wanda_c4_seed3_t20" \
bash scripts/run_wanda_c4_seed3_qwen3_1p7b_priority_then_full_8gpu.sh

echo "==== ALL DONE Wanda-C4 seed3 t30/t20 ===="
