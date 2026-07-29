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

cd "$(dirname "${BASH_SOURCE[0]}")/.."

WAIT_PATTERN="${WAIT_PATTERN:-fixed_global_t30_baseline|main_pilot_fixed_global_t30_baseline|run_fixed_global_t30_baseline}"
WAIT_SECONDS="${WAIT_SECONDS:-300}"
while pgrep -af "${WAIT_PATTERN}" >/dev/null; do
  echo "$(date) waiting for fixed-global jobs to finish..."
  pgrep -af "${WAIT_PATTERN}" || true
  sleep "${WAIT_SECONDS}"
done

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export PYTHON="${PYTHON:-/home/cike/jjy/envs/rasp_qwen3_eval/bin/python}"
export PYTHONPATH="${PYTHONPATH:-$(pwd)}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HOME="${HF_HOME:-/home/cike/jjy/rasp_cache/hf_home}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-/home/cike/jjy/rasp_cache/hf_cache}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-/home/cike/jjy/rasp_cache/hf_cache}"
export TORCH_HOME="${TORCH_HOME:-/home/cike/jjy/rasp_cache/torch_cache}"
export TMPDIR="${TMPDIR:-/home/cike/jjy/rasp_cache/tmp}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-300}"
export HF_HUB_ETAG_TIMEOUT="${HF_HUB_ETAG_TIMEOUT:-300}"
export C4_DOWNLOAD_TIMEOUT="${C4_DOWNLOAD_TIMEOUT:-300}"
export C4_SOURCE_MODE="${C4_SOURCE_MODE:-direct}"
export FINAL_GPUS="${FINAL_GPUS:-0 1 2 3 4 5 6 7}"
export STAGE_FINAL_SHARD_COUNT="${STAGE_FINAL_SHARD_COUNT:-8}"
export STAGE_SEED="${STAGE_SEED:-3}"
export STAGE_FINAL_METHODS="${STAGE_FINAL_METHODS:-wanda_c4_seed3_t20}"
export LOG_DIR="${LOG_DIR:-logs/wanda_c4_seed3_qwen3_1p7b_t20_165}"
export SKIP_EXISTING="${SKIP_EXISTING:-1}"

mkdir -p "${LOG_DIR}" "${TMPDIR}"
echo "==== START Qwen3-1.7B Wanda-C4 seed3 t20 on 165 ===="
bash scripts/run_wanda_c4_seed3_qwen3_1p7b_priority_then_full_8gpu.sh
