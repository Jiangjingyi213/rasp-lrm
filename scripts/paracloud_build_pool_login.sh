#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

RASP_ENV_PREFIX="${RASP_ENV_PREFIX:-${HOME}/workspace/envs/rasp_qwen3_eval}"
RASP_CACHE_ROOT="${RASP_CACHE_ROOT:-${HOME}/workspace/cache}"
PYTHON="${PYTHON:-${RASP_ENV_PREFIX}/bin/python}"
CONFIG="${CONFIG:-configs/stage_calibrated_pruning/mixed_reasoning_seed3.yaml}"
PROFILE="${PROFILE:-smoke}"
export STAGE_MODEL_DTYPE="${STAGE_MODEL_DTYPE:-bfloat16}"
export HF_HOME="${HF_HOME:-${RASP_CACHE_ROOT}/huggingface}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-${HF_HOME}/hub}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
export TMPDIR="${TMPDIR:-${RASP_CACHE_ROOT}/tmp}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://huggingface.co}"

mkdir -p logs/slurm "${TMPDIR}"

"${PYTHON}" -m src.main_stage_calibrated_pruning \
  --config "${CONFIG}" \
  --profile "${PROFILE}" \
  --stage build_pool
