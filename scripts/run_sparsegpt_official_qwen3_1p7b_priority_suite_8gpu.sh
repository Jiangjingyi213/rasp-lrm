#!/usr/bin/env bash
set -euo pipefail

export CONFIG="${CONFIG:-configs/stage_calibrated_pruning/sparsegpt_official_qwen3_1p7b_priority_suite.yaml}"
export RUN_ROOT="${RUN_ROOT:-runs/08_stage_calibrated_pruning/main_pilot_sparsegpt_official_qwen3_1p7b_priority_suite}"
export STAGE_FINAL_METHODS="${STAGE_FINAL_METHODS:-sparsegpt_t30_official}"
export LOG_PREFIX="${LOG_PREFIX:-sparsegpt_official_qwen3_1p7b_priority}"
export BASELINE_LABEL="${BASELINE_LABEL:-SparseGPT Official}"
export BASELINE_SCHEMA="${BASELINE_SCHEMA:-sparsegpt_official_priority_suite_aggregate_v1}"

PYTHON_BIN="${PYTHON:-/home/cike/jjy/envs/rasp_qwen3_eval/bin/python}"
read -r -a PREPARE_GPUS <<< "${FINAL_GPUS:-0 1 2 3 4 5 6 7}"
export PREPARE_GPU="${PREPARE_GPU:-${PREPARE_GPUS[0]}}"

CUDA_VISIBLE_DEVICES="${PREPARE_GPU}" \
HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}" \
HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}" \
STAGE_WORKFLOW_ROOT="${RUN_ROOT}" \
"${PYTHON_BIN}" -m src.main_stage_calibrated_pruning \
  --config "${CONFIG}" \
  --profile "${PROFILE:-pilot}" \
  --stage preflight \
  --force

CUDA_VISIBLE_DEVICES="${PREPARE_GPU}" \
HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}" \
HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}" \
STAGE_WORKFLOW_ROOT="${RUN_ROOT}" \
STAGE_FINAL_METHODS="${STAGE_FINAL_METHODS}" \
"${PYTHON_BIN}" -m src.main_stage_calibrated_pruning \
  --config "${CONFIG}" \
  --profile "${PROFILE:-pilot}" \
  --stage prepare_sparsegpt \
  --force

exec bash scripts/run_griffin_prompt_priority_suite_gpu.sh
