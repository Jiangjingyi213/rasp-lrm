#!/usr/bin/env bash
set -euo pipefail

export CONFIG="${CONFIG:-configs/stage_calibrated_pruning/sparsegpt_official_qwen3_4b_full.yaml}"
export RUN_ROOT="${RUN_ROOT:-runs/08_stage_calibrated_pruning/main_pilot_sparsegpt_official_qwen3_4b_full}"
export STAGE_FINAL_METHODS="${STAGE_FINAL_METHODS:-sparsegpt_t30_official}"
export RUN_LABEL="${RUN_LABEL:-sparsegpt_official_qwen3_4b_full}"
export SPARSEGPT_CALIBRATION_BATCH_SIZE="${SPARSEGPT_CALIBRATION_BATCH_SIZE:-4}"

PYTHON_BIN="${PYTHON:-/root/jjy/envs/rasp_qwen3_eval/bin/python}"
read -r -a PREPARE_GPUS <<< "${FINAL_GPUS:-0}"
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

exec bash scripts/run_griffin_prompt_matched_full_gpu.sh
