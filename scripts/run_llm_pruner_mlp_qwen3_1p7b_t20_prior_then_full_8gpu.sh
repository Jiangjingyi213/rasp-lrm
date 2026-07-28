#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

export CONFIG="${CONFIG:-configs/stage_calibrated_pruning/llm_pruner_mlp_qwen3_1p7b_t20_full.yaml}"
export STAGE_FINAL_METHODS="${STAGE_FINAL_METHODS:-llm_pruner_mlp_t20_matched}"
export BASELINE_LABEL="${BASELINE_LABEL:-LLM-Pruner-MLP Qwen3-1.7B t20}"
export BASELINE_SCHEMA="${BASELINE_SCHEMA:-llm_pruner_mlp_qwen3_1p7b_t20_prior_then_full_v1}"

RUN_PRIOR_FIRST="${RUN_PRIOR_FIRST:-1}"
RUN_FULL_AFTER_PRIOR="${RUN_FULL_AFTER_PRIOR:-1}"

PRIOR_PROFILE="${PRIOR_PROFILE:-smoke}"
PRIOR_RUN_ROOT="${PRIOR_RUN_ROOT:-runs/08_stage_calibrated_pruning/main_prior_llm_pruner_mlp_qwen3_1p7b_t20}"
PRIOR_EVAL_LIMIT="${PRIOR_EVAL_LIMIT:-16}"
PRIOR_RUN_LABEL="${PRIOR_RUN_LABEL:-llm_pruner_mlp_qwen3_1p7b_t20_prior}"

FULL_PROFILE="${FULL_PROFILE:-pilot}"
FULL_RUN_ROOT="${FULL_RUN_ROOT:-runs/08_stage_calibrated_pruning/main_pilot_llm_pruner_mlp_qwen3_1p7b_t20_full}"
FULL_EVAL_LIMIT="${FULL_EVAL_LIMIT:--1}"
FULL_RUN_LABEL="${FULL_RUN_LABEL:-llm_pruner_mlp_qwen3_1p7b_t20_full}"

if [[ "${RUN_PRIOR_FIRST}" == "1" ]]; then
  echo "START LLM-Pruner-style t20 prior/smoke run: root=${PRIOR_RUN_ROOT}, limit=${PRIOR_EVAL_LIMIT}"
  PROFILE="${PRIOR_PROFILE}" \
  RUN_ROOT="${PRIOR_RUN_ROOT}" \
  STAGE_FINAL_EVAL_LIMIT="${PRIOR_EVAL_LIMIT}" \
  RUN_LABEL="${PRIOR_RUN_LABEL}" \
  bash scripts/run_griffin_prompt_matched_full_gpu.sh
  echo "DONE LLM-Pruner-style t20 prior/smoke run: ${PRIOR_RUN_ROOT}/06_final/summary.json"
fi

if [[ "${RUN_FULL_AFTER_PRIOR}" == "1" ]]; then
  echo "START LLM-Pruner-style t20 full run: root=${FULL_RUN_ROOT}, limit=${FULL_EVAL_LIMIT}"
  PROFILE="${FULL_PROFILE}" \
  RUN_ROOT="${FULL_RUN_ROOT}" \
  STAGE_FINAL_EVAL_LIMIT="${FULL_EVAL_LIMIT}" \
  RUN_LABEL="${FULL_RUN_LABEL}" \
  bash scripts/run_griffin_prompt_matched_full_gpu.sh
  echo "DONE LLM-Pruner-style t20 full run: ${FULL_RUN_ROOT}/06_final/summary.json"
fi
