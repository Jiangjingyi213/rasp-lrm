#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-python3}"
OUTPUT_ROOT="${OUTPUT_ROOT:-runs/rasp_train_v2}"
RISK_ROUTER="${RISK_ROUTER:-runs/rasp_zero_runtime_router/router.pt}"

for tag in b15 b20; do
  args=(
    -m src.main_eval_rasp_train_offline
    --dataset "${OUTPUT_ROOT}/${tag}/11_rasp_train_policy_dataset.jsonl"
    --hidden-states "${OUTPUT_ROOT}/${tag}/11_rasp_train_policy_hidden_states.pt"
    --policy-checkpoint "${OUTPUT_ROOT}/${tag}/rasp_train_policy.pt"
    --output-dir "${OUTPUT_ROOT}/${tag}/offline_eval"
  )
  if [[ -f "${RISK_ROUTER}" ]]; then
    args+=(--risk-router-checkpoint "${RISK_ROUTER}" --risk-threshold "${RASP_ZERO_RISK_THRESHOLD:-0.25}")
  fi
  "${PYTHON}" "${args[@]}"
done
