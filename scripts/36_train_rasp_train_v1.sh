#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-python3}"
OUTPUT_ROOT="${OUTPUT_ROOT:-runs/rasp_train_v1}"

for tag in b15 b20; do
  if [[ ! -f "${OUTPUT_ROOT}/${tag}/11_rasp_train_policy_dataset.jsonl" ]] ||
     [[ ! -f "${OUTPUT_ROOT}/${tag}/11_rasp_train_policy_hidden_states.pt" ]]; then
    echo "Missing prepared RASP-Train data for ${tag}; run scripts/35_prepare_rasp_train_v1_data.sh first" >&2
    exit 1
  fi
done

for tag in b15 b20; do
  "${PYTHON}" -m src.main_train_rasp_train_router \
    --dataset "${OUTPUT_ROOT}/${tag}/11_rasp_train_policy_dataset.jsonl" \
    --hidden-states "${OUTPUT_ROOT}/${tag}/11_rasp_train_policy_hidden_states.pt" \
    --output "${OUTPUT_ROOT}/${tag}/rasp_train_policy.pt" \
    --metrics-output "${OUTPUT_ROOT}/${tag}/13_rasp_train_metrics.json" \
    --epochs "${RASP_TRAIN_EPOCHS:-40}" \
    --batch-size "${RASP_TRAIN_BATCH_SIZE:-128}" \
    --lr "${RASP_TRAIN_LR:-5e-4}" \
    --unsafe-weight "${RASP_TRAIN_UNSAFE_WEIGHT:-3.0}" \
    --budget-weight "${RASP_TRAIN_BUDGET_WEIGHT:-2.0}" \
    --val-fraction "${RASP_TRAIN_VAL_FRACTION:-0.25}" \
    --seed "${RASP_TRAIN_SEED:-1}"
done
