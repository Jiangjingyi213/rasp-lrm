#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-python3}"
OUTPUT_ROOT="${OUTPUT_ROOT:-runs/rasp_train_v2}"

for tag in b15 b20; do
  if [[ ! -f "${OUTPUT_ROOT}/${tag}/11_rasp_train_policy_dataset.jsonl" ]] ||
     [[ ! -f "${OUTPUT_ROOT}/${tag}/11_rasp_train_policy_hidden_states.pt" ]]; then
    echo "Missing prepared RASP-Train data for ${tag}; run scripts/35_prepare_rasp_train_v1_data.sh first" >&2
    exit 1
  fi
done

for tag in b15 b20; do
  if [[ "${tag}" == "b15" ]]; then
    max_flip="${RASP_TRAIN_B15_MAX_CALIBRATION_FLIP_RATE:-0.06}"
    max_unsafe="${RASP_TRAIN_B15_MAX_CALIBRATION_UNSAFE_RATE:-0.08}"
  else
    max_flip="${RASP_TRAIN_B20_MAX_CALIBRATION_FLIP_RATE:-0.08}"
    max_unsafe="${RASP_TRAIN_B20_MAX_CALIBRATION_UNSAFE_RATE:-0.10}"
  fi
  "${PYTHON}" -m src.main_train_rasp_train_router \
    --dataset "${OUTPUT_ROOT}/${tag}/11_rasp_train_policy_dataset.jsonl" \
    --hidden-states "${OUTPUT_ROOT}/${tag}/11_rasp_train_policy_hidden_states.pt" \
    --output "${OUTPUT_ROOT}/${tag}/rasp_train_policy.pt" \
    --metrics-output "${OUTPUT_ROOT}/${tag}/13_rasp_train_metrics.json" \
    --epochs "${RASP_TRAIN_EPOCHS:-40}" \
    --batch-size "${RASP_TRAIN_BATCH_SIZE:-128}" \
    --lr "${RASP_TRAIN_LR:-5e-4}" \
    --monotonic-weight "${RASP_TRAIN_MONOTONIC_WEIGHT:-1.0}" \
    --ranking-weight "${RASP_TRAIN_RANKING_WEIGHT:-1.0}" \
    --holdout-fraction "${RASP_TRAIN_HOLDOUT_FRACTION:-0.30}" \
    --max-calibration-flip-rate "${max_flip}" \
    --max-calibration-unsafe-rate "${max_unsafe}" \
    --seed "${RASP_TRAIN_SEED:-1}"
done
