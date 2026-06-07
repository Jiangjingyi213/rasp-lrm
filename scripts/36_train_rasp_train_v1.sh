#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-python3}"
OUTPUT_ROOT="${OUTPUT_ROOT:-runs/rasp_train_v2_1}"

for tag in b15 b20; do
  if [[ ! -f "${OUTPUT_ROOT}/${tag}/11_rasp_train_policy_dataset.jsonl" ]] ||
     [[ ! -f "${OUTPUT_ROOT}/${tag}/11_rasp_train_policy_hidden_states.pt" ]]; then
    echo "Missing prepared RASP-Train data for ${tag}; run scripts/35_prepare_rasp_train_v1_data.sh first" >&2
    exit 1
  fi
done

mkdir -p "${OUTPUT_ROOT}/shared"
"${PYTHON}" -m src.main_train_rasp_train_router \
  --dataset "${OUTPUT_ROOT}/b15/11_rasp_train_policy_dataset.jsonl" \
  --hidden-states "${OUTPUT_ROOT}/b15/11_rasp_train_policy_hidden_states.pt" \
  --equivalent-label-dataset "${OUTPUT_ROOT}/b20/11_rasp_train_policy_dataset.jsonl" \
  --output "${OUTPUT_ROOT}/shared/rasp_train_policy.pt" \
  --metrics-output "${OUTPUT_ROOT}/shared/13_rasp_train_metrics.json" \
  --epochs "${RASP_TRAIN_EPOCHS:-40}" \
  --batch-size "${RASP_TRAIN_BATCH_SIZE:-128}" \
  --lr "${RASP_TRAIN_LR:-5e-4}" \
  --monotonic-weight "${RASP_TRAIN_MONOTONIC_WEIGHT:-1.0}" \
  --ranking-weight "${RASP_TRAIN_RANKING_WEIGHT:-1.0}" \
  --holdout-fraction "${RASP_TRAIN_HOLDOUT_FRACTION:-0.30}" \
  --calibration-budgets 0.15 0.20 \
  --max-calibration-flip-rates \
    "${RASP_TRAIN_B15_MAX_CALIBRATION_FLIP_RATE:-0.06}" \
    "${RASP_TRAIN_B20_MAX_CALIBRATION_FLIP_RATE:-0.08}" \
  --max-calibration-unsafe-rates \
    "${RASP_TRAIN_B15_MAX_CALIBRATION_UNSAFE_RATE:-0.08}" \
    "${RASP_TRAIN_B20_MAX_CALIBRATION_UNSAFE_RATE:-0.10}" \
  --calibration-folds "${RASP_TRAIN_CALIBRATION_FOLDS:-3}" \
  --seed "${RASP_TRAIN_SEED:-1}"
