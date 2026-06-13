#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-python3}"
SOURCE_ROOT="${ACTION_RISK_SOURCE_ROOT:-runs/07_stage_aware/06_action_risk_pilot}"
OUTPUT_ROOT="${ACTION_RISK_LEARNED_ROOT:-runs/07_stage_aware/08_action_risk_learned_single_window_pilot}"

"${PYTHON}" -m src.main_train_action_risk_controller \
  --dataset "${SOURCE_ROOT}/data/01_action_risk_dataset.jsonl" \
  --hidden-states "${SOURCE_ROOT}/data/01_action_risk_hidden_states.pt" \
  --oof-predictions "${SOURCE_ROOT}/analysis/02_action_risk_oof_predictions.jsonl" \
  --output-dir "${OUTPUT_ROOT}/checkpoints" \
  --pca-dim "${ACTION_RISK_PCA_DIM:-64}" \
  --seed "${ACTION_RISK_SEED:-1}"
