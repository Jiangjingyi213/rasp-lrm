#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-python3}"
ROOT="${ACTION_RISK_RUN_ROOT:-runs/07_stage_aware/06_action_risk_pilot}"
"${PYTHON}" -m src.main_train_action_risk_pilot \
  --dataset "${ROOT}/data/01_action_risk_dataset.jsonl" \
  --hidden-states "${ROOT}/data/01_action_risk_hidden_states.pt" \
  --output-dir "${ROOT}/analysis" \
  --folds "${ACTION_RISK_FOLDS:-5}" \
  --pca-dim "${ACTION_RISK_PCA_DIM:-64}" \
  --seed "${ACTION_RISK_SEED:-1}"
