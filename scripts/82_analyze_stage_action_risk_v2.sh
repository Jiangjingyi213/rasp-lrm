#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-python3}"
ROOT="${STAGE_ACTION_RISK_V2_ROOT:-runs/07_stage_aware/09_stage_action_risk_v2}"
"${PYTHON}" -m src.main_analyze_stage_action_risk_v2 \
  --dataset "${ROOT}/data/01_stage_action_risk_dataset.jsonl" \
  --hidden-states "${ROOT}/data/01_stage_action_risk_hidden_states.pt" \
  --output-dir "${ROOT}/analysis" \
  --folds "${STAGE_ACTION_RISK_V2_FOLDS:-5}" \
  --pca-dim "${STAGE_ACTION_RISK_V2_PCA_DIM:-64}" \
  --seed "${STAGE_ACTION_RISK_V2_SEED:-1}"
