#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-python3}"
ROOT="${ACTION_RISK_RUN_ROOT:-runs/07_stage_aware/06_action_risk_pilot}"
"${PYTHON}" -m src.main_prepare_action_risk_pilot \
  --bank-root "${ROOT}/bank" \
  --output-dir "${ROOT}/data" \
  --manifest "configs/generated_action_risk_pilot/manifest.json" \
  --min-dense-correct-per-source "${MIN_DENSE_CORRECT_PER_SOURCE:-100}"
