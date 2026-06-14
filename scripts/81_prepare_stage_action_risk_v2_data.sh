#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-python3}"
ROOT="${STAGE_ACTION_RISK_V2_ROOT:-runs/07_stage_aware/09_stage_action_risk_v2}"
"${PYTHON}" -m src.main_prepare_stage_action_risk_v2 \
  --bank-root "${ROOT}/bank" \
  --output-dir "${ROOT}/data" \
  --manifest "configs/generated_stage_action_risk_v2/manifest.json" \
  --min-complete-problems-per-source "${MIN_COMPLETE_PROBLEMS_PER_SOURCE:-100}"
