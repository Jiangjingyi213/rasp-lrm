#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-python3}"
PROFILE="${PROFILE:-dense_smoke}"
ROOT="${FULL_TRAJECTORY_ROOT:-runs/07_stage_aware/10_full_trajectory_multi_window}"
case "${PROFILE}" in
  dense_smoke)
    PHASE_DIR="${ROOT}/01_dense_bank_smoke"
    MINIMUM=4
    ;;
  dense_pilot)
    PHASE_DIR="${ROOT}/02_dense_bank_pilot"
    MINIMUM=20
    ;;
  *)
    echo "Unsupported PROFILE=${PROFILE}" >&2
    exit 2
    ;;
esac

"${PYTHON}" -m src.main_prepare_full_trajectory_bank \
  --manifest "configs/generated_full_trajectory_${PROFILE}/manifest.json" \
  --output-dir "${PHASE_DIR}/data" \
  --minimum-problems-per-source "${MINIMUM}" \
  --maximum-problems-per-source "${MINIMUM}"

if [[ "${PROFILE}" == "dense_pilot" ]]; then
  "${PYTHON}" -m src.main_analyze_full_trajectory_bank \
    --dataset "${PHASE_DIR}/data/01_full_trajectory_causal_dataset.jsonl" \
    --hidden-states "${PHASE_DIR}/data/01_full_trajectory_causal_hidden_states.pt" \
    --data-summary "${PHASE_DIR}/data/01_full_trajectory_data_summary.json" \
    --output-dir "${PHASE_DIR}/analysis" \
    --folds 5 \
    --pca-dim 64 \
    --seed 1
fi

"${PYTHON}" -m src.main_gate_full_trajectory_bank \
  --profile "${PROFILE}" \
  --phase-dir "${PHASE_DIR}"
