#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON:-python3}"
OUTPUT_DIR="${OUTPUT_DIR:-runs/08_stage_calibrated_pruning/orca_pilot_fast_policy_selection_labels_v4}"
ROOT_TEMPLATE="${ROOT_TEMPLATE:-runs/08_stage_calibrated_pruning/main_pilot_fast_orca_math_labels_v4_seed{seed}}"

read -r -a seeds <<< "${PILOT_SEEDS:-1 2 3 4}"

roots=()
for seed in "${seeds[@]}"; do
  roots+=("${ROOT_TEMPLATE/\{seed\}/${seed}}")
done

"${PYTHON_BIN}" -m src.stage_calibration.policy_selection \
  --roots "${roots[@]}" \
  --output-dir "${OUTPUT_DIR}"

echo "Wrote ${OUTPUT_DIR}/policy_selection.json"
echo "Wrote ${OUTPUT_DIR}/policy_selection.md"
