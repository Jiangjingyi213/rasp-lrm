#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-python3}"
RUN_ROOT="${RUN_ROOT:-runs/rasp_zero_runtime_bank_formal}"
OUTPUT_ROOT="${OUTPUT_ROOT:-runs/rasp_train_v1}"

RUN_DIRS=()
for path in "${RUN_ROOT}"/*; do
  if [[ -d "${path}" ]]; then
    RUN_DIRS+=("${path}")
  fi
done
if [[ "${#RUN_DIRS[@]}" -eq 0 ]]; then
  echo "No runtime-bank shards found under ${RUN_ROOT}" >&2
  exit 1
fi

"${PYTHON}" -m src.main_prepare_runtime_router_data \
  --run-dirs "${RUN_DIRS[@]}" \
  --output-dir "${OUTPUT_ROOT}/common" \
  --budgets 0.15 0.20

for budget in 0.15 0.20; do
  tag="b${budget#0.}"
  tag="${tag/./}"
  "${PYTHON}" -m src.main_prepare_rasp_train_data \
    --merged-dataset "${OUTPUT_ROOT}/common/05_probe_dataset_merged.jsonl" \
    --merged-hidden-states "${OUTPUT_ROOT}/common/05_probe_hidden_states_merged.pt" \
    --output-dir "${OUTPUT_ROOT}/${tag}" \
    --budgets "${budget}" \
    --ratio-field monotonic_safe_ratio \
    --ratios 0.00 0.02 0.05 0.10 0.20 0.30 0.40
done
