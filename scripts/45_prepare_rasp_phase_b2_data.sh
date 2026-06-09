#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-python3}"
SOURCE_ROOT="${SOURCE_ROOT:-runs/rasp_phase_b_aligned_bank_12w}"
OUTPUT_ROOT="${OUTPUT_ROOT:-runs/rasp_phase_b2}"
if [[ ! -d "${SOURCE_ROOT}" && -d "runs/未命名" ]]; then
  SOURCE_ROOT="runs/未命名"
fi
RUN_DIRS=()
while IFS= read -r run_dir; do
  RUN_DIRS+=("${run_dir}")
done < <(find "${SOURCE_ROOT}" -mindepth 1 -maxdepth 1 -type d | sort)
[[ "${#RUN_DIRS[@]}" -gt 0 ]] || { echo "No aligned-bank shards found under ${SOURCE_ROOT}" >&2; exit 1; }

"${PYTHON}" -m src.main_prepare_rasp_phase_b2_data \
  --run-dirs "${RUN_DIRS[@]}" \
  --output-dir "${OUTPUT_ROOT}/data" \
  --seeds ${PHASE_B2_SEEDS:-1 2 3}
