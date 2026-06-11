#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-python3}"
OUTPUT_DIR="${OUTPUT_DIR:-runs/03_rasp_zero/03_runtime_router/rasp_zero_runtime_router}"
RUN_ROOT="${RUN_ROOT:-runs/03_rasp_zero/02_runtime_banks/rasp_zero_runtime_bank_formal}"

RUN_DIRS=("${RUN_ROOT}"/*)
if [[ ! -d "${RUN_DIRS[0]}" ]]; then
  echo "No runtime-bank shards found under ${RUN_ROOT}" >&2
  exit 1
fi

"${PYTHON}" -m src.main_prepare_runtime_router_data \
  --run-dirs "${RUN_DIRS[@]}" \
  --output-dir "${OUTPUT_DIR}" \
  --budgets 0.05 0.10 0.20
