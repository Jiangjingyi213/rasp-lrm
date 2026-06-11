#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-python3}"
SOURCE_ROOT="${SOURCE_ROOT:-runs/04_rasp_train/01_legacy/rasp_train_v2_1}"
OUTPUT_ROOT="${OUTPUT_ROOT:-runs/04_rasp_train/02_fair_benchmark/rasp_train_fair_benchmark}"

"${PYTHON}" -m src.main_prepare_rasp_train_fair_benchmark \
  --b15-dataset "${SOURCE_ROOT}/b15/11_rasp_train_policy_dataset.jsonl" \
  --b20-dataset "${SOURCE_ROOT}/b20/11_rasp_train_policy_dataset.jsonl" \
  --output-dir "${OUTPUT_ROOT}/split_manifests" \
  --seeds ${FAIR_SEEDS:-1 2 3}
