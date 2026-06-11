#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-python3}"
OUTPUT_DIR="${OUTPUT_DIR:-runs/03_rasp_zero/03_runtime_router/rasp_zero_runtime_router}"
export PYTHON OUTPUT_DIR

bash scripts/26_prepare_runtime_router_data.sh

"${PYTHON}" -m src.main_train_runtime_router \
  --dataset "${OUTPUT_DIR}/09_action_conditioned_risk_dataset.jsonl" \
  --hidden-states "${OUTPUT_DIR}/09_action_conditioned_risk_hidden_states.pt" \
  --output "${OUTPUT_DIR}/router.pt" \
  --metrics-output "${OUTPUT_DIR}/10_router_metrics.json" \
  --epochs 30 \
  --batch-size 128 \
  --lr 1e-3 \
  --val-fraction 0.25 \
  --seed 1
