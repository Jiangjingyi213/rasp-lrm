#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-python3}"

for CONFIG in \
  configs/eval_griffin_qwen3_gsm8k_smoke_d10.yaml \
  configs/eval_griffin_qwen3_gsm8k_smoke_d09.yaml \
  configs/eval_griffin_qwen3_gsm8k_smoke_d08.yaml \
  configs/eval_griffin_qwen3_gsm8k_smoke.yaml
do
  echo "==> $CONFIG"
  "$PYTHON" -m src.main_generate --config "$CONFIG"
done
