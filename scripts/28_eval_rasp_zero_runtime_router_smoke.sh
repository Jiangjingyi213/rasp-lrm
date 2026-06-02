#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-python3}"

"${PYTHON}" -m src.main_eval_rasp_zero_runtime \
  --config configs/exp_rasp_zero_runtime_dense_gsm8k_smoke.yaml
"${PYTHON}" -m src.main_eval_rasp_zero_runtime \
  --config configs/exp_rasp_zero_runtime_dense_math500_smoke.yaml
"${PYTHON}" -m src.main_eval_rasp_zero_runtime \
  --config configs/exp_rasp_zero_runtime_router_gsm8k_smoke.yaml
"${PYTHON}" -m src.main_eval_rasp_zero_runtime \
  --config configs/exp_rasp_zero_runtime_router_math500_smoke.yaml
