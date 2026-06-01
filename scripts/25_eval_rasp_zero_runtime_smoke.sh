#!/usr/bin/env bash
set -euo pipefail

CONFIG="${1:-configs/exp_rasp_zero_runtime_smoke.yaml}"
PYTHON="${PYTHON:-python3}"

"${PYTHON}" -m src.main_eval_rasp_zero_runtime --config "${CONFIG}"
