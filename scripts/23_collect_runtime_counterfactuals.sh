#!/usr/bin/env bash
set -euo pipefail

CONFIG="${1:-configs/exp_rasp_zero_runtime_bank_gsm8k_smoke.yaml}"
PYTHON="${PYTHON:-python3}"

"${PYTHON}" -m src.main_generate --config "${CONFIG}"
"${PYTHON}" -m src.main_segment --config "${CONFIG}"
"${PYTHON}" -m src.main_counterfactual_prune --config "${CONFIG}"
