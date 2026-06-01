#!/usr/bin/env bash
set -euo pipefail

CONFIG="${1:-configs/exp_rasp_zero_runtime_bank_gsm8k_smoke.yaml}"
PYTHON="${PYTHON:-python3}"
RESET_SMOKE_RUN="${RESET_SMOKE_RUN:-1}"

if [[ "${CONFIG}" == "configs/exp_rasp_zero_runtime_bank_gsm8k_smoke.yaml" && "${RESET_SMOKE_RUN}" == "1" ]]; then
  rm -rf runs/rasp_zero_runtime_bank_gsm8k_smoke
fi
"${PYTHON}" -m src.main_generate --config "${CONFIG}"
"${PYTHON}" -m src.main_segment --config "${CONFIG}"
"${PYTHON}" -m src.main_counterfactual_prune --config "${CONFIG}"
