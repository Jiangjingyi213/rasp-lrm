#!/usr/bin/env bash
set -euo pipefail

CONFIG="${1:-configs/exp_rasp_zero_runtime_bank_gsm8k_smoke.yaml}"
PYTHON="${PYTHON:-python3}"
RESET_SMOKE_RUN="${RESET_SMOKE_RUN:-1}"

if [[ "${RESET_SMOKE_RUN}" == "1" && "${CONFIG}" == *"_smoke.yaml" ]]; then
  RUN_DIR="$("${PYTHON}" -c 'import sys; from src.utils.io import read_yaml; print(read_yaml(sys.argv[1])["paths"]["run_dir"])' "${CONFIG}")"
  rm -rf "${RUN_DIR}"
fi
"${PYTHON}" -m src.main_generate --config "${CONFIG}"
"${PYTHON}" -m src.main_segment --config "${CONFIG}"
"${PYTHON}" -m src.main_counterfactual_prune --config "${CONFIG}"
"${PYTHON}" -m src.main_validate_runtime_bank --config "${CONFIG}"
