#!/usr/bin/env bash
set -euo pipefail

CONFIG="${1:-configs/exp_rasp_zero_runtime_bank_gsm8k_smoke.yaml}"
PYTHON="${PYTHON:-python3}"
RESET_SMOKE_RUN="${RESET_SMOKE_RUN:-1}"
REUSE_RUNTIME_TRAJECTORIES="${REUSE_RUNTIME_TRAJECTORIES:-1}"

if [[ "${RESET_SMOKE_RUN}" == "1" && "${CONFIG}" == *"_smoke.yaml" ]]; then
  RUN_DIR="$("${PYTHON}" -c 'import sys; from src.utils.io import read_yaml; print(read_yaml(sys.argv[1])["paths"]["run_dir"])' "${CONFIG}")"
  rm -rf "${RUN_DIR}"
fi
TRAJECTORIES="$("${PYTHON}" -c 'import sys; from src.utils.io import read_yaml; print(read_yaml(sys.argv[1])["paths"]["trajectories"])' "${CONFIG}")"
if [[ "${REUSE_RUNTIME_TRAJECTORIES}" == "1" && -s "${TRAJECTORIES}" ]]; then
  echo "reuse dense trajectories: ${TRAJECTORIES}"
else
  "${PYTHON}" -m src.main_generate --config "${CONFIG}"
fi
"${PYTHON}" -m src.main_segment --config "${CONFIG}"
"${PYTHON}" -m src.main_counterfactual_prune --config "${CONFIG}"
"${PYTHON}" -m src.main_validate_runtime_bank --config "${CONFIG}"
