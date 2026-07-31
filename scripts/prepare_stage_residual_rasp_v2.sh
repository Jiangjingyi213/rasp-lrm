#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

RUN_ROOT="${STAGE_RESIDUAL_ROOT:-runs/09_stage_residual_rasp_v2}"
LOG_ROOT="${STAGE_RESIDUAL_LOG_ROOT:-logs/09_stage_residual_rasp_v2}"
for dir in 00_preflight 01_existing_evidence_audit 02_stage_residual_dev 03_output_aware_continuity_dev 04_frozen_full_qwen3_1p7b 05_external_pressure_test 06_figures_tables; do
  mkdir -p "${RUN_ROOT}/${dir}"
  test -f "${RUN_ROOT}/${dir}/README.md" || printf '# Stage-Residual RASP v2\n\nStatus is recorded in status.json.\n' > "${RUN_ROOT}/${dir}/README.md"
  test -f "${RUN_ROOT}/${dir}/manifest.json" || printf '{"schema":"stage_residual_rasp_v2"}\n' > "${RUN_ROOT}/${dir}/manifest.json"
  test -f "${RUN_ROOT}/${dir}/status.json" || printf '{"status":"not_started"}\n' > "${RUN_ROOT}/${dir}/status.json"
done
mkdir -p "${LOG_ROOT}"/{00_preflight,02_dev,03_dev,04_full,05_external}
printf 'prepared %s\n' "${RUN_ROOT}"
