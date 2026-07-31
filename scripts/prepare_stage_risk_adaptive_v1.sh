#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"
RUN_ROOT="${STAGE_RISK_ROOT:-runs/08_stage_calibrated_pruning/09_stage_risk_adaptive_v1}"
CONFIG_ROOT="${STAGE_RISK_CONFIG_ROOT:-configs/generated_stage_risk_adaptive_v1}"
mkdir -p "${RUN_ROOT}"/{00_preflight,01_stage_counterfactual_bank,02_stage_sensitivity_analysis,03_controller_oof,04_dev_matched_t30,05_full_qwen3_1p7b,06_transfer_qwen3_4b,07_figures_tables}
for dir in "${RUN_ROOT}"/*/; do
  test -f "${dir}/status.json" || printf '{"status":"not_started"}\n' > "${dir}/status.json"
  test -f "${dir}/manifest.json" || printf '{"schema":"stage_risk_adaptive_v1"}\n' > "${dir}/manifest.json"
  test -f "${dir}/README.md" || printf '# Stage-Risk Adaptive v1\n\nThis stage has not started. Its manifest and status files are the source of truth.\n' > "${dir}/README.md"
done
mkdir -p logs/08_stage_calibrated_pruning/09_stage_risk_adaptive_v1/{00_preflight,01_bank,03_controller,04_dev,05_full,06_transfer}
mkdir -p "${CONFIG_ROOT}"
if [[ -n "${STAGE_RISK_CONFIG_TEMPLATE:-}" ]]; then
  cp "${STAGE_RISK_CONFIG_TEMPLATE}" "${CONFIG_ROOT}/stage_risk_adaptive_v1_frozen.yaml"
fi
printf '%s\n' "prepared ${RUN_ROOT}"
