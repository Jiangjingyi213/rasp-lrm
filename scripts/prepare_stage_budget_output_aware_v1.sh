#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

RUN_ROOT="${RUN_ROOT:-runs/10_stage_budget_output_aware_v1}"
LOG_ROOT="${LOG_ROOT:-logs/10_stage_budget_output_aware_v1}"

mkdir -p \
  "${RUN_ROOT}/00_existing_evidence_audit" \
  "${RUN_ROOT}/01_budget_only_dev" \
  "${RUN_ROOT}/02_output_aware_only_dev" \
  "${RUN_ROOT}/03_combined_dev" \
  "${RUN_ROOT}/04_seed42_confirmation" \
  "${RUN_ROOT}/05_frozen_full" \
  "${RUN_ROOT}/06_figures_tables" \
  "${RUN_ROOT}/07_budget_v2_grid_repair/00_smoke_actual_pruning" \
  "${RUN_ROOT}/07_budget_v2_grid_repair/01_dev_seed3" \
  "${RUN_ROOT}/07_budget_v2_grid_repair/02_seed42_confirmation" \
  "${RUN_ROOT}/07_budget_v2_grid_repair/03_frozen_full" \
  "${LOG_ROOT}/00_preflight" \
  "${LOG_ROOT}/01_budget_only_dev" \
  "${LOG_ROOT}/02_output_aware_only_dev" \
  "${LOG_ROOT}/03_combined_dev" \
  "${LOG_ROOT}/04_seed42_confirmation" \
  "${LOG_ROOT}/05_frozen_full" \
  "${LOG_ROOT}/07_budget_v2_grid_repair/00_smoke_actual_pruning" \
  "${LOG_ROOT}/07_budget_v2_grid_repair/01_dev_seed3" \
  "${LOG_ROOT}/07_budget_v2_grid_repair/02_seed42_confirmation" \
  "${LOG_ROOT}/07_budget_v2_grid_repair/03_frozen_full"

for dir in \
  "${RUN_ROOT}/00_existing_evidence_audit" \
  "${RUN_ROOT}/01_budget_only_dev" \
  "${RUN_ROOT}/02_output_aware_only_dev" \
  "${RUN_ROOT}/03_combined_dev" \
  "${RUN_ROOT}/04_seed42_confirmation" \
  "${RUN_ROOT}/05_frozen_full" \
  "${RUN_ROOT}/06_figures_tables" \
  "${RUN_ROOT}/07_budget_v2_grid_repair/00_smoke_actual_pruning" \
  "${RUN_ROOT}/07_budget_v2_grid_repair/01_dev_seed3" \
  "${RUN_ROOT}/07_budget_v2_grid_repair/02_seed42_confirmation" \
  "${RUN_ROOT}/07_budget_v2_grid_repair/03_frozen_full"; do
  if [[ ! -f "${dir}/README.md" ]]; then
    cat > "${dir}/README.md" <<EOF
# $(basename "${dir}")

Stage-Budget + Output-Aware RASP v1 artifact directory.
EOF
  fi
  if [[ ! -f "${dir}/manifest.json" ]]; then
    cat > "${dir}/manifest.json" <<EOF
{
  "schema": "stage_budget_output_aware_v1_manifest",
  "path": "${dir}",
  "created_by": "scripts/prepare_stage_budget_output_aware_v1.sh"
}
EOF
  fi
  if [[ ! -f "${dir}/status.json" ]]; then
    cat > "${dir}/status.json" <<EOF
{
  "status": "initialized"
}
EOF
  fi
done

cat > "${RUN_ROOT}/00_existing_evidence_audit/README.md" <<'EOF'
# Existing Evidence Audit

This v10 line starts from the existing negative/mixed evidence:

- Old fixed stage-budget ablations under `runs/08_stage_calibrated_pruning/` changed
  hand-written per-stage ratios, but did not implement online global budget debt
  control.
- Old core/residual prior experiments showed that merely renaming or interpolating
  priors is not enough.
- v10 therefore isolates two untested mechanisms: online stage budget scheduling
  and output-aware channel scoring.
EOF

cat > "${RUN_ROOT}/00_existing_evidence_audit/status.json" <<'EOF'
{
  "status": "completed",
  "gpu_required": false,
  "conclusion": "Do not repeat old prior-name ablations; test budget scheduling and output-aware scoring separately."
}
EOF

if ! grep -q "10_stage_budget_output_aware_v1" runs/README.md 2>/dev/null; then
  cat >> runs/README.md <<'EOF'

## 10_stage_budget_output_aware_v1

New optimization line focused on two mechanisms: online stage budget scheduling
under a 34% actual-pruning target, and output-aware channel scoring. Results are
kept in `runs/10_stage_budget_output_aware_v1/`; logs mirror to
`logs/10_stage_budget_output_aware_v1/`.
EOF
fi

echo "Prepared ${RUN_ROOT} and ${LOG_ROOT}"
