#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-python3}"
SOURCE_ROOT="${SOURCE_ROOT:-runs/01_motivation}"
STAGE_PROBE_ROOT="${STAGE_PROBE_ROOT:-runs/07_stage_aware/03_s1_three_stage_probe}"
RUN_DIRS=(
  "${SOURCE_ROOT}/formal_qwen3_gsm8k_full_s0"
  "${SOURCE_ROOT}/formal_qwen3_gsm8k_full_s1"
  "${SOURCE_ROOT}/formal_qwen3_math500_full_s0"
  "${SOURCE_ROOT}/formal_qwen3_math500_full_s1"
)

"${PYTHON}" -m src.main_prepare_rasp_stage_probe_data \
  --run-dirs "${RUN_DIRS[@]}" \
  --output-dir "${STAGE_PROBE_ROOT}/data" \
  --audit-size "${STAGE_AUDIT_SIZE:-100}" \
  --exclude-audit-labels "${STAGE_PRIOR_AUDIT_LABELS:-configs/stage_audits/s1_operational_v2_labels.csv}" \
  --min-stage-rows "${STAGE_MIN_ROWS:-100}" \
  --seeds ${STAGE_PROBE_SEEDS:-1 2 3}
