#!/usr/bin/env bash
set -euo pipefail

unset HF_DATASETS_OFFLINE
unset TRANSFORMERS_OFFLINE
unset HF_HUB_OFFLINE

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

PYTHON_BIN="${PYTHON:-/home/cike/jjy/envs/rasp_qwen3_eval/bin/python}"
BASE_RUN_ROOT="runs/09_stage_residual_rasp_v2"
BASE_LOG_ROOT="logs/09_stage_residual_rasp_v2"
FULL_STATUS="${BASE_RUN_ROOT}/04_frozen_full_qwen3_1p7b/status.json"

bash scripts/prepare_stage_residual_rasp_v2.sh

if [[ ! -f "${FULL_STATUS}" ]]; then
  echo "Missing full-stage status: ${FULL_STATUS}" >&2
  echo "Run scripts/run_stage_residual_full_8gpu.sh and inspect the frozen full results before external pressure tests." >&2
  exit 2
fi

if [[ "${CONFIRM_INTERNAL_FULL_PASS:-0}" != "1" ]]; then
  echo "External pressure tests are gated. Set CONFIRM_INTERNAL_FULL_PASS=1 only after the frozen full internal comparison passes." >&2
  exit 2
fi

mkdir -p "${BASE_LOG_ROOT}/05_external" "${BASE_RUN_ROOT}/05_external_pressure_test"

echo "==== START Stage-Residual v2 external pressure: Wanda-C4 seed3 t30 ===="
PYTHON="${PYTHON_BIN}" \
LOG_DIR="${BASE_LOG_ROOT}/05_external/wanda_c4_seed3_t30" \
STAGE_FINAL_METHODS="wanda_c4_seed3_t30" \
C4_CALIBRATION_PATH="${BASE_RUN_ROOT}/05_external_pressure_test/wanda_c4_seed3_calibration/c4_128_seed3.jsonl" \
PRIORITY_CONFIG="configs/generated_stage_residual_rasp_v2/wanda_c4_seed3_qwen3_1p7b_priority_suite.yaml" \
FULL_CONFIG="configs/generated_stage_residual_rasp_v2/wanda_c4_seed3_qwen3_1p7b_full.yaml" \
PRIORITY_ROOT="${BASE_RUN_ROOT}/05_external_pressure_test/wanda_c4_seed3_t30_priority_suite" \
FULL_ROOT="${BASE_RUN_ROOT}/05_external_pressure_test/wanda_c4_seed3_t30_full" \
bash scripts/run_wanda_c4_seed3_qwen3_1p7b_priority_then_full_8gpu.sh

echo "==== START Stage-Residual v2 external pressure: SparseGPT t30 ===="
PYTHON="${PYTHON_BIN}" \
CONFIG="configs/generated_stage_residual_rasp_v2/sparsegpt_official_qwen3_1p7b_full.yaml" \
RUN_ROOT="${BASE_RUN_ROOT}/05_external_pressure_test/sparsegpt_official_t30_full" \
STAGE_FINAL_METHODS="sparsegpt_t30_official" \
RUN_LABEL="stage_residual_v2_sparsegpt_t30_full" \
LOG_DIR="${BASE_LOG_ROOT}/05_external/sparsegpt_t30" \
bash scripts/run_sparsegpt_official_qwen3_1p7b_full_8gpu.sh

"${PYTHON_BIN}" - <<'PY'
import json
from pathlib import Path

root = Path("runs/09_stage_residual_rasp_v2/05_external_pressure_test")
status = {
    "stage": "05_external_pressure_test",
    "status": "completed",
    "baselines": {
        "wanda_c4_seed3_t30_full": str(root / "wanda_c4_seed3_t30_full" / "06_final" / "summary.json"),
        "sparsegpt_t30_full": str(root / "sparsegpt_official_t30_full" / "06_final" / "summary.json"),
    },
}
(root / "status.json").write_text(json.dumps(status, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
PY

echo "==== ALL DONE Stage-Residual v2 external pressure tests ===="
