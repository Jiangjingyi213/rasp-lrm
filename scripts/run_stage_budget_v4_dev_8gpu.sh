#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

PYTHON_BIN="${PYTHON:-/home/cike/jjy/envs/rasp_qwen3_eval/bin/python}"
RUN_ROOT="${RUN_ROOT:-runs/10_stage_budget_output_aware_v1}"
LOG_ROOT="${LOG_ROOT:-logs/10_stage_budget_output_aware_v1}"
SOURCE_ROOT="${SOURCE_ROOT:-runs/08_stage_calibrated_pruning/main_pilot_mixed_reasoning_seed3_t30_math_safe_full}"
CONFIG="${CONFIG:-configs/generated_stage_budget_output_aware_v1/budget_v4_dev.yaml}"
SEEDS="${STAGE_BUDGET_SEEDS:-3}"
SMOKE_SELECTION="${RUN_ROOT}/09_budget_v4_perf_guarded_32p/00_smoke_32p/budget_v4_smoke_selection.json"
REFERENCE_METHOD="dynamic_global_activation_budgeted_v3_calibrated"

selected_method="$(SELECTION="${SMOKE_SELECTION}" "${PYTHON_BIN}" - <<'PY'
import json
import os
from pathlib import Path
path = Path(os.environ["SELECTION"])
if not path.exists():
    raise SystemExit(f"Missing smoke selection: {path}")
data = json.load(open(path, "r", encoding="utf-8"))
if not data.get("phase_passed"):
    print("")
else:
    print(data.get("selected_candidate_method") or "")
PY
)"
if [[ -z "${selected_method}" ]]; then
  echo "Budget v4 smoke did not pass; stopping before dev."
  exit 0
fi

bash scripts/prepare_stage_budget_output_aware_v1.sh

for seed in ${SEEDS}; do
  phase_root="${RUN_ROOT}/09_budget_v4_perf_guarded_32p/01_dev_seed3/seed_${seed}"
  CONFIG="${CONFIG}" RUN_ROOT="${phase_root}" SOURCE_ROOT="${SOURCE_ROOT}" LOG_DIR="${LOG_ROOT}/09_budget_v4_perf_guarded_32p/01_dev_seed3/seed_${seed}" \
    STAGE_SEED="${seed}" STAGE_FINAL_SEEDS="${seed}" DATASETS_OVERRIDE="gsm8k math500" STAGE_FINAL_EVAL_LIMIT="${STAGE_BUDGET_V4_DEV_LIMIT:-520}" STAGE_FINAL_METHODS="${selected_method},${REFERENCE_METHOD}" \
    bash scripts/run_t30_math_safe_priority_suite_8gpu.sh
done

"${PYTHON_BIN}" -m src.main_select_stage_budget_output_aware \
  --phase a2 \
  --phase-label A4_budget_v4_32p \
  --selection-mode dev \
  --target-min 0.320 \
  --target-max 0.335 \
  --target-center 0.325 \
  --phase-root "${RUN_ROOT}/09_budget_v4_perf_guarded_32p/01_dev_seed3" \
  --reference-root "${RUN_ROOT}/01_budget_only_dev" \
  --baseline-method dynamic_global_activation_fixed_t30 \
  --performance-reference-method "${REFERENCE_METHOD}" \
  --max-performance-reference-accuracy-drop 0.02 \
  --max-fallback-delta 0.01 \
  --max-truncation-delta 0.01 \
  --candidate-methods "${selected_method}" \
  --output "${RUN_ROOT}/09_budget_v4_perf_guarded_32p/01_dev_seed3/budget_v4_dev_selection.json"
