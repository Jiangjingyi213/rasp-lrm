#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

PYTHON_BIN="${PYTHON:-/home/cike/jjy/envs/rasp_qwen3_eval/bin/python}"
RUN_ROOT="${RUN_ROOT:-runs/10_stage_budget_output_aware_v1}"
LOG_ROOT="${LOG_ROOT:-logs/10_stage_budget_output_aware_v1}"
SOURCE_ROOT="${SOURCE_ROOT:-runs/08_stage_calibrated_pruning/main_pilot_mixed_reasoning_seed3_t30_math_safe_full}"
CONFIG="${CONFIG:-configs/generated_stage_budget_output_aware_v1/budget_v4_seed42.yaml}"
DEV_SELECTION="${RUN_ROOT}/09_budget_v4_perf_guarded_32p/01_dev_seed3/budget_v4_dev_selection.json"
REFERENCE_METHOD="dynamic_global_activation_budgeted_v3_calibrated"
BASELINE_METHOD="dynamic_global_activation_fixed_t30"

selected_method="$(SELECTION="${DEV_SELECTION}" "${PYTHON_BIN}" - <<'PY'
import json
import os
from pathlib import Path
path = Path(os.environ["SELECTION"])
if not path.exists():
    raise SystemExit(f"Missing dev selection: {path}")
data = json.load(open(path, "r", encoding="utf-8"))
if not data.get("phase_passed"):
    print("")
else:
    print(data.get("selected_candidate_method") or "")
PY
)"
if [[ -z "${selected_method}" ]]; then
  echo "Budget v4 dev did not pass; stopping before seed42 confirmation."
  exit 0
fi

bash scripts/prepare_stage_budget_output_aware_v1.sh

phase_root="${RUN_ROOT}/09_budget_v4_perf_guarded_32p/02_seed42_confirmation/seed_42"
CONFIG="${CONFIG}" RUN_ROOT="${phase_root}" SOURCE_ROOT="${SOURCE_ROOT}" LOG_DIR="${LOG_ROOT}/09_budget_v4_perf_guarded_32p/02_seed42_confirmation/seed_42" \
  STAGE_SEED=42 STAGE_FINAL_SEEDS=42 DATASETS_OVERRIDE="gsm8k math500" STAGE_FINAL_EVAL_LIMIT="${STAGE_BUDGET_V4_DEV_LIMIT:-520}" STAGE_FINAL_METHODS="${selected_method},${REFERENCE_METHOD},${BASELINE_METHOD}" \
  bash scripts/run_t30_math_safe_priority_suite_8gpu.sh

"${PYTHON_BIN}" -m src.main_select_stage_budget_output_aware \
  --phase a2 \
  --phase-label A4_budget_v4_32p \
  --selection-mode confirm \
  --seed 42 \
  --target-min 0.320 \
  --target-max 0.335 \
  --target-center 0.325 \
  --phase-root "${RUN_ROOT}/09_budget_v4_perf_guarded_32p/02_seed42_confirmation" \
  --reference-root "${RUN_ROOT}/01_budget_only_dev" \
  --baseline-method dynamic_global_activation_fixed_t30 \
  --performance-reference-method "${REFERENCE_METHOD}" \
  --max-performance-reference-accuracy-drop 0.02 \
  --max-fallback-delta 0.01 \
  --max-truncation-delta 0.01 \
  --candidate-methods "${selected_method}" \
  --output "${RUN_ROOT}/09_budget_v4_perf_guarded_32p/02_seed42_confirmation/budget_v4_seed42_selection.json"
