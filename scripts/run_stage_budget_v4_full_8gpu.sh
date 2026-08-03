#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

PYTHON_BIN="${PYTHON:-/home/cike/jjy/envs/rasp_qwen3_eval/bin/python}"
RUN_ROOT="${RUN_ROOT:-runs/10_stage_budget_output_aware_v1}"
LOG_ROOT="${LOG_ROOT:-logs/10_stage_budget_output_aware_v1}"
SOURCE_ROOT="${SOURCE_ROOT:-runs/08_stage_calibrated_pruning/main_pilot_mixed_reasoning_seed3_t30_math_safe_full}"
CONFIG="${CONFIG:-configs/generated_stage_budget_output_aware_v1/budget_v4_full.template.yaml}"
DEV_SELECTION="${RUN_ROOT}/09_budget_v4_perf_guarded_32p/01_dev_seed3/budget_v4_dev_selection.json"
SEED42_SELECTION="${RUN_ROOT}/09_budget_v4_perf_guarded_32p/02_seed42_confirmation/budget_v4_seed42_selection.json"

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
  echo "Budget v4 dev did not pass; stopping before full."
  exit 0
fi

seed42_passed="$(SELECTION="${SEED42_SELECTION}" "${PYTHON_BIN}" - <<'PY'
import json
import os
from pathlib import Path
path = Path(os.environ["SELECTION"])
print("1" if path.exists() and json.load(open(path, "r", encoding="utf-8")).get("phase_passed") else "0")
PY
)"
if [[ "${seed42_passed}" != "1" ]]; then
  echo "Budget v4 seed42 confirmation did not pass; stopping before full."
  exit 0
fi

bash scripts/prepare_stage_budget_output_aware_v1.sh

METHODS="structured_dense,current_t30_math_safe,dynamic_global_activation_fixed_t30,${selected_method}"
for seed in ${STAGE_BUDGET_SEEDS:-3}; do
  phase_root="${RUN_ROOT}/09_budget_v4_perf_guarded_32p/03_frozen_full/seed_${seed}"
  CONFIG="${CONFIG}" RUN_ROOT="${phase_root}" SOURCE_ROOT="${SOURCE_ROOT}" LOG_DIR="${LOG_ROOT}/09_budget_v4_perf_guarded_32p/03_frozen_full/seed_${seed}" \
    STAGE_SEED="${seed}" STAGE_FINAL_SEEDS="${seed}" DATASETS_OVERRIDE="gsm8k math500" STAGE_FINAL_EVAL_LIMIT="${STAGE_FINAL_EVAL_LIMIT:--1}" STAGE_FINAL_METHODS="${METHODS}" \
    bash scripts/run_t30_math_safe_priority_suite_8gpu.sh
done
