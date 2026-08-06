#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

PYTHON_BIN="${PYTHON:-/home/cike/jjy/envs/rasp_qwen3_eval/bin/python}"
RUN_ROOT="${RUN_ROOT:-runs/10_stage_budget_output_aware_v1}"
LOG_ROOT="${LOG_ROOT:-logs/10_stage_budget_output_aware_v1}"
SOURCE_ROOT="${SOURCE_ROOT:-runs/08_stage_calibrated_pruning/main_pilot_mixed_reasoning_seed3_t30_math_safe_full}"
CONFIG="${CONFIG:-configs/generated_stage_budget_output_aware_v1/budget_v41_full.template.yaml}"
DEV_SELECTION="${RUN_ROOT}/10_budget_v41_stage_lift_32p/01_dev_seed3/budget_v41_dev_selection.json"

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
  echo "Budget v4.1 dev seed3 did not pass; stopping before exploratory full."
  exit 0
fi

if [[ "${selected_method}" != "dynamic_global_activation_budgeted_v41_stage_lift_plus_32p" ]]; then
  echo "Unexpected v4.1 selected method: ${selected_method}" >&2
  echo "Expected dynamic_global_activation_budgeted_v41_stage_lift_plus_32p for seed3 exploratory full." >&2
  exit 2
fi

bash scripts/prepare_stage_budget_output_aware_v1.sh

METHODS="dynamic_global_activation_fixed_t30,${selected_method}"

main_root="${RUN_ROOT}/10_budget_v41_stage_lift_32p/03_frozen_full/seed_3_exploratory"
main_log_dir="${LOG_ROOT}/10_budget_v41_stage_lift_32p/03_frozen_full/seed_3_exploratory"
mkdir -p "${main_root}" "${main_log_dir}"
cat > "${main_root}/status.json" <<'EOF'
{
  "status": "running",
  "mode": "exploratory_full_seed3",
  "datasets": ["gsm8k", "math500"],
  "note": "Seed42 confirmation did not pass; this run is best-seed exploratory full, not frozen multi-seed final evidence."
}
EOF

CONFIG="${CONFIG}" RUN_ROOT="${main_root}" SOURCE_ROOT="${SOURCE_ROOT}" LOG_DIR="${main_log_dir}" \
  STAGE_SEED=3 STAGE_FINAL_SEEDS=3 DATASETS_OVERRIDE="gsm8k math500" STAGE_FINAL_EVAL_LIMIT="${STAGE_FINAL_EVAL_LIMIT:--1}" STAGE_FINAL_METHODS="${METHODS}" \
  bash scripts/run_t30_math_safe_priority_suite_8gpu.sh

cat > "${main_root}/status.json" <<'EOF'
{
  "status": "completed",
  "mode": "exploratory_full_seed3",
  "datasets": ["gsm8k", "math500"],
  "note": "Seed42 confirmation did not pass; this run is best-seed exploratory full, not frozen multi-seed final evidence."
}
EOF

priority_root="${RUN_ROOT}/10_budget_v41_stage_lift_32p/03_frozen_full/seed_3_exploratory_priority_suite"
priority_log_dir="${LOG_ROOT}/10_budget_v41_stage_lift_32p/03_frozen_full/seed_3_exploratory_priority_suite"
mkdir -p "${priority_root}" "${priority_log_dir}"
cat > "${priority_root}/status.json" <<'EOF'
{
  "status": "running",
  "mode": "exploratory_priority_suite_seed3",
  "datasets": ["aime2024", "aime2025", "amc2023", "gpqa_diamond", "arc_challenge"],
  "note": "Priority-suite exploratory run for the best seed3 v4.1 candidate."
}
EOF

CONFIG="${CONFIG}" RUN_ROOT="${priority_root}" SOURCE_ROOT="${SOURCE_ROOT}" LOG_DIR="${priority_log_dir}" \
  STAGE_SEED=3 STAGE_FINAL_SEEDS=3 DATASETS_OVERRIDE="aime2024 aime2025 amc2023 gpqa_diamond arc_challenge" STAGE_FINAL_EVAL_LIMIT="${STAGE_PRIORITY_EVAL_LIMIT:--1}" STAGE_FINAL_METHODS="${METHODS}" \
  bash scripts/run_t30_math_safe_priority_suite_8gpu.sh

cat > "${priority_root}/status.json" <<'EOF'
{
  "status": "completed",
  "mode": "exploratory_priority_suite_seed3",
  "datasets": ["aime2024", "aime2025", "amc2023", "gpqa_diamond", "arc_challenge"],
  "note": "Priority-suite exploratory run for the best seed3 v4.1 candidate."
}
EOF
