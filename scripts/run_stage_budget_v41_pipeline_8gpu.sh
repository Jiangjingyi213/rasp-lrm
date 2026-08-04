#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

PYTHON_BIN="${PYTHON:-/home/cike/jjy/envs/rasp_qwen3_eval/bin/python}"
RUN_ROOT="${RUN_ROOT:-runs/10_stage_budget_output_aware_v1}"

RUN_ROOT="${RUN_ROOT}" bash scripts/run_stage_budget_v41_smoke_8gpu.sh

smoke_passed="$(RUN_ROOT="${RUN_ROOT}" "${PYTHON_BIN}" - <<'PY'
import json
import os
from pathlib import Path
path = Path(os.environ["RUN_ROOT"]) / "10_budget_v41_stage_lift_32p" / "00_smoke_32p" / "budget_v41_smoke_selection.json"
print("1" if path.exists() and json.load(open(path, "r", encoding="utf-8")).get("phase_passed") else "0")
PY
)"
if [[ "${smoke_passed}" != "1" ]]; then
  echo "Budget v4.1 smoke did not reach 32.0%-34.0% actual pruning; stopping before dev."
  exit 0
fi

RUN_ROOT="${RUN_ROOT}" bash scripts/run_stage_budget_v41_dev_8gpu.sh

dev_passed="$(RUN_ROOT="${RUN_ROOT}" "${PYTHON_BIN}" - <<'PY'
import json
import os
from pathlib import Path
path = Path(os.environ["RUN_ROOT"]) / "10_budget_v41_stage_lift_32p" / "01_dev_seed3" / "budget_v41_dev_selection.json"
print("1" if path.exists() and json.load(open(path, "r", encoding="utf-8")).get("phase_passed") else "0")
PY
)"
if [[ "${dev_passed}" != "1" ]]; then
  echo "Budget v4.1 dev did not pass; stopping before seed42 confirmation."
  exit 0
fi

RUN_ROOT="${RUN_ROOT}" bash scripts/run_stage_budget_v41_seed42_8gpu.sh

seed42_passed="$(RUN_ROOT="${RUN_ROOT}" "${PYTHON_BIN}" - <<'PY'
import json
import os
from pathlib import Path
path = Path(os.environ["RUN_ROOT"]) / "10_budget_v41_stage_lift_32p" / "02_seed42_confirmation" / "budget_v41_seed42_selection.json"
print("1" if path.exists() and json.load(open(path, "r", encoding="utf-8")).get("phase_passed") else "0")
PY
)"
if [[ "${seed42_passed}" != "1" ]]; then
  echo "Budget v4.1 seed42 confirmation did not pass; stopping before full."
  exit 0
fi

RUN_ROOT="${RUN_ROOT}" bash scripts/run_stage_budget_v41_full_8gpu.sh
