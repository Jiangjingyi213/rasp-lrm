#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-python3}"
CONFIG_DIR="configs/generated_rasp_zero_online_conservative"

"${PYTHON}" scripts/33_prepare_rasp_zero_online_conservative_configs.py

while IFS= read -r config; do
  [[ -n "${config}" ]] || continue
  echo "START ${config}"
  "${PYTHON}" -m src.main_eval_rasp_zero_runtime --config "${config}"
  echo "DONE ${config}"
done < <("${PYTHON}" - <<'PY'
import json
from pathlib import Path
manifest = json.load(open(Path("configs/generated_rasp_zero_online_conservative") / "manifest.json"))
for row in manifest:
    print(row["config"])
PY
)

"${PYTHON}" scripts/34_summarize_rasp_zero_online_conservative.py
