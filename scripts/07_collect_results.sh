#!/usr/bin/env bash
set -euo pipefail

python3 -m src.main_collect_results --output "${1:-runs/01_motivation/motivation_qwen3_combined_l100_no_l20.json}" --configs "${@:2}"
