#!/usr/bin/env bash
set -euo pipefail

python3 -m src.main_segment --config "${1:-configs/exp_minimal_gsm8k.yaml}"
