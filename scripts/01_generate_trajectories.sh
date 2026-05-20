#!/usr/bin/env bash
set -euo pipefail

python3 -m src.main_generate --config "${1:-configs/exp_minimal_gsm8k.yaml}"
