#!/usr/bin/env bash
set -euo pipefail

python3 -m src.main_train_probe --config "${1:-configs/exp_minimal_gsm8k.yaml}"
