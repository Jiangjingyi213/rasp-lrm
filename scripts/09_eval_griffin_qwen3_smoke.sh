#!/usr/bin/env bash
set -euo pipefail

python3 -m src.main_generate --config "${1:-configs/eval_griffin_qwen3_gsm8k_smoke.yaml}"
