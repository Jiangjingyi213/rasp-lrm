#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-python3}"
COMBINED="${COMBINED:-runs/formal_qwen3_gsm8k_math500_combined.json}"
OUTPUT_DIR="${OUTPUT_DIR:-runs/motivation_analysis}"

"$PYTHON" -m src.main_motivation_analysis \
  --combined "$COMBINED" \
  --output-dir "$OUTPUT_DIR" \
  --run-dirs \
    runs/formal_qwen3_gsm8k_full_s0 \
    runs/formal_qwen3_gsm8k_full_s1 \
    runs/formal_qwen3_math500_full_s0 \
    runs/formal_qwen3_math500_full_s1
