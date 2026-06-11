#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-python3}"
COMBINED="${COMBINED:-runs/01_motivation/formal_qwen3_gsm8k_math500_combined.json}"
OUTPUT_DIR="${OUTPUT_DIR:-runs/01_motivation/motivation_analysis}"

"$PYTHON" -m src.main_motivation_analysis \
  --combined "$COMBINED" \
  --output-dir "$OUTPUT_DIR" \
  --run-dirs \
    runs/01_motivation/formal_qwen3_gsm8k_full_s0 \
    runs/01_motivation/formal_qwen3_gsm8k_full_s1 \
    runs/01_motivation/formal_qwen3_math500_full_s0 \
    runs/01_motivation/formal_qwen3_math500_full_s1
