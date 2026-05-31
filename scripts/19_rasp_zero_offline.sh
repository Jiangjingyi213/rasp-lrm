#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-python3}"
OUTPUT_DIR="${OUTPUT_DIR:-runs/rasp_zero_offline}"
OOF_SCORES="${OOF_SCORES:-${OUTPUT_DIR}/hidden_probe_oof_scores.jsonl}"

RUN_DIRS=(
  runs/formal_qwen3_gsm8k_full_s0
  runs/formal_qwen3_gsm8k_full_s1
  runs/formal_qwen3_math500_full_s0
  runs/formal_qwen3_math500_full_s1
)

INPUTS=(
  runs/formal_qwen3_gsm8k_full_s0/05_probe_dataset.jsonl
  runs/formal_qwen3_gsm8k_full_s1/05_probe_dataset.jsonl
  runs/formal_qwen3_math500_full_s0/05_probe_dataset.jsonl
  runs/formal_qwen3_math500_full_s1/05_probe_dataset.jsonl
)

mkdir -p "${OUTPUT_DIR}"

"${PYTHON}" -m src.main_validate_rasp_inputs \
  --run-dirs "${RUN_DIRS[@]}"

"${PYTHON}" -m src.main_generate_oof_probe_scores \
  --run-dirs "${RUN_DIRS[@]}" \
  --output "${OOF_SCORES}" \
  --summary-output "${OUTPUT_DIR}/hidden_probe_oof_summary.json" \
  --feature-set hidden \
  --folds 5 \
  --epochs 20 \
  --batch-size 128

"${PYTHON}" -m src.main_rasp_zero_offline \
  --inputs "${INPUTS[@]}" \
  --probe-scores "${OOF_SCORES}" \
  --output-dir "${OUTPUT_DIR}" \
  --module mlp_block \
  --budgets 0.2 0.4 0.6 \
  --ratios 0.2 0.4 0.6

if "${PYTHON}" -c "import matplotlib, pandas, seaborn" >/dev/null 2>&1; then
  "${PYTHON}" scripts/20_plot_rasp_zero_offline.py \
    --input "${OUTPUT_DIR}/rasp_zero_offline_summary.csv" \
    --output-dir "${OUTPUT_DIR}/figures"
else
  echo "Skipping policy plot: matplotlib, pandas, or seaborn is unavailable."
fi

echo "RASP-Zero offline evaluation complete: ${OUTPUT_DIR}"
