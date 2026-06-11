#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-python3}"
OUTPUT_DIR="${OUTPUT_DIR:-runs/03_rasp_zero/01_offline/rasp_zero_offline_v2}"

RUN_DIRS=(
  runs/01_motivation/formal_qwen3_gsm8k_full_s0
  runs/01_motivation/formal_qwen3_gsm8k_full_s1
  runs/01_motivation/formal_qwen3_math500_full_s0
  runs/01_motivation/formal_qwen3_math500_full_s1
)

INPUTS=(
  runs/01_motivation/formal_qwen3_gsm8k_full_s0/05_probe_dataset.jsonl
  runs/01_motivation/formal_qwen3_gsm8k_full_s1/05_probe_dataset.jsonl
  runs/01_motivation/formal_qwen3_math500_full_s0/05_probe_dataset.jsonl
  runs/01_motivation/formal_qwen3_math500_full_s1/05_probe_dataset.jsonl
)

mkdir -p "${OUTPUT_DIR}"

"${PYTHON}" -m src.main_validate_rasp_inputs \
  --run-dirs "${RUN_DIRS[@]}"

"${PYTHON}" -m src.main_generate_oof_probe_scores \
  --run-dirs "${RUN_DIRS[@]}" \
  --output "${OUTPUT_DIR}/hidden_step_oof_scores.jsonl" \
  --summary-output "${OUTPUT_DIR}/hidden_step_oof_summary.json" \
  --feature-set hidden \
  --folds 5 \
  --epochs 20 \
  --batch-size 128

"${PYTHON}" -m src.main_generate_oof_probe_scores \
  --run-dirs "${RUN_DIRS[@]}" \
  --output "${OUTPUT_DIR}/action_conditioned_oof_scores.jsonl" \
  --summary-output "${OUTPUT_DIR}/action_conditioned_oof_summary.json" \
  --feature-set action_hidden \
  --folds 5 \
  --epochs 20 \
  --batch-size 128

"${PYTHON}" -m src.main_generate_oof_probe_scores \
  --run-dirs "${RUN_DIRS[@]}" \
  --output "${OUTPUT_DIR}/action_stage_conditioned_oof_scores.jsonl" \
  --summary-output "${OUTPUT_DIR}/action_stage_conditioned_oof_summary.json" \
  --feature-set action_hidden_stage \
  --folds 5 \
  --epochs 20 \
  --batch-size 128

"${PYTHON}" -m src.main_rasp_zero_offline_v2 \
  --inputs "${INPUTS[@]}" \
  --hidden-step-scores "${OUTPUT_DIR}/hidden_step_oof_scores.jsonl" \
  --action-scores "${OUTPUT_DIR}/action_conditioned_oof_scores.jsonl" \
  --action-stage-scores "${OUTPUT_DIR}/action_stage_conditioned_oof_scores.jsonl" \
  --output-dir "${OUTPUT_DIR}" \
  --strength-budgets 0.1 0.2 0.3

if "${PYTHON}" -c "import matplotlib, pandas, seaborn" >/dev/null 2>&1; then
  "${PYTHON}" scripts/22_plot_rasp_zero_offline_v2.py \
    --input "${OUTPUT_DIR}/rasp_zero_offline_v2_summary.csv" \
    --output-dir "${OUTPUT_DIR}/figures"
else
  echo "Skipping v2 policy plot: matplotlib, pandas, or seaborn is unavailable."
fi

echo "RASP-Zero offline v2 evaluation complete: ${OUTPUT_DIR}"
