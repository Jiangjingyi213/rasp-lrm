#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-python3}"
DATA_ROOT="${DATA_ROOT:-runs/05_phase_b/02_phase_b2/rasp_phase_b2_v3}"
OUTPUT_ROOT="${OUTPUT_ROOT:-runs/05_phase_b/03_phase_b25/rasp_phase_b25b}"

for seed in ${PHASE_B25B_SEEDS:-1 2 3}; do
  run_dir="${OUTPUT_ROOT}/seed_${seed}/frozen_uncertainty_residual"
  "${PYTHON}" -m src.main_eval_rasp_phase_b25b \
    --dataset "${DATA_ROOT}/data/01_phase_b2_dataset.jsonl" \
    --hidden-states "${DATA_ROOT}/data/01_phase_b2_hidden_states.pt" \
    --manifest "${DATA_ROOT}/data/split_seed_${seed}.json" \
    --checkpoint "${run_dir}/policy.pt" \
    --output "${run_dir}/eval.json"
done
"${PYTHON}" scripts/54_summarize_rasp_phase_b25b.py --root "${OUTPUT_ROOT}"
