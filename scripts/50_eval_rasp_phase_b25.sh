#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-python3}"
DATA_ROOT="${DATA_ROOT:-runs/rasp_phase_b2_v3}"
OUTPUT_ROOT="${OUTPUT_ROOT:-runs/rasp_phase_b25}"
VARIANTS="${PHASE_B25_VARIANTS:-uncertainty_nonlinear hidden_pca_linear hidden_pca_nonlinear uncertainty_hidden_residual}"

for seed in ${PHASE_B25_SEEDS:-1 2 3}; do
  for variant in ${VARIANTS}; do
    run_dir="${OUTPUT_ROOT}/seed_${seed}/${variant}"
    "${PYTHON}" -m src.main_eval_rasp_phase_b25 \
      --dataset "${DATA_ROOT}/data/01_phase_b2_dataset.jsonl" \
      --hidden-states "${DATA_ROOT}/data/01_phase_b2_hidden_states.pt" \
      --manifest "${DATA_ROOT}/data/split_seed_${seed}.json" \
      --checkpoint "${run_dir}/policy.pt" \
      --output "${run_dir}/eval.json"
  done
done
"${PYTHON}" scripts/51_summarize_rasp_phase_b25.py --root "${OUTPUT_ROOT}"
