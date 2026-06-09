#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-python3}"
OUTPUT_ROOT="${OUTPUT_ROOT:-runs/rasp_phase_b2}"
VARIANTS="${PHASE_B2_VARIANTS:-hidden_multitask hidden_flip_only uncertainty_multitask}"

for seed in ${PHASE_B2_SEEDS:-1 2 3}; do
  for variant in ${VARIANTS}; do
    run_dir="${OUTPUT_ROOT}/seed_${seed}/${variant}"
    "${PYTHON}" -m src.main_eval_rasp_phase_b2 \
      --dataset "${OUTPUT_ROOT}/data/01_phase_b2_dataset.jsonl" \
      --hidden-states "${OUTPUT_ROOT}/data/01_phase_b2_hidden_states.pt" \
      --manifest "${OUTPUT_ROOT}/data/split_seed_${seed}.json" \
      --checkpoint "${run_dir}/policy.pt" \
      --output "${run_dir}/eval.json"
  done
done
"${PYTHON}" scripts/48_summarize_rasp_phase_b2.py --root "${OUTPUT_ROOT}"
