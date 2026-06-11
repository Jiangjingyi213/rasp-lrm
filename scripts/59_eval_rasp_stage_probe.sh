#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-python3}"
OUTPUT_ROOT="${OUTPUT_ROOT:-runs/07_stage_aware/02_s1_operational_stage_probe}"
VARIANTS="${STAGE_PROBE_VARIANTS:-position_only uncertainty_only hidden_pca_linear hidden_pca_nonlinear hidden_uncertainty}"

for seed in ${STAGE_PROBE_SEEDS:-1 2 3}; do
  for variant in ${VARIANTS}; do
    run_dir="${OUTPUT_ROOT}/seed_${seed}/${variant}"
    "${PYTHON}" -m src.main_eval_rasp_stage_probe \
      --dataset "${OUTPUT_ROOT}/data/01_stage_dataset.jsonl" \
      --hidden-states "${OUTPUT_ROOT}/data/01_stage_hidden_states.pt" \
      --manifest "${OUTPUT_ROOT}/data/split_seed_${seed}.json" \
      --checkpoint "${run_dir}/stage_probe.pt" \
      --output "${run_dir}/eval.json"
  done
done

"${PYTHON}" scripts/60_summarize_rasp_stage_probe.py --root "${OUTPUT_ROOT}"
