#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-python3}"
STAGE_PROBE_ROOT="${STAGE_PROBE_ROOT:-runs/07_stage_aware/03_s1_three_stage_probe}"
VARIANTS="${STAGE_PROBE_VARIANTS:-position_only uncertainty_only hidden_pca_linear hidden_pca_nonlinear hidden_uncertainty}"

for seed in ${STAGE_PROBE_SEEDS:-1 2 3}; do
  for variant in ${VARIANTS}; do
    run_dir="${STAGE_PROBE_ROOT}/seed_${seed}/${variant}"
    "${PYTHON}" -m src.main_eval_rasp_stage_probe \
      --dataset "${STAGE_PROBE_ROOT}/data/01_stage_dataset.jsonl" \
      --hidden-states "${STAGE_PROBE_ROOT}/data/01_stage_hidden_states.pt" \
      --manifest "${STAGE_PROBE_ROOT}/data/split_seed_${seed}.json" \
      --checkpoint "${run_dir}/stage_probe.pt" \
      --output "${run_dir}/eval.json"
  done
done

"${PYTHON}" scripts/60_summarize_rasp_stage_probe.py --root "${STAGE_PROBE_ROOT}"
