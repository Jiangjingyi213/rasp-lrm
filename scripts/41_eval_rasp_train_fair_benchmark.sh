#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-python3}"
SOURCE_ROOT="${SOURCE_ROOT:-runs/04_rasp_train/01_legacy/rasp_train_v2_1}"
OUTPUT_ROOT="${OUTPUT_ROOT:-runs/04_rasp_train/02_fair_benchmark/rasp_train_fair_benchmark}"
VARIANTS="${FAIR_VARIANTS:-ratio_only_linear position_ratio_linear uncertainty_ratio_linear hidden_ratio_linear hidden_ratio_nonlinear}"
LABELS="${FAIR_LABELS:-flipped unsafe}"

for seed in ${FAIR_SEEDS:-1 2 3}; do
  for label in ${LABELS}; do
    for variant in ${VARIANTS}; do
      run_dir="${OUTPUT_ROOT}/seed_${seed}/${label}/${variant}"
      "${PYTHON}" -m src.main_eval_rasp_train_fair_benchmark \
        --b15-dataset "${SOURCE_ROOT}/b15/11_rasp_train_policy_dataset.jsonl" \
        --b15-hidden-states "${SOURCE_ROOT}/b15/11_rasp_train_policy_hidden_states.pt" \
        --b20-dataset "${SOURCE_ROOT}/b20/11_rasp_train_policy_dataset.jsonl" \
        --b20-hidden-states "${SOURCE_ROOT}/b20/11_rasp_train_policy_hidden_states.pt" \
        --manifest "${OUTPUT_ROOT}/split_manifests/seed_${seed}.json" \
        --checkpoint "${run_dir}/policy.pt" \
        --output "${run_dir}/eval.json"
    done
  done
done

"${PYTHON}" scripts/42_summarize_rasp_train_fair_benchmark.py --root "${OUTPUT_ROOT}"
