#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-python3}"
OUTPUT_ROOT="${OUTPUT_ROOT:-runs/rasp_phase_b2_v3}"
VARIANTS="${PHASE_B2_VARIANTS:-hidden_multitask hidden_flip_only hidden_flip_linear uncertainty_multitask uncertainty_flip_only uncertainty_flip_linear position_flip_only position_flip_linear ratio_only_flip_only ratio_only_flip_linear}"

for seed in ${PHASE_B2_SEEDS:-1 2 3}; do
  for variant in ${VARIANTS}; do
    run_dir="${OUTPUT_ROOT}/seed_${seed}/${variant}"
    mkdir -p "${run_dir}"
    "${PYTHON}" -m src.main_train_rasp_phase_b2 \
      --dataset "${OUTPUT_ROOT}/data/01_phase_b2_dataset.jsonl" \
      --hidden-states "${OUTPUT_ROOT}/data/01_phase_b2_hidden_states.pt" \
      --manifest "${OUTPUT_ROOT}/data/split_seed_${seed}.json" \
      --variant "${variant}" \
      --output "${run_dir}/policy.pt" \
      --metrics-output "${run_dir}/train_metrics.json" \
      --epochs "${PHASE_B2_EPOCHS:-40}" \
      --batch-size "${PHASE_B2_BATCH_SIZE:-128}" \
      --lr "${PHASE_B2_LR:-5e-4}" \
      --divergence-weight "${PHASE_B2_DIVERGENCE_WEIGHT:-0.5}" \
      --hidden-drift-weight "${PHASE_B2_HIDDEN_DRIFT_WEIGHT:-0.5}" \
      --seed "${seed}"
  done
done
