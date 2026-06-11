#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-python3}"
DATA_ROOT="${DATA_ROOT:-runs/rasp_phase_b2_v3}"
BASE_ROOT="${BASE_ROOT:-runs/rasp_phase_b2_v3}"
OUTPUT_ROOT="${OUTPUT_ROOT:-runs/rasp_phase_b25b}"

for seed in ${PHASE_B25B_SEEDS:-1 2 3}; do
  run_dir="${OUTPUT_ROOT}/seed_${seed}/frozen_uncertainty_residual"
  mkdir -p "${run_dir}"
  "${PYTHON}" -m src.main_train_rasp_phase_b25b \
    --dataset "${DATA_ROOT}/data/01_phase_b2_dataset.jsonl" \
    --hidden-states "${DATA_ROOT}/data/01_phase_b2_hidden_states.pt" \
    --manifest "${DATA_ROOT}/data/split_seed_${seed}.json" \
    --base-checkpoint "${BASE_ROOT}/seed_${seed}/uncertainty_flip_only/policy.pt" \
    --output "${run_dir}/policy.pt" \
    --metrics-output "${run_dir}/train_metrics.json" \
    --epochs "${PHASE_B25B_EPOCHS:-80}" \
    --lr "${PHASE_B25B_LR:-1e-3}" \
    --weight-decay "${PHASE_B25B_WEIGHT_DECAY:-0.05}" \
    --pca-dim "${PHASE_B25B_PCA_DIM:-32}" \
    --model-dim "${PHASE_B25B_MODEL_DIM:-32}" \
    --seed "${seed}"
done
