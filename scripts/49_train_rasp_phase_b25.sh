#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-python3}"
DATA_ROOT="${DATA_ROOT:-runs/05_phase_b/02_phase_b2/rasp_phase_b2_v3}"
OUTPUT_ROOT="${OUTPUT_ROOT:-runs/05_phase_b/03_phase_b25/rasp_phase_b25}"
VARIANTS="${PHASE_B25_VARIANTS:-uncertainty_nonlinear hidden_pca_linear hidden_pca_nonlinear uncertainty_hidden_residual}"

for seed in ${PHASE_B25_SEEDS:-1 2 3}; do
  for variant in ${VARIANTS}; do
    run_dir="${OUTPUT_ROOT}/seed_${seed}/${variant}"
    mkdir -p "${run_dir}"
    "${PYTHON}" -m src.main_train_rasp_phase_b25 \
      --dataset "${DATA_ROOT}/data/01_phase_b2_dataset.jsonl" \
      --hidden-states "${DATA_ROOT}/data/01_phase_b2_hidden_states.pt" \
      --manifest "${DATA_ROOT}/data/split_seed_${seed}.json" \
      --variant "${variant}" \
      --output "${run_dir}/policy.pt" \
      --metrics-output "${run_dir}/train_metrics.json" \
      --epochs "${PHASE_B25_EPOCHS:-80}" \
      --lr "${PHASE_B25_LR:-1e-3}" \
      --weight-decay "${PHASE_B25_WEIGHT_DECAY:-0.05}" \
      --pca-dim "${PHASE_B25_PCA_DIM:-32}" \
      --model-dim "${PHASE_B25_MODEL_DIM:-32}" \
      --seed "${seed}"
  done
done
