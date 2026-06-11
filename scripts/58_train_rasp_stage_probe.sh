#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-python3}"
STAGE_PROBE_ROOT="${STAGE_PROBE_ROOT:-runs/07_stage_aware/03_s1_three_stage_probe}"
VARIANTS="${STAGE_PROBE_VARIANTS:-position_only uncertainty_only hidden_pca_linear hidden_pca_nonlinear hidden_uncertainty}"
AUDIT_LABELS="${STAGE_AUDIT_LABELS:-configs/stage_audits/s1_three_stage_v3_labels.csv}"

"${PYTHON}" scripts/62_apply_rasp_stage_audit_labels.py \
  --audit "${STAGE_PROBE_ROOT}/data/02_stage_manual_audit.csv" \
  --labels "${AUDIT_LABELS}"
"${PYTHON}" scripts/61_validate_rasp_stage_audit.py \
  --audit "${STAGE_PROBE_ROOT}/data/02_stage_manual_audit.csv"

for seed in ${STAGE_PROBE_SEEDS:-1 2 3}; do
  for variant in ${VARIANTS}; do
    run_dir="${STAGE_PROBE_ROOT}/seed_${seed}/${variant}"
    mkdir -p "${run_dir}"
    "${PYTHON}" -m src.main_train_rasp_stage_probe \
      --dataset "${STAGE_PROBE_ROOT}/data/01_stage_dataset.jsonl" \
      --hidden-states "${STAGE_PROBE_ROOT}/data/01_stage_hidden_states.pt" \
      --manifest "${STAGE_PROBE_ROOT}/data/split_seed_${seed}.json" \
      --variant "${variant}" \
      --output "${run_dir}/stage_probe.pt" \
      --metrics-output "${run_dir}/train_metrics.json" \
      --pca-dim "${STAGE_PROBE_PCA_DIM:-32}" \
      --model-dim "${STAGE_PROBE_MODEL_DIM:-64}" \
      --epochs "${STAGE_PROBE_EPOCHS:-50}" \
      --batch-size "${STAGE_PROBE_BATCH_SIZE:-128}" \
      --lr "${STAGE_PROBE_LR:-5e-4}" \
      --seed "${seed}"
  done
done
