#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-python3}"
SOURCE_ROOT="${SOURCE_ROOT:-runs/04_rasp_train/01_legacy/rasp_train_v2_1}"
OUTPUT_ROOT="${OUTPUT_ROOT:-runs/04_rasp_train/02_fair_benchmark/rasp_train_fair_benchmark}"
VARIANTS="${FAIR_VARIANTS:-ratio_only_linear position_ratio_linear uncertainty_ratio_linear hidden_ratio_linear hidden_ratio_nonlinear}"
LABELS="${FAIR_LABELS:-flipped unsafe}"

for seed in ${FAIR_SEEDS:-1 2 3}; do
  manifest="${OUTPUT_ROOT}/split_manifests/seed_${seed}.json"
  [[ -f "${manifest}" ]] || { echo "Missing ${manifest}; run script 39 first" >&2; exit 1; }
  for label in ${LABELS}; do
    for variant in ${VARIANTS}; do
      run_dir="${OUTPUT_ROOT}/seed_${seed}/${label}/${variant}"
      mkdir -p "${run_dir}"
      "${PYTHON}" -m src.main_train_rasp_train_fair_benchmark \
        --dataset "${SOURCE_ROOT}/b15/11_rasp_train_policy_dataset.jsonl" \
        --hidden-states "${SOURCE_ROOT}/b15/11_rasp_train_policy_hidden_states.pt" \
        --manifest "${manifest}" \
        --variant "${variant}" \
        --label-type "${label}" \
        --output "${run_dir}/policy.pt" \
        --metrics-output "${run_dir}/train_metrics.json" \
        --epochs "${FAIR_EPOCHS:-40}" \
        --batch-size "${FAIR_BATCH_SIZE:-128}" \
        --lr "${FAIR_LR:-5e-4}" \
        --seed "${seed}"
    done
  done
done
