#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-python3}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-runs/05_phase_b/02_phase_b2/rasp_phase_b2_v3}"
OUTPUT_ROOT="${OUTPUT_ROOT:-runs/06_phase_b3_online/rasp_phase_b2_uncertainty_online_smoke}"
CONFIG_DIR="${CONFIG_DIR:-/tmp/rasp_phase_b2_uncertainty_online_smoke_configs}"
SEED="${PHASE_B2_ONLINE_SEED:-1}"
LIMIT="${PHASE_B2_ONLINE_LIMIT:-20}"
mkdir -p "${CONFIG_DIR}" "${OUTPUT_ROOT}"

write_config() {
  local dataset="$1"
  local tag="$2"
  local controller="$3"
  local budget="$4"
  local run_dir="${OUTPUT_ROOT}/${dataset}/${tag}"
  local config="${CONFIG_DIR}/${dataset}_${tag}.yaml"
  local dataset_extra=""
  local controller_config=""
  if [[ "${dataset}" == "math500" ]]; then
    dataset_extra=$'  name_or_path: HuggingFaceH4/MATH-500\n'
  fi
  if [[ "${controller}" == "fixed" ]]; then
    controller_config=$'  controller: fixed\n  fixed_ratio: 0.00'
  else
    controller_config="  controller: phase_b2_uncertainty
  policy_checkpoint: ${CHECKPOINT_ROOT}/seed_${SEED}/uncertainty_flip_only/policy.pt
  target_average_ratio: ${budget}
  policy_horizon_tokens: 192"
  fi
  cat > "${config}" <<YAML
seed: 1

model:
  name_or_path: Qwen/Qwen3-1.7B
  dtype: float32
  device_map: auto
  trust_remote_code: true
  attn_implementation: eager

prompt:
  use_chat_template: true
  enable_thinking: false
  system: You are a careful math reasoning assistant.

data:
  dataset: ${dataset}
${dataset_extra}  split: test
  limit: ${LIMIT}

generation:
  max_input_tokens: 2048
  max_new_tokens: 768
  temperature: 0.0
  top_p: 1.0

runtime_rasp:
  backend: logical_mask_v0
${controller_config}
  ratios: [0.00, 0.02, 0.05, 0.10, 0.20, 0.30, 0.40]
  window_tokens: 16
  default_max_ratio: 0.40

paths:
  run_dir: ${run_dir}
  trajectories: ${run_dir}/01_trajectories.jsonl
  runtime_summary: ${run_dir}/00_runtime_summary.json
YAML
  echo "${config}"
}

for dataset in ${PHASE_B2_ONLINE_DATASETS:-gsm8k math500}; do
  config=$(write_config "${dataset}" dense fixed 0.00)
  "${PYTHON}" -m src.main_eval_rasp_zero_runtime --config "${config}"
  for tag_budget in "b15 0.15" "b20 0.20"; do
    read -r tag budget <<< "${tag_budget}"
    config=$(write_config "${dataset}" "${tag}" phase_b2_uncertainty "${budget}")
    "${PYTHON}" -m src.main_eval_rasp_zero_runtime --config "${config}"
    "${PYTHON}" -m src.main_compare_rasp_train_online \
      --dense "${OUTPUT_ROOT}/${dataset}/dense/01_trajectories.jsonl" \
      --policy "${OUTPUT_ROOT}/${dataset}/${tag}/01_trajectories.jsonl" \
      --output "${OUTPUT_ROOT}/${dataset}/${tag}/14_paired_dense_comparison.json"
  done
done
