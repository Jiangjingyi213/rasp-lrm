#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-python3}"
OUTPUT_ROOT="${OUTPUT_ROOT:-runs/rasp_train_v1}"
CONFIG_DIR="${CONFIG_DIR:-/tmp/rasp_train_v1_online_smoke_configs}"
mkdir -p "${CONFIG_DIR}"

write_config() {
  local dataset="$1"
  local tag="$2"
  local budget="$3"
  local limit="$4"
  local controller="$5"
  local policy="${OUTPUT_ROOT}/${tag}/rasp_train_policy.pt"
  local run_dir
  local config
  local controller_config
  if [[ "${controller}" == "fixed" ]]; then
    run_dir="runs/eval_rasp_train_dense_${dataset}_smoke"
    config="${CONFIG_DIR}/dense_${dataset}.yaml"
    controller_config=$'  controller: fixed\n  fixed_ratio: 0.00'
  else
    run_dir="runs/eval_rasp_train_${tag}_${dataset}_smoke"
    config="${CONFIG_DIR}/${tag}_${dataset}.yaml"
    controller_config="  controller: rasp_train_policy
  policy_checkpoint: ${policy}
  target_average_ratio: ${budget}"
  fi
  local dataset_extra=""
  if [[ "${dataset}" == "math500" ]]; then
    dataset_extra=$'  name_or_path: HuggingFaceH4/MATH-500\n'
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
  limit: ${limit}

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

for dataset in gsm8k math500; do
  config=$(write_config "${dataset}" "dense" "0.00" "${RASP_TRAIN_ONLINE_LIMIT:-20}" "fixed")
  "${PYTHON}" -m src.main_eval_rasp_zero_runtime --config "${config}"
done

for tag_budget in "b15 0.15" "b20 0.20"; do
  read -r tag budget <<< "${tag_budget}"
  for dataset in gsm8k math500; do
    config=$(write_config "${dataset}" "${tag}" "${budget}" "${RASP_TRAIN_ONLINE_LIMIT:-20}" "rasp_train_policy")
    "${PYTHON}" -m src.main_eval_rasp_zero_runtime --config "${config}"
    "${PYTHON}" -m src.main_compare_rasp_train_online \
      --dense "runs/eval_rasp_train_dense_${dataset}_smoke/01_trajectories.jsonl" \
      --policy "runs/eval_rasp_train_${tag}_${dataset}_smoke/01_trajectories.jsonl" \
      --output "runs/eval_rasp_train_${tag}_${dataset}_smoke/14_paired_dense_comparison.json"
  done
done
