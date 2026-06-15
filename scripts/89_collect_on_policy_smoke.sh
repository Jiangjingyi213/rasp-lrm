#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"
PYTHON="${PYTHON:-/home/cike/jjy/envs/rasp_qwen3/bin/python}"
GPU_IDS="${GPU_IDS:-0,1,2,3,4,5,6,7}"
IFS=',' read -r -a GPUS <<< "${GPU_IDS}"
export ON_POLICY_GPU_COUNT="${#GPUS[@]}"
WORKFLOW_ROOT="${FULL_TRAJECTORY_ROOT:-runs/07_stage_aware/10_full_trajectory_multi_window}"
RUN_ROOT="${WORKFLOW_ROOT}/04_on_policy_smoke"
CONFIG_DIR="configs/generated_on_policy_smoke"
LOG_DIR="${LOG_DIR:-logs/07_stage_aware/10_full_trajectory_multi_window/on_policy_smoke}"
mkdir -p "${LOG_DIR}"
"${PYTHON}" scripts/88_prepare_on_policy_smoke_configs.py

pids=()
for queue in "${!GPUS[@]}"; do
  gpu="${GPUS[$queue]}"
  (
    set -euo pipefail
    export CUDA_VISIBLE_DEVICES="${gpu}"
    export TOKENIZERS_PARALLELISM=false
    "${PYTHON}" -c '
import json, sys
for item in json.load(open(sys.argv[1], encoding="utf-8")):
    if int(item["gpu_queue"]) == int(sys.argv[2]):
        print(item["config"])
' "${CONFIG_DIR}/manifest.json" "${queue}" | while IFS= read -r config; do
      [[ -n "${config}" ]] || continue
      validation="$("${PYTHON}" -c 'import sys; from src.utils.io import read_yaml; print(read_yaml(sys.argv[1])["paths"]["on_policy_validation"])' "${config}")"
      if [[ -f "${validation}" ]] && "${PYTHON}" -c '
import sys
from pathlib import Path
from src.rasp.config_fingerprint import config_fingerprint
from src.utils.io import read_json, read_yaml
v, c = read_json(sys.argv[1]), read_yaml(sys.argv[2])
expected = config_fingerprint(
    c,
    ("seed", "model", "prompt", "data", "generation", "runtime_rasp", "on_policy_bank", "stage_sensitivity"),
)
raise SystemExit(
    0
    if v.get("status") == "ok"
    and v.get("risk_label_semantics") == "harmful_flip_conditioned_on_correct_dense_control_v1"
    and int(v.get("valid_problems", 0)) >= 4
    and v.get("on_policy_config_fingerprint") == expected
    and Path(c["paths"]["on_policy_dataset"]).is_file()
    and Path(c["paths"]["on_policy_hidden_states"]).is_file()
    else 1
)
' "${validation}" "${config}"; then
        echo "SKIP validated on-policy source ${config}"
        continue
      fi
      echo "START ${config}"
      "${PYTHON}" -m src.main_collect_on_policy_bank --config "${config}"
      echo "DONE ${config}"
    done
  ) > "${LOG_DIR}/gpu${gpu}.log" 2>&1 &
  pids+=("$!")
  echo "worker pid=$! queue=${queue} gpu=${gpu} log=${LOG_DIR}/gpu${gpu}.log"
done

status=0
for pid in "${pids[@]}"; do
  wait "${pid}" || status=1
done
[[ "${status}" == 0 ]] || exit "${status}"

"${PYTHON}" -m src.main_summarize_on_policy_smoke \
  --root "${RUN_ROOT}" \
  --manifest "${CONFIG_DIR}/manifest.json"
