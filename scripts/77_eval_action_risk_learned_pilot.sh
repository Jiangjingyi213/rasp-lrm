#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"
PYTHON="${PYTHON:-/home/cike/jjy/envs/rasp_qwen3/bin/python}"
GPU_IDS="${GPU_IDS:-0,1,2,3,4,5,6,7}"
IFS=',' read -r -a GPUS <<< "${GPU_IDS}"
export ACTION_RISK_LEARNED_GPU_COUNT="${#GPUS[@]}"
CONFIG_DIR="configs/generated_action_risk_learned_pilot"
LOG_DIR="${LOG_DIR:-logs/07_stage_aware/08_action_risk_learned_single_window_pilot}"
mkdir -p "${LOG_DIR}"
"${PYTHON}" scripts/76_prepare_action_risk_learned_configs.py

for queue in "${!GPUS[@]}"; do
  gpu="${GPUS[$queue]}"
  nohup bash -lc "
    set -euo pipefail
    cd '${ROOT}'
    export CUDA_VISIBLE_DEVICES='${gpu}'
    export TOKENIZERS_PARALLELISM=false
    while IFS= read -r config; do
      [[ -n \"\${config}\" ]] || continue
      summary=\"\$(${PYTHON} -c 'import sys; from src.utils.io import read_yaml; print(read_yaml(sys.argv[1])[\"paths\"][\"runtime_summary\"])' \"\${config}\")\"
      if [[ -f \"\${summary}\" ]] && ${PYTHON} -c 'import sys; from src.utils.io import read_json, read_yaml; s=read_json(sys.argv[1]); c=read_yaml(sys.argv[2]); r=c[\"runtime_rasp\"]; ok=s.get(\"examples\")==c[\"data\"][\"limit\"] and s.get(\"controller\")==r[\"controller\"] and s.get(\"policy_variant\")==r.get(\"policy_variant\") and s.get(\"operating_point\")==r.get(\"operating_point\"); raise SystemExit(0 if ok else 1)' \"\${summary}\" \"\${config}\"; then
        echo \"SKIP completed \${config}\"
        continue
      fi
      echo \"START \${config}\"
      ${PYTHON} -m src.main_eval_rasp_zero_runtime --config \"\${config}\"
      echo \"DONE \${config}\"
    done < '${CONFIG_DIR}/gpu${queue}.list'
  " > "${LOG_DIR}/gpu${gpu}.log" 2>&1 &
  echo "worker pid=$! queue=${queue} gpu=${gpu} log=${LOG_DIR}/gpu${gpu}.log"
done
