#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"
PYTHON="${PYTHON:-/home/cike/jjy/envs/rasp_qwen3/bin/python}"
GPU_IDS="${GPU_IDS:-0,1,2,3}"
IFS=',' read -r -a GPUS <<< "${GPU_IDS}"
export ACTION_RISK_GPU_COUNT="${#GPUS[@]}"
CONFIG_DIR="configs/generated_action_risk_pilot"
LOG_DIR="${LOG_DIR:-logs/07_stage_aware/06_action_risk_pilot}"
mkdir -p "${LOG_DIR}"
"${PYTHON}" scripts/68_prepare_action_risk_pilot_configs.py

for queue in "${!GPUS[@]}"; do
  gpu="${GPUS[$queue]}"
  nohup bash -lc "
    set -euo pipefail
    cd '${ROOT}'
    export CUDA_VISIBLE_DEVICES='${gpu}'
    export TOKENIZERS_PARALLELISM=false
    while IFS= read -r config; do
      [[ -n \"\${config}\" ]] || continue
      run_dir=\"\$(${PYTHON} -c 'import sys; from src.utils.io import read_yaml; print(read_yaml(sys.argv[1])[\"paths\"][\"run_dir\"])' \"\${config}\")\"
      validation=\"\${run_dir}/07_action_window_bank_validation.json\"
      if [[ -f \"\${validation}\" ]] && ${PYTHON} -c 'import sys; from src.utils.io import read_json, read_yaml; v=read_json(sys.argv[1]); c=read_yaml(sys.argv[2]); b=c[\"aligned_window_bank\"]; ok=v.get(\"status\")==\"ok\" and v.get(\"ratios\")==b[\"ratios\"] and v.get(\"configured_max_boundaries_per_example\")==b[\"max_boundaries_per_example\"] and v.get(\"boundary_sampling\")==b[\"boundary_sampling\"]; raise SystemExit(0 if ok else 1)' \"\${validation}\" \"\${config}\"; then
        echo \"SKIP validated shard \${run_dir}\"
        continue
      fi
      echo \"START \${config}\"
      trajectories=\"\${run_dir}/01_trajectories.jsonl\"
      if [[ -s \"\${trajectories}\" ]] && ${PYTHON} -c 'import sys; from src.utils.io import read_jsonl, read_yaml; c=read_yaml(sys.argv[1]); raise SystemExit(0 if len(read_jsonl(sys.argv[2]))==int(c[\"data\"][\"limit\"]) else 1)' \"\${config}\" \"\${trajectories}\"; then
        echo \"REUSE complete dense trajectories \${trajectories}\"
      else
        ${PYTHON} -m src.main_generate --config \"\${config}\"
      fi
      ${PYTHON} -m src.main_collect_aligned_window_bank --config \"\${config}\"
      ${PYTHON} -m src.main_validate_aligned_window_bank --config \"\${config}\"
      echo \"DONE \${config}\"
    done < '${CONFIG_DIR}/gpu${queue}.list'
  " > "${LOG_DIR}/gpu${gpu}.log" 2>&1 &
  echo "worker pid=$! queue=${queue} gpu=${gpu} log=${LOG_DIR}/gpu${gpu}.log"
done
