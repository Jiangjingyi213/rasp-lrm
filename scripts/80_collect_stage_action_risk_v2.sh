#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"
PYTHON="${PYTHON:-/home/cike/jjy/envs/rasp_qwen3/bin/python}"
GPU_IDS="${GPU_IDS:-0,1,2,3,4,5,6,7}"
IFS=',' read -r -a GPUS <<< "${GPU_IDS}"
export STAGE_ACTION_RISK_V2_GPU_COUNT="${#GPUS[@]}"
CONFIG_DIR="configs/generated_stage_action_risk_v2"
LOG_DIR="${LOG_DIR:-logs/07_stage_aware/09_stage_action_risk_v2}"
mkdir -p "${LOG_DIR}"
"${PYTHON}" scripts/79_prepare_stage_action_risk_v2_configs.py

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
      validation=\"\${run_dir}/07_stage_action_bank_validation.json\"
      if [[ -f \"\${validation}\" ]] && ${PYTHON} -c 'import sys; from src.utils.io import read_json; v=read_json(sys.argv[1]); ok=v.get(\"status\")==\"ok\" and v.get(\"configured_boundary_positions\")==[32,96,160] and v.get(\"stage_sensitivity_enabled\"); raise SystemExit(0 if ok else 1)' \"\${validation}\"; then
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
