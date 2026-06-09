#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"
PYTHON="${PYTHON:-/home/cike/jjy/envs/rasp_qwen3/bin/python}"
GPU_COUNT="${RASP_PHASE_B_GPU_COUNT:-8}"
CONFIG_DIR="configs/generated_rasp_phase_b_aligned_bank"
LOG_DIR="${LOG_DIR:-logs/rasp_phase_b_aligned_bank}"
mkdir -p "${LOG_DIR}"
"${PYTHON}" scripts/43_prepare_rasp_phase_b_aligned_bank_configs.py

for ((gpu = 0; gpu < GPU_COUNT; gpu++)); do
  nohup bash -lc "
    set -euo pipefail
    cd '${ROOT}'
    export CUDA_VISIBLE_DEVICES='${gpu}'
    export TOKENIZERS_PARALLELISM=false
    while IFS= read -r config; do
      [[ -n \"\${config}\" ]] || continue
      run_dir=\"\$(${PYTHON} -c 'import sys; from src.utils.io import read_yaml; print(read_yaml(sys.argv[1])[\"paths\"][\"run_dir\"])' \"\${config}\")\"
      validation=\"\${run_dir}/07_aligned_window_bank_validation.json\"
      if [[ -f \"\${validation}\" ]]; then
        if ${PYTHON} -c 'import sys; from src.utils.io import read_json, read_yaml; y=read_yaml(sys.argv[1]); c=y[\"aligned_window_bank\"]; v=read_json(sys.argv[2]); raise SystemExit(0 if v.get(\"status\")==\"ok\" and v.get(\"action_window_alignment\")==\"affected_next_token_decisions_v2\" and v.get(\"configured_window_tokens\")==c.get(\"window_tokens\",16) and v.get(\"configured_max_new_tokens\")==y.get(\"generation\",{}).get(\"max_new_tokens\",512) and v.get(\"configured_max_boundaries_per_example\")==c.get(\"max_boundaries_per_example\") else 1)' \"\${config}\" \"\${validation}\"; then
          echo \"SKIP validated matching config \${run_dir}\"
          continue
        fi
        echo \"RECOLLECT config changed \${run_dir}\"
      fi
      echo \"START \${config}\"
      trajectories=\"\${run_dir}/01_trajectories.jsonl\"
      if [[ -s \"\${trajectories}\" ]]; then
        echo \"REUSE dense trajectories \${trajectories}\"
      else
        ${PYTHON} -m src.main_generate --config \"\${config}\"
      fi
      ${PYTHON} -m src.main_collect_aligned_window_bank --config \"\${config}\"
      ${PYTHON} -m src.main_validate_aligned_window_bank --config \"\${config}\"
      echo \"DONE \${config}\"
    done < '${CONFIG_DIR}/gpu${gpu}.list'
  " > "${LOG_DIR}/gpu${gpu}.log" 2>&1 &
  echo "worker pid=$! gpu=${gpu} log=${LOG_DIR}/gpu${gpu}.log"
done
