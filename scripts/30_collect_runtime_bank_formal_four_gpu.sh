#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

PYTHON="${PYTHON:-/home/cike/jjy/envs/rasp_qwen3/bin/python}"
LOG_DIR="${LOG_DIR:-logs/rasp_zero_runtime_bank_formal}"
CONFIG_DIR="configs/generated_runtime_bank_formal"
mkdir -p "${LOG_DIR}"

"${PYTHON}" scripts/29_prepare_runtime_bank_formal_configs.py

launch_worker() {
  local gpu="$1"
  nohup bash -lc "
    set -euo pipefail
    cd '${ROOT}'
    export CUDA_VISIBLE_DEVICES='${gpu}'
    export TOKENIZERS_PARALLELISM=false
    export PYTHON='${PYTHON}'
    while IFS= read -r config; do
      [[ -n \"\${config}\" ]] || continue
      run_dir=\"\$(\"\${PYTHON}\" -c 'import sys; from src.utils.io import read_yaml; print(read_yaml(sys.argv[1])[\"paths\"][\"run_dir\"])' \"\${config}\")\"
      if [[ -f \"\${run_dir}/07_runtime_bank_validation.json\" ]] && grep -q '\"status\": \"ok\"' \"\${run_dir}/07_runtime_bank_validation.json\"; then
        echo \"SKIP validated \${run_dir}\"
        continue
      fi
      rm -rf \"\${run_dir}\"
      echo \"START \${config}\"
      bash scripts/23_collect_runtime_counterfactuals.sh \"\${config}\"
      echo \"DONE \${config}\"
    done < '${CONFIG_DIR}/gpu${gpu}.list'
  " > "${LOG_DIR}/gpu${gpu}.log" 2>&1 &
  echo "worker pid=$! physical_gpu=${gpu} log=${LOG_DIR}/gpu${gpu}.log"
}

for gpu in 0 1 2 3; do
  launch_worker "${gpu}"
done

echo
echo "Monitor:"
echo "  watch -n 10 nvidia-smi"
echo "  tail -f ${LOG_DIR}/gpu0.log"
echo "  grep -R 'DONE\\|Traceback\\|CUDA out of memory' ${LOG_DIR}"

