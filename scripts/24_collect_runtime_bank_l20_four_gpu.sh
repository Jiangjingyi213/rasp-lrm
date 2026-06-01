#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

PYTHON="${PYTHON:-/home/cike/jjy/envs/rasp_qwen3/bin/python}"
LOG_DIR="${LOG_DIR:-logs/rasp_zero_runtime_bank_l20}"
MARKER_DIR="${MARKER_DIR:-runs/rasp_zero_runtime_bank_l20/.markers}"
RUN_ROOT="runs/rasp_zero_runtime_bank_l20"

mkdir -p "${LOG_DIR}"
rm -rf "${RUN_ROOT}"
mkdir -p "${MARKER_DIR}"

"${PYTHON}" scripts/24_prepare_runtime_bank_l20_configs.py

launch() {
  local gpu="$1"
  local name="$2"
  local command="$3"
  nohup bash -lc "
    set -euo pipefail
    cd '${ROOT}'
    export CUDA_VISIBLE_DEVICES='${gpu}'
    export TOKENIZERS_PARALLELISM=false
    export PYTHON='${PYTHON}'
    ${command}
  " > "${LOG_DIR}/${name}.log" 2>&1 &
  echo "${name}: pid=$! physical_gpu=${gpu} log=${LOG_DIR}/${name}.log"
}

launch 0 gsm8k_s0 "
  \"\${PYTHON}\" -m unittest discover -s tests -v
  \"\${PYTHON}\" -m src.main_eval_rasp_zero_runtime --config configs/exp_rasp_zero_runtime_smoke.yaml
  bash scripts/23_collect_runtime_counterfactuals.sh configs/exp_rasp_zero_runtime_bank_gsm8k_smoke.yaml
  touch '${MARKER_DIR}/gsm8k_smoke.ok'
  bash scripts/23_collect_runtime_counterfactuals.sh configs/generated_runtime_bank_l20/gsm8k_s0.yaml
"

launch 2 gsm8k_s1 "
  until [[ -f '${MARKER_DIR}/gsm8k_smoke.ok' ]]; do sleep 30; done
  bash scripts/23_collect_runtime_counterfactuals.sh configs/generated_runtime_bank_l20/gsm8k_s1.yaml
"

launch 1 math500_s0 "
  bash scripts/23_collect_runtime_counterfactuals.sh configs/exp_rasp_zero_runtime_bank_math500_smoke.yaml
  touch '${MARKER_DIR}/math500_smoke.ok'
  bash scripts/23_collect_runtime_counterfactuals.sh configs/generated_runtime_bank_l20/math500_s0.yaml
"

launch 3 math500_s1 "
  until [[ -f '${MARKER_DIR}/math500_smoke.ok' ]]; do sleep 30; done
  bash scripts/23_collect_runtime_counterfactuals.sh configs/generated_runtime_bank_l20/math500_s1.yaml
"

echo
echo "Monitor:"
echo "  tail -f ${LOG_DIR}/gsm8k_s0.log"
echo "  tail -f ${LOG_DIR}/math500_s0.log"
echo "  watch -n 10 nvidia-smi"
