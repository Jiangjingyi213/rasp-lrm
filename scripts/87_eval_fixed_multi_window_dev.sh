#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"
PYTHON="${PYTHON:-/home/cike/jjy/envs/rasp_qwen3/bin/python}"
GPU_IDS="${GPU_IDS:-0,1,2,3,4,5,6,7}"
IFS=',' read -r -a GPUS <<< "${GPU_IDS}"
export MULTI_WINDOW_GPU_COUNT="${#GPUS[@]}"
WORKFLOW_ROOT="${FULL_TRAJECTORY_ROOT:-runs/07_stage_aware/10_full_trajectory_multi_window}"
RUN_ROOT="${WORKFLOW_ROOT}/03_fixed_multi_window_dev"
CONFIG_DIR="configs/generated_fixed_multi_window_dev"
LOG_DIR="${LOG_DIR:-logs/07_stage_aware/10_full_trajectory_multi_window/fixed_multi_window_dev}"
mkdir -p "${LOG_DIR}"
"${PYTHON}" scripts/86_prepare_fixed_multi_window_configs.py

pids=()
for queue in "${!GPUS[@]}"; do
  gpu="${GPUS[$queue]}"
  (
    set -euo pipefail
    export CUDA_VISIBLE_DEVICES="${gpu}"
    export TOKENIZERS_PARALLELISM=false
    while IFS= read -r config; do
      [[ -n "${config}" ]] || continue
      summary="$("${PYTHON}" -c 'import sys; from src.utils.io import read_yaml; print(read_yaml(sys.argv[1])["paths"]["runtime_summary"])' "${config}")"
      if [[ -f "${summary}" ]] && "${PYTHON}" -c '
import sys
from pathlib import Path
from src.rasp.config_fingerprint import config_fingerprint
from src.utils.io import read_json, read_yaml
s, c = read_json(sys.argv[1]), read_yaml(sys.argv[2])
r = c["runtime_rasp"]
ok = (
    s.get("examples") == c["data"]["limit"]
    and s.get("controller") == r["controller"]
    and s.get("fixed_ratio") == r.get("fixed_ratio")
    and s.get("cadence_tokens") == r.get("cadence_tokens")
    and s.get("max_windows") == r.get("max_windows")
    and s.get("runtime_config_fingerprint") == config_fingerprint(
        c, ("seed", "model", "prompt", "data", "generation", "runtime_rasp")
    )
    and Path(c["paths"]["trajectories"]).is_file()
)
raise SystemExit(0 if ok else 1)
' "${summary}" "${config}"; then
        echo "SKIP completed ${config}"
        continue
      fi
      echo "START ${config}"
      "${PYTHON}" -m src.main_eval_rasp_zero_runtime --config "${config}"
      echo "DONE ${config}"
    done < "${CONFIG_DIR}/gpu${queue}.list"
  ) > "${LOG_DIR}/gpu${gpu}.log" 2>&1 &
  pids+=("$!")
  echo "worker pid=$! queue=${queue} gpu=${gpu} log=${LOG_DIR}/gpu${gpu}.log"
done

status=0
for pid in "${pids[@]}"; do
  wait "${pid}" || status=1
done
[[ "${status}" == 0 ]] || exit "${status}"

"${PYTHON}" -m src.main_summarize_fixed_multi_window \
  --root "${RUN_ROOT}" \
  --manifest "${CONFIG_DIR}/manifest.json"
