#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

PYTHON_BIN="${PYTHON:-/home/cike/jjy/envs/rasp_qwen3_eval/bin/python}"
CONFIG_PATH="${CONFIG:-configs/stage_calibrated_pruning/griffin_prompt_priority_suite.yaml}"
PROFILE="${PROFILE:-pilot}"
SUITE_ROOT="${RUN_ROOT:-runs/08_stage_calibrated_pruning/main_pilot_griffin_prompt_priority_suite}"
FINAL_METHODS="${STAGE_FINAL_METHODS:-griffin_t20_matched,griffin_t30_matched}"
FINAL_EVAL_LIMIT="${STAGE_FINAL_EVAL_LIMIT:--1}"
LOG_DIR="${LOG_DIR:-logs}"
HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
SEED="${STAGE_SEED:-3}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
LOG_PREFIX="${LOG_PREFIX:-griffin_prompt_priority}"

read -r -a GPUS <<< "${FINAL_GPUS:-0 1 2 3 4 5 6 7}"
SHARD_COUNT="${STAGE_FINAL_SHARD_COUNT:-${#GPUS[@]}}"
if [[ "${#GPUS[@]}" -lt "${SHARD_COUNT}" ]]; then
  echo "Need at least ${SHARD_COUNT} GPU ids, got ${#GPUS[@]}: ${GPUS[*]}" >&2
  exit 2
fi

DATASETS=(amc2023 gpqa_diamond arc_challenge)
if [[ -n "${DATASETS_OVERRIDE:-}" ]]; then
  read -r -a DATASETS <<< "${DATASETS_OVERRIDE}"
fi

mkdir -p "${LOG_DIR}" "${SUITE_ROOT}"

write_suite_summaries() {
  SUITE_ROOT="${SUITE_ROOT}" "${PYTHON_BIN}" - <<'PY'
import json
import os
from pathlib import Path

suite = Path(os.environ["SUITE_ROOT"])

def pct(value):
    return "-" if value is None else f"{100 * float(value):.2f}%"

items = []
for summary_path in sorted(suite.glob("*/06_final/summary.json")):
    run_root = summary_path.parents[1]
    data = json.load(open(summary_path, "r", encoding="utf-8"))
    for dataset_name, summaries in data.get("datasets", {}).items():
        rows = []
        for summary in sorted(summaries, key=lambda row: row["method"]["name"]):
            rows.append(
                {
                    "dataset": dataset_name,
                    "method": summary["method"]["name"],
                    "problems": summary.get("problems", 0),
                    "correct": summary.get("correct", 0),
                    "accuracy": summary.get("accuracy"),
                    "actual_pruning": summary.get(
                        "actual_average_mlp_pruning_ratio",
                        summary.get("theoretical_average_mlp_pruning_ratio"),
                    ),
                    "fallback": summary.get("fallback_rate"),
                    "truncation": summary.get("truncation_rate"),
                    "mean_tokens": summary.get("mean_generated_tokens"),
                }
            )
        items.append({"dataset": dataset_name, "run_root": str(run_root), "rows": rows})
        lines = [
            f"# {dataset_name} GRIFFIN Prompt Baseline Summary",
            "",
            f"- run_root: `{run_root}`",
            f"- summary: `{summary_path}`",
            "",
            "| method | correct / total | accuracy | actual pruning | fallback | truncation | mean tokens |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
        for row in rows:
            total = int(row["problems"])
            correct = int(row["correct"])
            mean_tokens = "-" if row["mean_tokens"] is None else f"{float(row['mean_tokens']):.1f}"
            lines.append(
                f"| `{row['method']}` | {correct} / {total} | {pct(row['accuracy'])} | "
                f"{pct(row['actual_pruning'])} | {pct(row['fallback'])} | "
                f"{pct(row['truncation'])} | {mean_tokens} |"
            )
        (run_root / "06_final" / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

(suite / "aggregate_summary.json").write_text(
    json.dumps({"schema": "griffin_prompt_priority_suite_aggregate_v1", "datasets": items}, ensure_ascii=False, indent=2) + "\n",
    encoding="utf-8",
)
lines = [
    "# GRIFFIN Prompt Priority Suite Aggregate Summary",
    "",
    f"- completed_datasets: {len(items)}",
    "",
    "| dataset | method | correct / total | accuracy | actual pruning | fallback | truncation | mean tokens |",
    "|---|---|---:|---:|---:|---:|---:|---:|",
]
for item in items:
    for row in item["rows"]:
        total = int(row["problems"])
        correct = int(row["correct"])
        mean_tokens = "-" if row["mean_tokens"] is None else f"{float(row['mean_tokens']):.1f}"
        lines.append(
            f"| `{item['dataset']}` | `{row['method']}` | {correct} / {total} | "
            f"{pct(row['accuracy'])} | {pct(row['actual_pruning'])} | "
            f"{pct(row['fallback'])} | {pct(row['truncation'])} | {mean_tokens} |"
        )
(suite / "aggregate_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
PY
}

QUEUE_DIR="${SUITE_ROOT}/.priority_queue_seed${SEED}_shards${SHARD_COUNT}_pid$$"
mkdir -p "${QUEUE_DIR}"
job_count=0
active_datasets=()

for dataset in "${DATASETS[@]}"; do
  dataset_root="${SUITE_ROOT}/${dataset}"
  summary_path="${dataset_root}/06_final/summary.json"
  if [[ "${SKIP_EXISTING}" == "1" && -f "${summary_path}" ]]; then
    echo "SKIP dataset=${dataset}; existing summary=${summary_path}"
    write_suite_summaries
    continue
  fi
  mkdir -p "${dataset_root}"
  echo "PREPARE dataset=${dataset} root=${dataset_root}"
  HF_ENDPOINT="${HF_ENDPOINT}" \
  HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}" \
  HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-120}" \
  HF_HUB_ETAG_TIMEOUT="${HF_HUB_ETAG_TIMEOUT:-120}" \
  STAGE_SEED="${SEED}" \
  STAGE_WORKFLOW_ROOT="${dataset_root}" \
  STAGE_FINAL_DATASET_NAME="${dataset}" \
  "${PYTHON_BIN}" -m src.main_stage_calibrated_pruning \
    --config "${CONFIG_PATH}" \
    --profile "${PROFILE}" \
    --stage preflight \
    --force
  active_datasets+=("${dataset}")
  for shard_index in $(seq 0 $((SHARD_COUNT - 1))); do
    shard_summary="$(printf "%s/06_final/summary_shard_%05d_of_%05d.json" "${dataset_root}" "${shard_index}" "${SHARD_COUNT}")"
    if [[ "${SKIP_EXISTING}" == "1" && -f "${shard_summary}" ]]; then
      echo "SKIP dataset=${dataset} shard=${shard_index}; existing ${shard_summary}"
      touch "${QUEUE_DIR}/done_${dataset}_${shard_index}"
      continue
    fi
    job_path="$(printf "%s/job_%05d_%s_%05d.env" "${QUEUE_DIR}" "${job_count}" "${dataset}" "${shard_index}")"
    {
      printf "DATASET=%q\n" "${dataset}"
      printf "SHARD_INDEX=%q\n" "${shard_index}"
    } > "${job_path}"
    job_count=$((job_count + 1))
  done
done

all_dataset_shards_done() {
  local dataset="$1"
  local shard_index
  for shard_index in $(seq 0 $((SHARD_COUNT - 1))); do
    [[ -f "${QUEUE_DIR}/done_${dataset}_${shard_index}" ]] || return 1
  done
  return 0
}

try_merge_dataset() {
  local dataset="$1"
  local dataset_root="${SUITE_ROOT}/${dataset}"
  local lock_dir="${QUEUE_DIR}/merge_${dataset}.lock"
  all_dataset_shards_done "${dataset}" || return 0
  [[ ! -f "${QUEUE_DIR}/merged_${dataset}" ]] || return 0
  mkdir "${lock_dir}" 2>/dev/null || return 0
  if [[ ! -f "${QUEUE_DIR}/merged_${dataset}" ]]; then
    echo "START merge_final_shards dataset=${dataset}"
    HF_ENDPOINT="${HF_ENDPOINT}" \
    STAGE_SEED="${SEED}" \
    STAGE_WORKFLOW_ROOT="${dataset_root}" \
    STAGE_FINAL_DATASET_NAME="${dataset}" \
    STAGE_FINAL_EVAL_LIMIT="${FINAL_EVAL_LIMIT}" \
    STAGE_FINAL_SHARD_COUNT="${SHARD_COUNT}" \
    "${PYTHON_BIN}" -m src.main_stage_calibrated_pruning \
      --config "${CONFIG_PATH}" \
      --profile "${PROFILE}" \
      --stage merge_final_shards \
      --force
    touch "${QUEUE_DIR}/merged_${dataset}"
    write_suite_summaries
    echo "DONE dataset=${dataset}; summary=${dataset_root}/06_final/summary.json"
  fi
  rmdir "${lock_dir}" 2>/dev/null || true
}

if [[ "${job_count}" -eq 0 ]]; then
  echo "No new GRIFFIN priority suite shards to run."
  write_suite_summaries
  echo "ALL DONE: ${SUITE_ROOT}/aggregate_summary.md"
  exit 0
fi

claim_next_job() {
  local gpu="$1"
  local candidate
  local claimed
  while true; do
    for candidate in "${QUEUE_DIR}"/job_*.env; do
      [[ -e "${candidate}" ]] || return 1
      claimed="${candidate}.gpu${gpu}.claimed"
      if mv "${candidate}" "${claimed}" 2>/dev/null; then
        printf "%s\n" "${claimed}"
        return 0
      fi
    done
    sleep 1
  done
}

run_worker() {
  local worker_index="$1"
  local gpu="$2"
  local job_file
  local dataset_root
  local log_path
  while job_file="$(claim_next_job "${gpu}")"; do
    # shellcheck disable=SC1090
    source "${job_file}"
    dataset_root="${SUITE_ROOT}/${DATASET}"
    log_path="${LOG_DIR}/${LOG_PREFIX}_${DATASET}_seed${SEED}_shard${SHARD_INDEX}_of${SHARD_COUNT}_gpu${gpu}.log"
    echo "Launching dataset=${DATASET} shard ${SHARD_INDEX}/${SHARD_COUNT} on GPU ${gpu}; log=${log_path}"
    if CUDA_VISIBLE_DEVICES="${gpu}" \
      HF_ENDPOINT="${HF_ENDPOINT}" \
      HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}" \
      HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-120}" \
      HF_HUB_ETAG_TIMEOUT="${HF_HUB_ETAG_TIMEOUT:-120}" \
      STAGE_SEED="${SEED}" \
      STAGE_WORKFLOW_ROOT="${dataset_root}" \
      STAGE_FINAL_DATASET_NAME="${DATASET}" \
      STAGE_FINAL_EVAL_LIMIT="${FINAL_EVAL_LIMIT}" \
      STAGE_FINAL_METHODS="${FINAL_METHODS}" \
      STAGE_FINAL_SHARD_INDEX="${SHARD_INDEX}" \
      STAGE_FINAL_SHARD_COUNT="${SHARD_COUNT}" \
      "${PYTHON_BIN}" -m src.main_stage_calibrated_pruning \
        --config "${CONFIG_PATH}" \
        --profile "${PROFILE}" \
        --stage evaluate_final \
        --force \
        > "${log_path}" 2>&1; then
      touch "${QUEUE_DIR}/done_${DATASET}_${SHARD_INDEX}"
      try_merge_dataset "${DATASET}"
    else
      echo "FAILED dataset=${DATASET} shard=${SHARD_INDEX} gpu=${gpu}; log=${log_path}" >&2
      touch "${QUEUE_DIR}/failed_${DATASET}_${SHARD_INDEX}"
      return 1
    fi
  done
  echo "Worker ${worker_index} on GPU ${gpu} has no more jobs."
}

pids=()
for worker_index in $(seq 0 $((${#GPUS[@]} - 1))); do
  gpu="${GPUS[$worker_index]}"
  worker_log="${LOG_DIR}/${LOG_PREFIX}_worker${worker_index}_gpu${gpu}.log"
  echo "START worker=${worker_index} gpu=${gpu}; log=${worker_log}"
  run_worker "${worker_index}" "${gpu}" > "${worker_log}" 2>&1 &
  pids+=("$!")
done

failed=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    failed=1
  fi
done
if [[ "${failed}" -ne 0 ]] || compgen -G "${QUEUE_DIR}/failed_*" > /dev/null; then
  echo "At least one GRIFFIN priority worker failed. Check ${LOG_DIR}/${LOG_PREFIX}_worker*_gpu*.log and shard logs." >&2
  exit 1
fi

for dataset in "${active_datasets[@]}"; do
  try_merge_dataset "${dataset}"
done
write_suite_summaries
echo "ALL DONE: ${SUITE_ROOT}/aggregate_summary.md"
