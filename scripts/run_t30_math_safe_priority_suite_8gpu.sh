#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

PYTHON_BIN="${PYTHON:-/home/cike/jjy/envs/rasp_qwen3_eval/bin/python}"
CONFIG_PATH="${CONFIG:-configs/stage_calibrated_pruning/mixed_reasoning_seed3_t30_math_safe_priority_suite.yaml}"
PROFILE="${PROFILE:-pilot}"
SOURCE_ROOT="${SOURCE_ROOT:-runs/08_stage_calibrated_pruning/main_pilot_mixed_reasoning_seed3}"
SUITE_ROOT="${RUN_ROOT:-runs/08_stage_calibrated_pruning/main_pilot_mixed_reasoning_seed3_t30_math_safe_priority_suite}"
FINAL_METHODS="${STAGE_FINAL_METHODS:-structured_dense,static_t30_0p37,t30_math_safe}"
FINAL_EVAL_LIMIT="${STAGE_FINAL_EVAL_LIMIT:--1}"
LOG_DIR="${LOG_DIR:-logs}"
HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
SEED="${STAGE_SEED:-3}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"

read -r -a GPUS <<< "${FINAL_GPUS:-0 1 2 3 4 5 6 7}"
SHARD_COUNT="${STAGE_FINAL_SHARD_COUNT:-${#GPUS[@]}}"
if [[ "${#GPUS[@]}" -lt "${SHARD_COUNT}" ]]; then
  echo "Need at least ${SHARD_COUNT} GPU ids, got ${#GPUS[@]}: ${GPUS[*]}" >&2
  exit 2
fi

for artifact_dir in 03_selected 04_masks; do
  if [[ ! -d "${SOURCE_ROOT}/${artifact_dir}" ]]; then
    echo "Missing reusable artifact: ${SOURCE_ROOT}/${artifact_dir}" >&2
    echo "Set SOURCE_ROOT to an existing mixed pilot run that contains 03_selected and 04_masks." >&2
    exit 2
  fi
done

DATASETS=(
  aime2024
  aime2025
  amc2023
  gpqa_diamond
  arc_challenge
)

mkdir -p "${LOG_DIR}" "${SUITE_ROOT}"

write_suite_summaries() {
  SUITE_ROOT="${SUITE_ROOT}" "${PYTHON_BIN}" - <<'PY'
import json
import os
from pathlib import Path

suite = Path(os.environ["SUITE_ROOT"])
dataset_summaries = []

def pct(value):
    if value is None:
        return "-"
    return f"{100 * float(value):.2f}%"

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
        dataset_summaries.append({"dataset": dataset_name, "run_root": str(run_root), "rows": rows})
        md_lines = [
            f"# {dataset_name} Priority Suite Summary",
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
            md_lines.append(
                f"| `{row['method']}` | {correct} / {total} | {pct(row['accuracy'])} | "
                f"{pct(row['actual_pruning'])} | {pct(row['fallback'])} | "
                f"{pct(row['truncation'])} | {mean_tokens} |"
            )
        (run_root / "06_final" / "summary.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")

aggregate = {
    "schema": "priority_suite_aggregate_v1",
    "completed_datasets": [row["dataset"] for row in dataset_summaries],
    "datasets": dataset_summaries,
}
(suite / "aggregate_summary.json").write_text(
    json.dumps(aggregate, ensure_ascii=False, indent=2) + "\n",
    encoding="utf-8",
)

lines = [
    "# Priority Suite Aggregate Summary",
    "",
    f"- completed_datasets: {len(dataset_summaries)}",
    "",
    "| dataset | method | correct / total | accuracy | actual pruning | fallback | truncation | mean tokens |",
    "|---|---|---:|---:|---:|---:|---:|---:|",
]
for item in dataset_summaries:
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

for dataset in "${DATASETS[@]}"; do
  DATASET_ROOT="${SUITE_ROOT}/${dataset}"
  SUMMARY_PATH="${DATASET_ROOT}/06_final/summary.json"
  if [[ "${SKIP_EXISTING}" == "1" && -f "${SUMMARY_PATH}" ]]; then
    echo "SKIP dataset=${dataset}; existing summary=${SUMMARY_PATH}"
    write_suite_summaries
    continue
  fi

  mkdir -p "${DATASET_ROOT}"
  echo "START dataset=${dataset} root=${DATASET_ROOT}"

  HF_ENDPOINT="${HF_ENDPOINT}" \
  HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}" \
  HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-120}" \
  HF_HUB_ETAG_TIMEOUT="${HF_HUB_ETAG_TIMEOUT:-120}" \
  STAGE_SEED="${SEED}" \
  STAGE_WORKFLOW_ROOT="${DATASET_ROOT}" \
  STAGE_FINAL_DATASET_NAME="${dataset}" \
  "${PYTHON_BIN}" -m src.main_stage_calibrated_pruning \
    --config "${CONFIG_PATH}" \
    --profile "${PROFILE}" \
    --stage preflight \
    --force

  for artifact_dir in 03_selected 04_masks; do
    if [[ ! -d "${DATASET_ROOT}/${artifact_dir}" ]]; then
      echo "Reusing ${SOURCE_ROOT}/${artifact_dir} -> ${DATASET_ROOT}/${artifact_dir}"
      cp -a "${SOURCE_ROOT}/${artifact_dir}" "${DATASET_ROOT}/"
    else
      echo "Keeping existing ${DATASET_ROOT}/${artifact_dir}"
    fi
  done

  echo "START dataset=${dataset} sharded final; methods=${FINAL_METHODS}; shards=${SHARD_COUNT}"
  pids=()
  for shard_index in $(seq 0 $((SHARD_COUNT - 1))); do
    gpu="${GPUS[$shard_index]}"
    log_path="${LOG_DIR}/priority_suite_${dataset}_seed${SEED}_shard${shard_index}_of${SHARD_COUNT}_gpu${gpu}.log"
    echo "Launching dataset=${dataset} shard ${shard_index}/${SHARD_COUNT} on GPU ${gpu}; log=${log_path}"
    CUDA_VISIBLE_DEVICES="${gpu}" \
    HF_ENDPOINT="${HF_ENDPOINT}" \
    HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}" \
    HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-120}" \
    HF_HUB_ETAG_TIMEOUT="${HF_HUB_ETAG_TIMEOUT:-120}" \
    STAGE_SEED="${SEED}" \
    STAGE_WORKFLOW_ROOT="${DATASET_ROOT}" \
    STAGE_FINAL_DATASET_NAME="${dataset}" \
    STAGE_FINAL_EVAL_LIMIT="${FINAL_EVAL_LIMIT}" \
    STAGE_FINAL_METHODS="${FINAL_METHODS}" \
    STAGE_FINAL_SHARD_INDEX="${shard_index}" \
    STAGE_FINAL_SHARD_COUNT="${SHARD_COUNT}" \
    "${PYTHON_BIN}" -m src.main_stage_calibrated_pruning \
      --config "${CONFIG_PATH}" \
      --profile "${PROFILE}" \
      --stage evaluate_final \
      --force \
      > "${log_path}" 2>&1 &
    pids+=("$!")
  done

  failed=0
  for pid in "${pids[@]}"; do
    if ! wait "${pid}"; then
      failed=1
    fi
  done
  if [[ "${failed}" -ne 0 ]]; then
    echo "At least one priority suite shard failed for dataset=${dataset}. Check logs under ${LOG_DIR}." >&2
    exit 1
  fi

  echo "START merge_final_shards dataset=${dataset}"
  HF_ENDPOINT="${HF_ENDPOINT}" \
  STAGE_SEED="${SEED}" \
  STAGE_WORKFLOW_ROOT="${DATASET_ROOT}" \
  STAGE_FINAL_DATASET_NAME="${dataset}" \
  STAGE_FINAL_EVAL_LIMIT="${FINAL_EVAL_LIMIT}" \
  STAGE_FINAL_SHARD_COUNT="${SHARD_COUNT}" \
  "${PYTHON_BIN}" -m src.main_stage_calibrated_pruning \
    --config "${CONFIG_PATH}" \
    --profile "${PROFILE}" \
    --stage merge_final_shards \
    --force
  echo "DONE dataset=${dataset}; summary=${SUMMARY_PATH}"
  write_suite_summaries
done

echo "ALL DONE: ${SUITE_ROOT}/aggregate_summary.md"
