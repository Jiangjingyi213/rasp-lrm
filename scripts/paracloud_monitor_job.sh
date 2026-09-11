#!/usr/bin/env bash
set -euo pipefail

JOB_ID="${1:-}"
if [[ -z "${JOB_ID}" ]]; then
  echo "Usage: bash scripts/paracloud_monitor_job.sh <job_id>" >&2
  exit 2
fi

echo "== parajobs =="
parajobs || true

echo
echo "== squeue =="
squeue -u "${USER}" || true

echo
echo "== scontrol =="
scontrol show job "${JOB_ID}" || true

echo
echo "== recent slurm logs =="
shopt -s nullglob
logs=(logs/slurm/*_"${JOB_ID}".out logs/slurm/*_"${JOB_ID}".err slurm-"${JOB_ID}".out)
if [[ "${#logs[@]}" -eq 0 ]]; then
  echo "No local log file found for job ${JOB_ID} yet."
else
  for log in "${logs[@]}"; do
    echo "--- ${log} ---"
    tail -n 80 "${log}" || true
  done
fi

echo
echo "== allocation GPU snapshot =="
srun --jobid="${JOB_ID}" --overlap -N1 -n1 nvidia-smi || true
