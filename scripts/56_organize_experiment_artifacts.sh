#!/usr/bin/env bash
set -euo pipefail

RUNS_ROOT="${RUNS_ROOT:-runs}"
LOGS_ROOT="${LOGS_ROOT:-logs}"

move_entry() {
  local source="$1"
  local destination_dir="$2"
  if [[ ! -e "${source}" ]]; then
    return
  fi
  mkdir -p "${destination_dir}"
  local destination="${destination_dir}/$(basename "${source}")"
  if [[ -e "${destination}" ]]; then
    echo "skip: destination already exists: ${destination}" >&2
    return
  fi
  mv "${source}" "${destination}"
  echo "moved: ${source} -> ${destination}"
}

mkdir -p \
  "${RUNS_ROOT}/00_shared" \
  "${RUNS_ROOT}/01_motivation" \
  "${RUNS_ROOT}/02_baselines" \
  "${RUNS_ROOT}/03_rasp_zero/01_offline" \
  "${RUNS_ROOT}/03_rasp_zero/02_runtime_banks" \
  "${RUNS_ROOT}/03_rasp_zero/03_runtime_router" \
  "${RUNS_ROOT}/03_rasp_zero/04_online_eval" \
  "${RUNS_ROOT}/04_rasp_train/01_legacy" \
  "${RUNS_ROOT}/04_rasp_train/02_fair_benchmark" \
  "${RUNS_ROOT}/05_phase_b/01_aligned_banks" \
  "${RUNS_ROOT}/05_phase_b/02_phase_b2" \
  "${RUNS_ROOT}/05_phase_b/03_phase_b25" \
  "${RUNS_ROOT}/06_phase_b3_online" \
  "${RUNS_ROOT}/07_stage_aware" \
  "${LOGS_ROOT}/01_motivation" \
  "${LOGS_ROOT}/02_baselines" \
  "${LOGS_ROOT}/03_rasp_zero" \
  "${LOGS_ROOT}/04_rasp_train" \
  "${LOGS_ROOT}/05_phase_b" \
  "${LOGS_ROOT}/06_phase_b3_online" \
  "${LOGS_ROOT}/07_stage_aware"

move_entry "${RUNS_ROOT}/cache" "${RUNS_ROOT}/00_shared"
move_entry "${RUNS_ROOT}/data" "${RUNS_ROOT}/00_shared"

for path in \
  "${RUNS_ROOT}"/formal_qwen3_* \
  "${RUNS_ROOT}"/minimal_* \
  "${RUNS_ROOT}"/motivation_* \
  "${RUNS_ROOT}"/p100_*; do
  move_entry "${path}" "${RUNS_ROOT}/01_motivation"
done

for path in \
  "${RUNS_ROOT}"/eval_dense_* \
  "${RUNS_ROOT}"/eval_griffin_* \
  "${RUNS_ROOT}"/eval_flap_* \
  "${RUNS_ROOT}"/flap_* \
  "${RUNS_ROOT}"/llm_pruner_* \
  "${RUNS_ROOT}"/external_baselines; do
  move_entry "${path}" "${RUNS_ROOT}/02_baselines"
done

for path in "${RUNS_ROOT}"/rasp_zero_offline*; do
  move_entry "${path}" "${RUNS_ROOT}/03_rasp_zero/01_offline"
done
for path in "${RUNS_ROOT}"/rasp_zero_runtime_bank*; do
  move_entry "${path}" "${RUNS_ROOT}/03_rasp_zero/02_runtime_banks"
done
for path in "${RUNS_ROOT}"/rasp_zero_runtime_router; do
  move_entry "${path}" "${RUNS_ROOT}/03_rasp_zero/03_runtime_router"
done
for path in \
  "${RUNS_ROOT}"/rasp_zero_online_* \
  "${RUNS_ROOT}"/rasp_zero_runtime_smoke_dense \
  "${RUNS_ROOT}"/eval_rasp_zero_runtime_*; do
  move_entry "${path}" "${RUNS_ROOT}/03_rasp_zero/04_online_eval"
done

for path in \
  "${RUNS_ROOT}"/rasp_train_v1 \
  "${RUNS_ROOT}"/rasp_train_v2 \
  "${RUNS_ROOT}"/rasp_train_v2_1 \
  "${RUNS_ROOT}"/eval_rasp_train_*; do
  move_entry "${path}" "${RUNS_ROOT}/04_rasp_train/01_legacy"
done
move_entry "${RUNS_ROOT}/rasp_train_fair_benchmark" "${RUNS_ROOT}/04_rasp_train/02_fair_benchmark"

for path in "${RUNS_ROOT}"/rasp_phase_b_aligned_bank*; do
  move_entry "${path}" "${RUNS_ROOT}/05_phase_b/01_aligned_banks"
done
for path in "${RUNS_ROOT}"/rasp_phase_b2 "${RUNS_ROOT}"/rasp_phase_b2_v2 "${RUNS_ROOT}"/rasp_phase_b2_v3; do
  move_entry "${path}" "${RUNS_ROOT}/05_phase_b/02_phase_b2"
done
for path in "${RUNS_ROOT}"/rasp_phase_b25 "${RUNS_ROOT}"/rasp_phase_b25b; do
  move_entry "${path}" "${RUNS_ROOT}/05_phase_b/03_phase_b25"
done
for path in "${RUNS_ROOT}"/rasp_phase_b2_uncertainty_online*; do
  move_entry "${path}" "${RUNS_ROOT}/06_phase_b3_online"
done

for path in "${LOGS_ROOT}"/motivation_* "${LOGS_ROOT}"/formal_*; do
  move_entry "${path}" "${LOGS_ROOT}/01_motivation"
done
for path in "${LOGS_ROOT}"/flap_* "${LOGS_ROOT}"/griffin_* "${LOGS_ROOT}"/llm_pruner_*; do
  move_entry "${path}" "${LOGS_ROOT}/02_baselines"
done
for path in "${LOGS_ROOT}"/rasp_zero_*; do
  move_entry "${path}" "${LOGS_ROOT}/03_rasp_zero"
done
for path in "${LOGS_ROOT}"/rasp_train_*; do
  move_entry "${path}" "${LOGS_ROOT}/04_rasp_train"
done
for path in "${LOGS_ROOT}"/rasp_phase_b*; do
  move_entry "${path}" "${LOGS_ROOT}/05_phase_b"
done
for path in "${LOGS_ROOT}"/phase_b2_uncertainty_online_*; do
  move_entry "${path}" "${LOGS_ROOT}/06_phase_b3_online"
done
