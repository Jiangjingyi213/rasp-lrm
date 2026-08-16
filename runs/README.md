# Experiment Artifacts

实验产物按研究阶段组织：

- `00_shared/`：缓存与共享数据。
- `01_motivation/`：Motivation、正式 counterfactual 数据与分析图表。
- `02_baselines/`：Dense、Griffin、FLAP、LLM-Pruner 和外部 baseline。
- `03_rasp_zero/01_offline/`：RASP-Zero 离线结果。
- `03_rasp_zero/02_runtime_banks/`：RASP-Zero runtime bank。
- `03_rasp_zero/03_runtime_router/`：RASP-Zero router checkpoint。
- `03_rasp_zero/04_online_eval/`：RASP-Zero 在线 smoke/calibration。
- `04_rasp_train/01_legacy/`：RASP-Train v1/v2/v2.1 历史链路。
- `04_rasp_train/02_fair_benchmark/`：共享 split 公平对照。
- `05_phase_b/01_aligned_banks/`：aligned short-window banks。
- `05_phase_b/02_phase_b2/`：Phase B2、v2、v3。
- `05_phase_b/03_phase_b25/`：Phase B2.5 与 B2.5b。
- `06_phase_b3_online/`：当前 uncertainty paired online 验证。
- `07_stage_aware/`：Stage-aware hidden controller 主线。
- `08_stage_calibrated_pruning/`：历史 stage-calibrated structured pruning 主线；后期核心整理结果见 `_late_core_2026_07_31/`，server165 同步整理结果见 `_server165_import_2026_07_31/`。
  - `09_stage_risk_adaptive_v1/`：未启用的 controller 历史分支，不再收集新 counterfactual bank。
- `09_stage_residual_rasp_v2/`：当前主线。以 global WIFV 为底座，测试受控 stage residual、output-aware runtime score 与 continuity-constrained dynamic mask。

使用 `bash scripts/56_organize_experiment_artifacts.sh` 可重复整理从服务器同步回来的旧顶层产物。

## 10_stage_budget_output_aware_v1

New optimization line focused on two mechanisms: online stage budget scheduling
under a 34% actual-pruning target, and output-aware channel scoring. Results are
kept in `runs/10_stage_budget_output_aware_v1/`; logs mirror to
`logs/10_stage_budget_output_aware_v1/`.

### 07_budget_v2_grid_repair

Follow-up to Phase A after the first budget controller improved accuracy but
undershot actual pruning. This branch keeps `trajectory_global + activation` and
only repairs the stage-budget action grid/catch-up strength to re-test at
33.5%-34.5% actual pruning.

### 08_budget_v3_actual_calibrated

Follow-up to A2 after nominal ratio selection still undershot actual pruning.
This branch keeps `trajectory_global + activation` and changes only the budget
controller selection rule so candidate ratios are chosen by estimated realized
mask sparsity under protected-core constraints.

### 09_budget_v4_perf_guarded_32p

Performance-guarded follow-up to v3. This branch keeps
`trajectory_global + activation + estimated_actual` and targets 32.0%-33.5%
actual pruning instead of forcing the full 34% T30 target.

### 10_budget_v41_stage_lift_32p

Small stage-budget lift after v4 smoke landed just below 32% actual pruning.
This branch keeps the same mainline mechanism and raises only stage floor ratios
and budget pressure, with v4 mild used as the immediate performance reference.

## 12_additional_baselines

Paper-facing additional baseline runs kept outside the method-search directories.

### 01_shortgpt_t30

ShortGPT T30 matched baseline for Qwen3-1.7B. It precomputes one frozen
Block-Influence layer-pruning artifact, then evaluates only
`shortgpt_t30_matched` on GSM8K/MATH-500 full and the five-dataset priority
suite.
