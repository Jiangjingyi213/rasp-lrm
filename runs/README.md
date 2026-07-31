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
- `08_stage_calibrated_pruning/`：当前 RASP-LRM stage-calibrated structured pruning 主线；后期核心整理结果见 `_late_core_2026_07_31/`，server165 同步整理结果见 `_server165_import_2026_07_31/`。
  - `09_stage_risk_adaptive_v1/`：新的阶段条件风险控制主线；目录、日志与生成配置独立，`07_stage_aware/` 仅保留为历史诊断。

使用 `bash scripts/56_organize_experiment_artifacts.sh` 可重复整理从服务器同步回来的旧顶层产物。
