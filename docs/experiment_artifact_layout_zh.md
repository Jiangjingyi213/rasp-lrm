# 实验产物目录约定

`runs/` 与 `logs/` 使用相同的阶段编号，避免结果、日志和脚本默认路径分离。

| 阶段目录 | 内容 |
|---|---|
| `runs/00_shared/` | 缓存与共享数据 |
| `runs/01_motivation/` | Motivation、正式 counterfactual 与分析图表 |
| `runs/02_baselines/` | Dense、Griffin、FLAP、LLM-Pruner、外部 baseline |
| `runs/03_rasp_zero/` | RASP-Zero offline、runtime banks、router、online eval |
| `runs/04_rasp_train/` | RASP-Train v1/v2/v2.1 与 fair benchmark |
| `runs/05_phase_b/` | aligned banks、Phase B2/v2/v3、B2.5/B2.5b |
| `runs/06_phase_b3_online/` | 当前 uncertainty paired online 验证 |
| `runs/07_stage_aware/` | Stage-aware hidden controller 主线 |

日志对应写入：

```text
logs/01_motivation/
logs/02_baselines/
logs/03_rasp_zero/
logs/04_rasp_train/
logs/05_phase_b/
logs/06_phase_b3_online/
logs/07_stage_aware/
```

服务器同步旧目录后执行：

```bash
bash scripts/56_organize_experiment_artifacts.sh
```

整理脚本可重复执行，只移动仍位于旧顶层路径的产物；目标已存在时会跳过，不删除或覆盖数据。
所有维护中的脚本、配置与文档默认路径均使用新目录。历史 checkpoint 内保存的旧路径字符串仅作为
训练元信息，不参与当前 controller 加载。
