# RASP-LRM 项目交接说明

> 本文件用于让新的对话快速了解项目状态。这里只记录主线结论和下一步，具体方法、实验口径与结果解释请继续阅读文末列出的详细文档。

## 1. 项目目标

项目研究 **Large Reasoning Model（LRM）的 reasoning-aware dynamic structured pruning**。

核心观点是：

> LRM 对模型结构的依赖会随 reasoning process 变化。剪枝策略不应只由 prompt 或静态重要性决定，还应根据当前 reasoning state 动态调整。

当前主模型为 `Qwen/Qwen3-1.7B`，主要任务为 GSM8K 与 MATH500。服务器主要使用 P100，因此模型默认采用 `float32 + eager attention`。

## 2. 已完成工作

### Motivation 实验

已经完成 GSM8K 与 MATH500 的正式 counterfactual motivation 实验，覆盖 reasoning segmentation、不同结构/比例的反事实剪枝、oracle、entropy、hidden-state probe 和 heatmap 分析。

主要结论：

- 不同 reasoning step 的剪枝敏感性明显不同；
- 不同 layer/module 的风险明显不同；
- step-level oracle 优于 static/prompt-level policy；
- hidden-state probe 能预测部分剪枝风险；
- 简单 entropy 信号不足以稳定指导在线剪枝。

主要结果位于：

```text
runs/formal_qwen3_gsm8k_full_s0
runs/formal_qwen3_gsm8k_full_s1
runs/formal_qwen3_math500_full_s0
runs/formal_qwen3_math500_full_s1
runs/formal_qwen3_gsm8k_math500_combined.json
runs/motivation_analysis
```

### Baseline

已经建立或尝试：

- Dense Qwen3；
- GRIFFIN-style FFN dynamic pruning；
- FLAP-MLP static pruning；
- LLM-Pruner-style MLP pruning。

GRIFFIN 和 FLAP 已形成可评估 baseline。LLM-Pruner 在 Qwen3 上即使轻度剪枝也出现严重生成退化，因此相关结果不能直接作为可信正式 baseline，需要谨慎表述。

### RASP-Zero

已经完成：

- RASP-Zero offline policy evaluation；
- runtime counterfactual bank；
- action-conditioned risk router；
- problem-level split；
- budget-aware safe oracle；
- 在线 RASP-Zero smoke 与阈值校准。

客观结论：

- 离线风险预测具有可学习信号；
- 简单的 `risk threshold + budget rule` 在线策略仍容易造成 reasoning drift；
- 手工 conservative cap 没有稳定恢复 GSM8K 准确率。

因此，**RASP-Zero 当前定位为分析型 prototype**：用于证明问题、构造训练数据和提供 oracle，而不是论文最终方法。

## 3. 当前阶段：RASP-Train v2.1

RASP-Train v1 已完成离线实验，但弱于 matched-budget RASP-Zero：

- B15：ratio `0.1241`，flip rate `0.1023`；RASP-Zero 为 `0.0621`。
- B20：ratio `0.1761`，flip rate `0.1368`；RASP-Zero 为 `0.0851`。

因此 v1 作为 oracle-ratio classification 失败消融保留，不进入在线实验。RASP-Train v2
action-risk policy 已完成服务器训练和离线评估，但仍未超过 RASP-Zero，因此也暂不进入在线实验。

v2.1 已完成代码实现，等待服务器训练与离线验证。当前设计：

- 不训练 Qwen3 本体，只训练轻量 ratio policy；
- 风险输入仅包含 hidden state、entropy、confidence、position、dataset/domain 和 candidate ratio；
- `target_budget/available_budget` 不进入风险网络，只由 causal controller 使用；
- 对每个候选 ratio 输出 unsafe probability：

```text
0.00, 0.02, 0.05, 0.10, 0.20, 0.30, 0.40
```

- 采用 action-conditioned multi-label risk learning；
- loss 为 weighted BCE、ratio-risk monotonic penalty 和 safe/unsafe ranking loss；
- 使用 problem-level 70/15/15 train/calibration/test；
- B15/B20 使用同一个共享风险 checkpoint；
- calibration problems 分别为 B15/B20 选择 threshold，并检查三折 problem-level 稳定性；
- checkpoint 记录各预算阈值、总体约束和最差 fold 指标；
- 在线由 calibrated threshold 与 causal prefix budget 共同选 ratio；
- 当前仍使用 logical MLP mask，只报告 activated-MLP proxy，不宣称真实硬件加速。

v1/v2 分别保留在 `runs/rasp_train_v1/` 与 `runs/rasp_train_v2/`。v2.1 默认写入
`runs/rasp_train_v2_1/`，共享 checkpoint 为 `shared/rasp_train_policy.pt`。

### RASP-Train v2 离线结果

| 方法 | B15 ratio | B15 flip | B15 unsafe | B20 ratio | B20 flip | B20 unsafe |
|---|---:|---:|---:|---:|---:|---:|
| RASP-Train v2 | 0.1336 | 0.0778 | 0.0875 | 0.1641 | 0.0895 | 0.1051 |
| RASP-Zero | 0.1372 | 0.0642 | 0.0778 | 0.1825 | 0.0856 | 0.1012 |

v2 的 action-risk prediction 有有效信号：B15/B20 ROC-AUC 分别为 `0.8611/0.8551`，
PR-AUC 为 `0.6628/0.6483`，monotonic violation rate 均为 `0`。但是独立 test 上：

- B15 比 RASP-Zero 少剪 `0.0036`，flip 仍高 `0.0136`；
- B20 比 RASP-Zero 少剪 `0.0184`，flip 仍高 `0.0039`；
- calibration 到 test 存在约 `0.01-0.02` 的 flip/unsafe 泛化差距；
- test 阈值扫描表明，更保守阈值只能通过明显降低 ratio 达到不差于 RASP-Zero，不能形成 Pareto 优势。

因此当前结论是：v2 相对 v1 呈明显改善趋势，但两版 test split 不同，不能作严格逐项比较；
v2 仍没有通过“相近 ratio 下优于 RASP-Zero”的离线门槛。另已确认 B15/B20 的全部
`candidate_unsafe` 标签完全一致，进一步说明风险模型不应依赖预算字段或按预算分别训练。

新增入口：

```text
scripts/35_prepare_rasp_train_v1_data.sh
scripts/36_train_rasp_train_v1.sh
scripts/37_eval_rasp_train_v1_offline.sh
scripts/38_eval_rasp_train_v1_online_smoke.sh
```

## 4. 下一步

在服务器运行 v2.1 离线链路：

```bash
export PYTHON=/home/cike/jjy/envs/rasp_qwen3/bin/python
mkdir -p logs
nohup env CUDA_VISIBLE_DEVICES=0 PYTHON="$PYTHON" bash -c '
set -e
bash scripts/35_prepare_rasp_train_v1_data.sh
bash scripts/36_train_rasp_train_v1.sh
bash scripts/37_eval_rasp_train_v1_offline.sh
' > logs/rasp_train_v2_1_offline.log 2>&1 &
echo $! > logs/rasp_train_v2_1_offline.pid
tail -f logs/rasp_train_v2_1_offline.log
```

重点检查 `runs/rasp_train_v2_1/shared/13_rasp_train_metrics.json` 中
`all_calibration_constraints_satisfied`，以及 B15/B20 独立 test 是否形成相对 RASP-Zero 的
Pareto 优势。未通过前不运行在线 smoke。

## 5. 建议优先阅读

- Motivation 全链路与专业概念解释：  
  [`motivation_experiment_details_explained.md`](motivation_experiment_details_explained.md)
- 当前科研进度周报：  
  [`research_weekly_report_2026_05_29.md`](research_weekly_report_2026_05_29.md)
- Baseline 结果汇总：  
  [`baseline_results_summary.md`](baseline_results_summary.md)
- RASP-Zero offline v2：  
  [`rasp_zero_offline_v2_zh.md`](rasp_zero_offline_v2_zh.md)
- RASP-Zero runtime 与在线实验：  
  [`rasp_zero_runtime_zh.md`](rasp_zero_runtime_zh.md)  
  [`rasp_zero_online_v1_zh.md`](rasp_zero_online_v1_zh.md)
- RASP-Train v1 方法和运行说明：  
  [`rasp_train_v1_zh.md`](rasp_train_v1_zh.md)
- FLAP / LLM-Pruner 的 Qwen3 迁移分析：  
  [`qwen3_flap_llm_pruner_porting_report.md`](qwen3_flap_llm_pruner_porting_report.md)

## 6. 给新对话的提醒

- 不要把 RASP-Zero 在线结果描述为最终成功方法；
- 不要把 logical mask 描述为真实 wall-clock speedup；
- 所有训练、验证划分必须保持 problem-level split；
- MATH 训练 bank 使用与 MATH500 test 隔离的数据源，避免评测泄漏；
- RASP-Train 当前最重要的任务是先完成离线验证，不要直接扩大在线全集实验。
- counterfactual segment boundary 与在线 fixed-window state 仍有分布差异，离线通过不等于在线安全。
