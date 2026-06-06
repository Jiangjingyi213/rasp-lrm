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

## 3. 当前阶段：RASP-Train v1

RASP-Train v1 已完成第一轮前置代码审查与修复，但尚未生成正式实验结果。

当前设计：

- 不训练 Qwen3 本体，只训练轻量 ratio policy；
- 输入 hidden state、entropy、confidence、position、dataset/domain、target budget 和当前可用预算；
- 输出 ratio：

```text
0.00, 0.02, 0.05, 0.10, 0.20, 0.30, 0.40
```

- 采用 `budget-aware monotonic-safe oracle imitation`；
- oracle 按每道题独立、按 step 顺序执行 causal prefix budget；
- loss 包括 oracle classification、unsafe action penalty 和 budget penalty；
- 分别训练 `RASP-Train-B15` 与 `RASP-Train-B20`；
- 当前仍使用 logical MLP mask，只报告 activated-MLP proxy，不宣称真实硬件加速。

2026-06-06 已修复 policy 训练 checkpoint 崩溃、跨题 oracle 预算、离线/在线预算回放不一致、
monotonic unsafe 定义不一致，以及在线 smoke 缺少 paired dense control 等问题。旧
RASP-Train dataset/checkpoint 与当前 feature schema 不兼容，必须重新生成。

新增入口：

```text
scripts/35_prepare_rasp_train_v1_data.sh
scripts/36_train_rasp_train_v1.sh
scripts/37_eval_rasp_train_v1_offline.sh
scripts/38_eval_rasp_train_v1_online_smoke.sh
```

## 4. 下一步

在服务器 `rasp_qwen3` 环境中按顺序执行：

```bash
export PYTHON=/home/cike/jjy/envs/rasp_qwen3/bin/python

bash scripts/35_prepare_rasp_train_v1_data.sh
bash scripts/36_train_rasp_train_v1.sh
bash scripts/37_eval_rasp_train_v1_offline.sh
```

首先检查离线结果：

- RASP-Train 在相近平均 ratio 下是否比 RASP-Zero 更低 flip rate；
- `unsafe_over_oracle_rate` 是否下降；
- B=0.15 与 B=0.20 的安全性和预算利用率；
- problem-level validation 上是否稳定。

只有离线结果通过后，再执行：

```bash
bash scripts/38_eval_rasp_train_v1_online_smoke.sh
```

在线阶段先跑 GSM8K-20 与 MATH500-20，并重点检查 dense-correct 样本被剪错的数量，而不是只看总体 accuracy。

在线脚本会自动运行 paired ratio-0 control，并在每个 policy run 下输出
`14_paired_dense_comparison.json`。

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
