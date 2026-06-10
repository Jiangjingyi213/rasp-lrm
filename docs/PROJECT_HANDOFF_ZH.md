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

v2.1 已完成代码实现、服务器训练与离线验证。当前设计：

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

### RASP-Train v2.1 离线结果

| 方法 | B15 ratio | B15 flip | B15 unsafe | B20 ratio | B20 flip | B20 unsafe |
|---|---:|---:|---:|---:|---:|---:|
| RASP-Train v2.1 | 0.1036 | 0.0603 | 0.0661 | 0.1224 | 0.0623 | 0.0720 |
| RASP-Zero | 0.1372 | 0.0642 | 0.0778 | 0.1825 | 0.0856 | 0.1012 |

v2.1 的共享风险模型 ROC-AUC/PR-AUC 为 `0.8406/0.6309`，monotonic violation rate 为 `0`，
所有 fold-stable calibration constraints 均满足。相对 v2，v2.1 明显降低 flip/unsafe，但预算
利用率只有 B15 `69.1%`、B20 `61.2%`，过于保守。

诊断性 test threshold sweep 表明：

- 在安全性不差于 RASP-Zero 时，B15 最大 ratio 仅 `0.1120`；
- 在安全性不差于 RASP-Zero 时，B20 最大 ratio 仅 `0.1492`；
- 与 RASP-Zero 接近的 ratio 下，v2.1 flip/unsafe 明显更高；
- 没有任何 threshold 同时达到不低于 RASP-Zero 的 ratio 和不高于它的 flip/unsafe。

因此 v2.1 修复了校准安全性，但仍未形成效率-安全 Pareto 优势，不进入在线 smoke。

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

当前已完成系统瓶颈审计，详细见
[`rasp_train_bottleneck_diagnosis_zh.md`](rasp_train_bottleneck_diagnosis_zh.md)。

下一步不应继续单纯调 threshold 或扩大 policy，而应先修复数据与评估协议：

1. 建立共享 split manifest，做同 split、同标签、同 controller 的 linear/nonlinear 公平对照；
2. 重建 action 只作用一个 16-token window、ranking 与 runtime 一致的 counterfactual bank；
3. 同时记录窗口内 drift 与最终 answer flip，改善 credit assignment；
4. 用 policy rollout 做一到两轮 state aggregation，覆盖剪枝后的状态分布；
5. 在 aligned bank 上再比较 linear、nonlinear 与 RASP-Zero residual。

其中第 1 步的代码已经完成，入口为 `scripts/39_prepare_rasp_train_fair_benchmark.sh` 至
`scripts/42_summarize_rasp_train_fair_benchmark.py`。它复用 v2.1 bank，使用共享 split manifest，
并在同标签、同 controller、同 calibration/test 口径下比较五种 policy variant 与两种标签定义。
完整 3-seed 训练与评估已经完成，结果写入 `runs/rasp_train_fair_benchmark/`。

公平 benchmark 的主要结论：

- raw-flip hidden-nonlinear 风险预测最强，ROC-AUC/PR-AUC 为 `0.8457/0.5914`；
- 它的平均 controller 结果为 B15 `ratio/flip/unsafe = 0.1341/0.0585/0.0753`，
  B20 为 `0.1695/0.0777/0.0965`；
- uncertainty-linear 安全性更高，说明旧 bank 风险中存在大量通用不确定性/位置可预测信号；
- monotonic unsafe 标签没有稳定优于 raw flip；
- 所有 calibration split 均通过约束，但 seed 3 test 明显恶化，确认当前瓶颈是 calibration
  distribution shift 与 bank/runtime 定义不一致，而不是继续扩大 policy 即可解决。

因此下一步应保留 raw-flip hidden-nonlinear 与 uncertainty-linear 作为主要对照，增加分层或
out-of-fold calibration，并开始重建 action 只作用一个 runtime window 的 aligned bank。旧
RASP-Zero 尚未在共享 manifests 上重跑，不能声称公平 benchmark 已严格超过它。

Phase B1 aligned window bank 代码已经实现。新采集器从原始 prompt 建立固定 ranking，dense
重放到固定 16-token boundary，仅在下一个窗口施加 ratio，之后恢复 dense；同时记录局部 token
divergence、窗口末 hidden drift 和最终 paired answer flip。入口为
`scripts/44_collect_rasp_phase_b_aligned_bank.sh`。下一步先采集每个数据源 25 题 smoke 并检查
所有 shard validation，再启动正式 bank。

Phase B1 smoke 已完成：10/10 shard 均通过严格 validation，dense replay flip 为 `0`，共得到
38 个 dense-correct problems、228 个 fixed-window boundaries 和 1596 条 counterfactual rows。
非零动作最终 flip rate 为 `3.51%`，但 window divergence 和 hidden drift 随 ratio 清晰增长，
证明短窗口 action 与辅助 drift 标签有效。由于最终 flip 稀疏，Phase B2 应使用 raw flip + drift
多任务训练。Smoke 只覆盖前 6 个窗口，下一步建议先采集每数据源 100 题、每题前 12 个窗口，
验证后半程分布与采集成本后再扩大到 500 题。

每数据源 100 题的中型采集已经完成，20/20 shard 均通过 validation，得到 164 个 dense-correct
problems、984 个 boundaries、6888 条 rows，非零动作 final-flip 正例率为 `4.00%`。但本次实际
仍只覆盖前 6 个窗口：旧 worker 只要发现 `status=ok` 就跳过，没有确认 validation 是否匹配新
配置。该问题已修复，worker 现在会在 max-boundary 配置改变时复用已有 dense trajectories 并重采
counterfactual。下一步先在现有 100 题上补齐 12-window bank，再决定扩到 500 题和进行正式
Phase B2 训练；多任务模型代码可并行开发，但不能用当前前 96-token bank 得出正式结论。

12-window bank 已完成并暂存于 `runs/未命名/`：20/20 shard 均通过严格 validation，164 个
dense-correct problems 产生 1926 个 boundary、13482 条 rows；非零动作 final-flip 正例为
`561/11556 = 4.85%`，覆盖 108 个问题和 299 个 boundary。后 6 个窗口 final-flip rate 为
`5.75%`，高于前 6 个窗口的 `4.00%`，证明补采后半程是必要的。局部 drift 与最终 flip 只有弱相关，
因此 Phase B2 应使用 raw final flip 主目标 + token/hidden drift 辅助目标，而不能用 drift 直接
定义安全。当前数据可进入 Phase B2 原型训练；正式结论前仍需扩大 problem 数量。
Phase B2 数据准备还应过滤 25 个提前遇到 EOS 的 incomplete-window boundaries，保证所有训练
action 都具有完整 16-token 暴露时长。

Phase B2 多任务原型代码已实现。它以 raw paired final flip 为主目标，以 token divergence 和
hidden cosine drift 为辅助回归目标；默认比较 hidden-multitask、hidden-flip-only 和
uncertainty-multitask。数据准备会过滤 incomplete-window boundaries，并生成 dataset/positive
分层的三 seed problem manifests。入口为 `scripts/45_prepare_rasp_phase_b2_data.sh` 至
`scripts/48_summarize_rasp_phase_b2.py`。

Phase B2 三 seed 原型已完成，结果位于 `runs/rasp_phase_b2/`。主要结论：

- 风险预测仍弱：hidden-flip-only、hidden-multitask、uncertainty-multitask 的 test ROC-AUC
  分别为 `0.6099/0.6202/0.6450`，PR-AUC 为 `0.0694/0.0736/0.0802`；
- drift 多任务相对 hidden-flip-only 只有小幅预测增益，且没有形成稳定 controller 增益；
- uncertainty-only 反而取得最高平均 AUC，当前 164 个问题上尚无证据证明 hidden state 提供了
  稳定的额外短窗口安全信号；
- hidden-flip-only 的平均 controller 结果为 B15 `ratio/flip=0.0923/0.0180`，
  B20 `0.1647/0.0236`，但 seed 间方差很大；seed 2 的 test flip 上升到
  `0.0407/0.0441`；
- 所有 calibration constraints 虽显示满足，但该状态只描述 calibration split，不能代表 test
  稳定满足。当前训练还使用同一 calibration split 做 best-checkpoint early stopping 和 threshold
  calibration，存在 calibration reuse，必须修复后重跑。

因此当前不能进入在线 rollout，也不应直接把 Phase C 数据聚合作为下一步。先增加独立 validation
split，重跑无泄漏 Phase B2，并补 position-only / ratio-only / RASP-Zero residual 公平对照；
若 hidden 仍不能稳定超过简单特征，应接受当前 hidden signal 不足，而不是继续扩大 policy。

Phase B2 v2 无泄漏重跑代码已实现，默认写入 `runs/rasp_phase_b2_v2/`。四路分层 split 为
train/validation/calibration/test，当前数据实际为 `97/17/25/25` 个问题；新增 ratio-only、
position-only、uncertainty-flip-only 对照。服务器命令与验收条件见
[`rasp_train_bottleneck_diagnosis_zh.md`](rasp_train_bottleneck_diagnosis_zh.md)。

v2 首次训练已完成 18/18 checkpoint，但尚未执行 test eval。训练结果审计发现一维
`position_flip_only` 被 `LayerNorm(1)` 擦除，因而错误地与 ratio-only 完全相同；该实现已修复。
这触发了后续端到端审查，v2 未执行正式 test eval。

随后完成的端到端审查表明，不能只重训两个基线，v2 应停止。旧永久干预 bank 的 nonzero flip
正例率为 `19.63%`，aligned 短窗口 bank 只有 `4.89%`，任务本身显著更难；同时确认 aligned
collector 的局部 token-divergence window 错位、position 与 runtime 定义不一致、混合输入
LayerNorm 破坏简单特征对照，以及 split 只按“是否有正例”而未按正例负担分层。Phase B2 v3 已
修复这些问题并增加 linear/nonlinear 对照。下一步
重采 `runs/rasp_phase_b_aligned_bank_v2/`，再训练 `runs/rasp_phase_b2_v3/`；v2 不执行正式
test，也不作为 hidden signal 的停止结论。

Phase B2 v3 数据已验收通过：新版 aligned bank 20/20 shard 均为 `ok`，164 个问题产生 1899 个
完整 affected-window boundary、11394 个 nonzero action rows 和 558 个 final-flip 正例；
hidden/action 强一致检查通过。window 修复改变了 `36.2%` 的 token-divergence 标签，position
最大值从错误的 `1.0` 修正为 runtime 对齐的 `0.2292`，final-flip 主标签保持不变。现在可以启动
六个 linear/nonlinear 裁决 variant，四卡队列命令见
[`rasp_train_bottleneck_diagnosis_zh.md`](rasp_train_bottleneck_diagnosis_zh.md)。

Phase B2 v3 六个裁决 variant 已完成。uncertainty nonlinear 在 test 上最佳：
ROC-AUC `0.627 ± 0.057`、PR-AUC `0.098 ± 0.022`；hidden/combined nonlinear 仅为
`0.555 ± 0.060`、`0.083 ± 0.019`，hidden/combined linear 接近随机。当前 `hidden` variant
实际包含 raw hidden 与 uncertainty/position/dataset 特征，但 raw hidden 仍未带来稳定增益；
高维 combined checkpoint 也普遍在 epoch `1–2` 停止，显示明显过拟合。当前裁决为：
**不进入 Phase C、不运行 multitask 扩张，先执行低维/正则化 hidden residual 的 Phase B2.5
受控诊断。**

注意该裁决不等于推翻 Motivation。Motivation 的 1342-problem、problem-level OOF hidden probe
五折 ROC-AUC 稳定在 `0.812–0.843`，证明 hidden 能表示永久/强结构扰动下的状态脆弱性。
Phase B2 v3 检验的是仅作用 16 token、随后恢复 dense 的 MLP 动作是否造成最终答案 flip，属于
更稀疏且可恢复的新任务。当前 aligned 数据语义、hidden/action 对齐和 split/test 隔离正确；
尚未充分的是 hidden 模型裁决协议：缺少纯 hidden、train-only 标准化/降维、以及相对 uncertainty
基线的 residual 增量对照。

Phase B2.5 受控诊断链路已实现，默认输出 `runs/rasp_phase_b25/`。它固定复用 v3 数据与 split，
比较 uncertainty baseline、纯 hidden PCA linear/nonlinear、以及 uncertainty + 低容量 hidden/action residual；
所有标准化/PCA 仅在 train rows 拟合。测试新增 boundary any-flip 与 problem-bootstrap 95% CI。
执行命令与 Phase C 准入标准见 `rasp_train_bottleneck_diagnosis_zh.md`。

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
