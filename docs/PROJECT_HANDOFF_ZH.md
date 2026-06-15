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
runs/01_motivation/formal_qwen3_gsm8k_full_s0
runs/01_motivation/formal_qwen3_gsm8k_full_s1
runs/01_motivation/formal_qwen3_math500_full_s0
runs/01_motivation/formal_qwen3_math500_full_s1
runs/01_motivation/formal_qwen3_gsm8k_math500_combined.json
runs/01_motivation/motivation_analysis
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

v1/v2 分别保留在 `runs/04_rasp_train/01_legacy/rasp_train_v1/` 与 `runs/04_rasp_train/01_legacy/rasp_train_v2/`。v2.1 默认写入
`runs/04_rasp_train/01_legacy/rasp_train_v2_1/`，共享 checkpoint 为 `shared/rasp_train_policy.pt`。

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
完整 3-seed 训练与评估已经完成，结果写入 `runs/04_rasp_train/02_fair_benchmark/rasp_train_fair_benchmark/`。

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

Phase B2 三 seed 原型已完成，结果位于 `runs/05_phase_b/02_phase_b2/rasp_phase_b2/`。主要结论：

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

Phase B2 v2 无泄漏重跑代码已实现，默认写入 `runs/05_phase_b/02_phase_b2/rasp_phase_b2_v2/`。四路分层 split 为
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
重采 `runs/05_phase_b/01_aligned_banks/rasp_phase_b_aligned_bank_v2/`，再训练 `runs/05_phase_b/02_phase_b2/rasp_phase_b2_v3/`；v2 不执行正式
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

Phase B2.5 受控诊断链路已实现，默认输出 `runs/05_phase_b/03_phase_b25/rasp_phase_b25/`。它固定复用 v3 数据与 split，
比较 uncertainty baseline、纯 hidden PCA linear/nonlinear、以及 uncertainty + 低容量 hidden/action residual；
所有标准化/PCA 仅在 train rows 拟合。测试新增 boundary any-flip 与 problem-bootstrap 95% CI。
执行命令与 Phase C 准入标准见 `rasp_train_bottleneck_diagnosis_zh.md`。

Phase B2.5 已完成。hidden PCA nonlinear 相对本轮 uncertainty baseline 的平均 action/boundary
ROC 仅提高约 `0.011/0.035`，且 action 增益只出现在 seed 1；bootstrap CI 大量重叠。联合
uncertainty + hidden residual 在三个 seed 的 action ROC 全部下降，平均下降约 `0.033`，同时
controller 用更多 ratio 但 flip 更高。当前不进入 Phase C。由于本轮 residual 主分支与 hidden
分支联合从零训练，最后只保留一次低成本 B2.5b：冻结同一个 uncertainty baseline，仅训练
零初始化 hidden/action residual，并做 paired problem bootstrap；若仍失败则停止 hidden router。
B2.5 最佳 hidden PCA nonlinear action ROC `0.565` 也低于 v3 最强 uncertainty router 的 `0.627`。

Phase B2.5b 冻结基线残差链路已实现。它直接冻结 v3 每 seed 的
`uncertainty_flip_only/policy.pt`，仅训练零初始化 hidden-PCA/action residual；validation 可选择
`alpha=0` 精确退回 baseline，test 使用 paired problem bootstrap 报告增量。输出默认位于
`runs/05_phase_b/03_phase_b25/rasp_phase_b25b/`，执行与停止条件见瓶颈诊断文档。

Phase B2.5b 已完成并触发最终停止条件。三 seed action ROC delta 为
`0.0000 / +0.0121 / -0.0644`，平均 `-0.0174 ± 0.0411`；所有 paired 95% CI 均未形成稳定正增量。
seed 1 validation 选择 `alpha=0`，seed 3 residual 明显降低 action ROC 并提高 B15 flip。因此停止
hidden router，不进入 Phase C，不再扩大 hidden bank 或训练 multitask。后续主线使用 v3
`uncertainty_flip_only` 与 conservative RASP-Zero 进行 paired online 验证；hidden probe 仅保留
为长期/强扰动 fragility 的分析模块。

Phase B2 uncertainty 在线入口已经补齐。`src/rasp/phase_b2_controller.py` 会直接加载 v3
`uncertainty_flip_only/policy.pt`，严格复用 `[entropy, confidence, position]`、checkpoint
calibration threshold、单调风险包络与因果 prefix budget，不读取 hidden state。统一 runtime
入口新增 `controller: phase_b2_uncertainty`，首轮 paired smoke 使用
`scripts/55_eval_phase_b2_uncertainty_online_smoke.sh`。这一步用于验证离线短窗口风险信号能否转化
为在线 accuracy/ratio Pareto。由于 v3 bank 只覆盖前 12 个窗口，smoke 在前 192 token 后强制
恢复 dense，避免训练范围外外推；它不代表进入 Phase C，也不能把 logical mask 描述为真实加速。

Phase B3 首轮 20 题 paired online smoke 已完成。GSM8K dense/B15/B20 accuracy 为
`0.75/0.60/0.55`，在全程平均逻辑 ratio 仅 `0.044/0.051` 时仍产生 `3/4` 个真实
dense-correct flip，未形成在线 Pareto。MATH500 原始统计为 `0.45/0.35/0.45`，其中 Evelyn
样例属于 categorical final-answer 匹配假阴性；修正匹配器后应解释为 `0.45/0.40/0.50`，但样本
仍过小且 B20 有一进一出。审查未发现 runtime masking/controller 接线错误；核心失配是离线 bank
只评估“单个 16-token window 后恢复 dense”，在线却连续施加多个窗口，导致未训练过的累积状态
漂移。当前不扩三 seed/大样本，也不直接进入完整 Phase C；下一步仅做单窗口在线 vs 连续窗口在线
暴露消融，若单窗口仍不安全则停止 learned router，若仅连续窗口失败才重新讨论最小化 state
aggregation。

项目主线随后收束为 Stage-Aware Hidden Controller。停止把 hidden 直接用于短窗口 final-flip
精确预测，改为先验证 hidden 能否稳定识别 reasoning stage。Phase S1 已实现：从正式 Motivation
数据按 problem/segment 去重，比较 position、uncertainty、hidden-PCA 与 hidden+uncertainty，
使用 problem-level train/validation/test、train-only 标准化/PCA、三 seed macro-F1/recall 裁决。
当前四类 operational taxonomy 的 setup、reasoning、verification、final recall 必须在每个 seed
均不低于 `0.70`。
S1 硬门槛还包括最佳 hidden macro-F1 跨 seed 标准差不超过 `0.05`，以及至少 100 条人工 stage
审计且规则标签一致率不低于 `0.80`。
执行与准入条件见 [`rasp_stage_aware_mainline_zh.md`](rasp_stage_aware_mainline_zh.md)。S1 未通过前
不采集 S2 bank、不实现 S3 controller。

S1 的 100 条分层人工审计已完成：规则伪标签总体一致率仅 `61%`，其中 planning 为 `20%`、
verification 为 `30%`，未通过 `80%` 标签质量门槛。主要原因是顺序词与
`therefore/check` 关键词把普通 derivation 大量误标。当前停止使用这批规则标签训练或解释
stage probe。已将不可稳定区分的 planning/derivation 合并为 operational `reasoning`，形成
`setup/reasoning/verification/final` 四类 taxonomy，并隔离到
`runs/07_stage_aware/02_s1_operational_stage_probe/`。旧审计仅作为开发诊断；必须重新生成并
审核独立样本，训练脚本会在 100 条审计与 `80%` 一致率 gate 未通过时直接退出。

四类 taxonomy 的第二批 100 条独立审计已完成并通过：总体一致率 `86%`，setup/reasoning/
verification/final 分别为 `96.4%/62.5%/90.9%/100%`。主要残余错误为规则将只列条件或定义变量的
setup 片段误标为 reasoning。当前允许启动四类 S1 probe 三 seed 训练，但 held-out confusion
matrix 必须重点检查 `setup -> reasoning`；S1 通过前仍不采集 S2 bank。
由于 `runs/` 不受 Git 管理，人工标签已同步保存为
`configs/stage_audits/s1_operational_v2_labels.csv`；训练脚本会自动应用标签并验证，服务器不需要
手工再次填写 CSV。

四类 S1 首轮训练已完成但审查后作废。全量 verification 仅 11 条；旧 problem split 还将
validation/test 分布切坏：validation 没有 final，test 没有 verification 且约 `98.7%` 为
reasoning。因此 checkpoint 选择和 held-out 结果均无效，不能据此宣称 hidden 通过。split 已改为
normalized stage deficit，并新增每类最少 100 行与每 split 全 stage 覆盖硬 gate。下一步将
verification 改为显式规则触发的 dense override，hidden probe 仅学习样本充足的
setup/reasoning/final；现有四类 checkpoint 全部作废，仍不进入 S2。

S1 v3 三类 learned-stage 链路已实现，默认输出到
`runs/07_stage_aware/03_s1_three_stage_probe/`。数据准备会排除显式 verification 并写入
`01_verification_dense_overrides.jsonl`，修复后的 split 强制三个 split 覆盖
setup/reasoning/final。正式 gate 新增每个 seed 的 `setup -> reasoning <= 10%`，防止 controller
在 setup 阶段过早剪枝。下一步只生成并审核 v3 的新审计 CSV；审核标签同步前不启动训练。

S1 v3 的 100 条独立审计已完成，总体一致率 `85%` 并通过标签 gate；final/reasoning/setup
规则一致率分别为 `100%/77.8%/78.1%`。人工标签已同步为
`configs/stage_audits/s1_three_stage_v3_labels.csv`，现在允许启动三 seed 训练。由于仍存在 8 条
真实 setup 被规则标作 reasoning，最终模型必须通过 `setup -> reasoning <= 10%` 安全 gate。

S1 v3 三 seed 五组模型训练已完成，但 held-out eval 尚未运行。validation 上
hidden-pca-nonlinear macro-F1 为 `0.7551 ± 0.0181`，显著高于 uncertainty-only 的
`0.3630 ± 0.0061`，证明 hidden 含有稳定 stage 信息；平均 recall 为
setup/reasoning/final=`0.700/0.807/0.982`。当前阻断是 argmax 下 setup→reasoning 为
`0.300`、最差 seed `0.402`，未满足 `0.10` 安全线。下一步先运行 held-out eval，再按 controller
设计评估 confidence-gated reasoning acceptance；必须同时满足安全错误率不超过 `10%` 和非零
reasoning coverage，才能进入 S2。

S1 v3 held-out 正式汇总已完成，`s1_passed=false`。最佳 hidden-pca-nonlinear macro-F1 为
`0.7562 ± 0.0034`，相对最佳简单 baseline 增益 `+0.3952`，证明 hidden 的 stage 信息强且稳定；
reasoning/final recall 最差 seed 为 `0.808/0.972`。失败集中在 setup 安全边界：setup recall 最差
seed `0.681`，setup→reasoning 在三个 seed 均约 `0.312–0.319`，高于 `0.10` 上限。线性 hidden
虽更保守，但 reasoning recall 仅 `0.607–0.671`。当前不进入 S2；下一步只做 confidence-gated
selective reasoning acceptance，要求 setup false-accept 不超过 `10%` 且 reasoning coverage
非零，不再修改 taxonomy 或盲目重训。

S1.5 selective acceptance 与全阶段 S2 smoke 已完成代码实现。S1.5 使用 validation-only 阈值，
正式 gate 要求三个 seed 的 test setup false-accept 均不超过 `10%`，且 reasoning coverage 均至少
`10%`。执行 `bash scripts/63_eval_rasp_stage_selective.sh` 后查看
`runs/07_stage_aware/03_s1_three_stage_probe/s1_5_gate.json`。需要区分：
`s2_diagnostic_allowed` 只要求存在 validation-eligible probe，用于全阶段 S2 测量；
`s3_controller_allowed` 才要求严格 S1.5 test gate。S1.5 未通过时不允许进入 S3，但不能因此阻止
本来用于回答“哪些阶段真实安全”的 S2 diagnostic bank。

S2 不再假定 setup/final/verification 必须 dense。它会对 hidden probe 标注的
setup/reasoning/final 和显式规则标注的 verification 全部施加相同的单个 16-token runtime MLP
剪枝窗口，ratio 为 `0/0.05/0.10/0.20`，随后恢复 dense。每个 boundary 的 stage 在动作前固定，
不会因某个 ratio 的结果改变。服务器使用四张卡执行：

```bash
RASP_S2_GPU_COUNT=4 bash scripts/66_collect_rasp_s2_stage_sensitivity.sh
python scripts/67_summarize_rasp_s2_stage_sensitivity.py
```

结果位于 `runs/07_stage_aware/04_s2_stage_sensitivity_smoke/`。smoke summary 只用于判断是否扩大
正式 bank；最终是否允许某阶段剪枝必须由足量 held-out paired flip 结果决定。

S2-v1 smoke 已完成，10/10 shard validator 均通过，449 个 boundary 的 dense paired flip 与
dense replay flip 均为 0；reasoning 在 ratio 0.05 下粗粒度 flip 为 `5/375=1.33%`。但该轮发现
stage-position 对齐错误：S1 使用相对完整轨迹的 segment position，S2-v1 却使用
`generated_tokens/max_new_tokens`，且只采前 12 个 boundary，造成
setup/reasoning/final/verification=`50/375/21/3` 的偏斜。故 v1 counterfactual 有效，但 stage
分组结论作废。代码已改为 `generated_tokens/(dense_trajectory_tokens-1)` 并默认覆盖完整轨迹；
下一轮隔离输出到 `runs/07_stage_aware/05_s2_stage_sensitivity_v2/`。

## 5. 建议优先阅读

### 产物目录约定

`runs/` 与 `logs/` 已按实验阶段整理，完整约定见
[`experiment_artifact_layout_zh.md`](experiment_artifact_layout_zh.md)。当前主线位置：

```text
runs/05_phase_b/01_aligned_banks/   # aligned banks
runs/05_phase_b/02_phase_b2/        # Phase B2/v2/v3
runs/05_phase_b/03_phase_b25/       # Phase B2.5/B2.5b
runs/06_phase_b3_online/             # 当前 paired online
runs/07_stage_aware/                 # Stage-aware hidden 主线
logs/05_phase_b/
logs/06_phase_b3_online/
logs/07_stage_aware/
```

服务器同步旧格式结果或执行 `git pull` 后，先运行：

```bash
bash scripts/56_organize_experiment_artifacts.sh
```

该脚本只移动旧顶层产物，不删除或覆盖已有结果，并可重复执行。

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

## 7. 当前下一步：Action-Risk Pilot

八卡并行 Action-Risk Pilot 已完成代码实现：

```text
GPU 0–3: GSM8K train + isolated math_train 离线单窗口 action bank
GPU 4–7: GSM8K test + MATH500 固定单窗口在线诊断
```

离线准入按每来源至少 100 个 `dense-correct problem` 计算，不按输入题数计算；七档 ratio 为
`0/0.05/0.10/0.20/0.30/0.40/0.50`，每题最多 12 个边界均匀覆盖完整轨迹。在线仅在
`32/96/160` token 边界执行一个 16-token 窗口，之后恢复 dense。新链路支持显式 `GPU_IDS`，
结果隔离到 `runs/07_stage_aware/06_action_risk_pilot/` 与
`runs/07_stage_aware/07_online_fixed_window_pilot/`。执行步骤见
[`rasp_stage_aware_mainline_zh.md`](rasp_stage_aware_mainline_zh.md) 第 13 节。

注意：当前 S1 stage probability 使用完整 dense trajectory 的相对位置，不能作为在线因果输入；
新 pilot 明确不使用该字段。100 dense-correct/source 只用于路线判断，不足以证明最终
accuracy loss `<=1%`。

GPU 4–7 的 fixed-single-window 在线 pilot 已完成并通过产物/接线审查，但固定动作安全准入失败。
GSM8K 最接近安全线的是 `boundary=96, ratio=0.10`：净 accuracy delta 为 `0`，但实际执行动作的
dense-correct flip 仍为 `1/80=1.25%`，95% CI 上界约 `3.95%`。MATH500 最安全 point estimate
仍为 `boundary=160, ratio=0.10` 的 `4/57=7.02%` 条件 flip 与 `-3%` accuracy delta。

这不否定离线 Action-Risk bank：事后 oracle 显示两个数据集在可执行状态上的平均最大安全 ratio
都约为 `0.47`，说明高 ratio 并非普遍不安全，而是必须识别脆弱状态。该 oracle 使用结果泄漏，
不能作为方法结果。当前应等待 GPU 0–3 完成，并以 5-fold OOF 判断 hidden/context/action 模型
能否稳定预测风险；在此之前不运行连续固定剪枝，也不宣称在线 controller 成功。

GPU 0–3 的离线 Action-Risk pilot 也已完成。34/34 shard validator 通过，共有 `278` 个
dense-correct problem、`2965` 个 boundary、`17790` 个 action row。整体 flip 随 ratio 从
`2.23% @ 0.05` 单调升至 `12.75% @ 0.50`，说明监督信号有效。

严格 hidden gate 未通过：context+action 相对 action-only 在 4/5 folds 稳定提升，但 hidden 的
增益主要体现在 PR-AUC，ROC 与跨数据集提升不稳定。当前正确裁决不是停止 Action-Risk，而是将
`causal context/action` 作为主风险模型，hidden 只作为风险 veto 消融。下一步允许训练最终
checkpoint 并运行每题最多一个窗口的 learned-controller pilot；只有它在线优于 fixed baseline
后才扩正式 bank 或讨论连续窗口。

## 8. 当前下一步：Learned Action-Risk Single-Window Pilot

已实现新的 `action_risk_single_window` controller。它不复用旧 Phase-B2/uncertainty router，
只在 `32/96/160` token 边界使用 causal context/action 风险模型判断；首次选择非零 action 后
执行一个 16-token MLP-channel 剪枝窗口，随后永久恢复 dense。每题最多一个窗口。

最终 checkpoint 使用现有 problem-level 5-fold OOF 预测校准三档 operating point：
`conservative / balanced / aggressive`。Context-only 是主策略；hidden 只能 veto context 已接受
的 action，不能提高 ratio。每个 hidden-veto 档位必须在 OOF 上保留至少 80% context exposure，
且两个训练数据集的 flip 均不恶化，否则该档不会生成在线任务。

服务器执行顺序：

```bash
PYTHON=/home/cike/jjy/envs/rasp_qwen3/bin/python \
bash scripts/75_train_action_risk_controller.sh

mkdir -p logs/07_stage_aware/08_action_risk_learned_single_window_pilot
nohup env GPU_IDS=0,1,2,3,4,5,6,7 \
PYTHON=/home/cike/jjy/envs/rasp_qwen3/bin/python \
bash scripts/77_eval_action_risk_learned_pilot.sh \
> logs/07_stage_aware/08_action_risk_learned_single_window_pilot/launcher.log 2>&1 &

PYTHON=/home/cike/jjy/envs/rasp_qwen3/bin/python \
python scripts/78_summarize_action_risk_learned_pilot.py
```

结果位于 `runs/07_stage_aware/08_action_risk_learned_single_window_pilot/`。本轮只有 learned
context-only 在 GSM8K/MATH500 均满足 pilot 门槛并优于相近理论 exposure 的 fixed baseline，
才允许扩充 bank、运行多 seed 或双窗口实验。

## 9. 当前下一步：Stage-Action-Risk v2 精确边界 Bank

Learned single-window pilot 已完成但未通过：所有策略题都在 token `32` 立即执行，说明该轮只学会
了“在 32 token 剪多少”，没有学会 reasoning-aware 时机选择。根因是旧 uniform bank 并未为每题
完整采集 `32/96/160`：离线校准中的晚边界选择经常只是因为早期边界不存在。

下一轮先修复数据，不提前训练新 controller。Stage-Action-Risk v2 强制采集：

```text
每个 retained dense-correct problem:
boundary = 32 / 96 / 160 全部存在
ratio    = 0 / 0.05 / 0.10 / 0.20 / 0.30 / 0.40 / 0.50
每个动作持续 16 token，随后恢复 dense
动作前记录 causal hidden-stage probability
```

缺任一边界或完整 action grid 的问题会整题剔除。v2 仅用于判断 stage 表征是否真正改善风险与时机
建模；当前 stage selective gate 未通过，因此 bank 标记为 diagnostic-only。

```bash
# 八卡采集
GPU_IDS=0,1,2,3,4,5,6,7 bash scripts/80_collect_stage_action_risk_v2.sh

# worker 全部完成后
PYTHON=/home/cike/jjy/envs/rasp_qwen3/bin/python \
bash scripts/81_prepare_stage_action_risk_v2_data.sh

PYTHON=/home/cike/jjy/envs/rasp_qwen3/bin/python \
bash scripts/82_analyze_stage_action_risk_v2.sh
```

只有 `analysis/02_stage_action_risk_analysis.json` 中
`stage_controller_training_allowed=true`，才实施 stage-gated waiting controller。

Stage-Action-Risk v2 已完成采集与分析。54/54 shard validation 为 `ok`，最终保留
`394` 个完整 dense-correct problem，其中 GSM8K `158` 个、math_train `236` 个；每题均包含
`32/96/160` 三个精确边界，共 `1182` 个 boundary 和 `7092` 个非零 action row。

本轮数据修复成功，但 stage gate 未通过：

```text
causal context + action       ROC 0.6680 / PR 0.1273
stage + context + action      ROC 0.6658 / PR 0.1236
hidden + stage + context      ROC 0.6703 / PR 0.1383
stage 同时胜过 context 的 fold 数：1/5
stage_controller_training_allowed=false
```

Stage policy 虽在相近 exposure 下将 math_train problem flip 从 `4.66%` 降至 `4.24%`，
但 GSM8K 不改善，且 OOF 排序指标不稳定。更重要的是，当前 first-accepted 策略只要 token `32`
存在任一低风险 ratio 就立即执行：context policy 的 `394/394` 题、stage policy 的 `391/394`
题仍在 token `32` 动作。因此精确边界 bank 修复了旧数据缺失问题，但当前目标函数仍未真正训练
“等待的价值”。

下一步不得直接训练 stage-gated controller。应先在 v2 paired exact-boundary bank 上建立显式
`act-now vs wait` 诊断：比较当前安全动作效用与未来边界可获得的安全动作效用，并使用当前因果
hidden/stage/context 预测等待是否有价值。只有该 timing-value 任务在 problem-level OOF 上稳定
优于固定早期动作，才实现新的 waiting controller。

## 10. Full-Trajectory Multi-Window 自动工作流

为避免继续被 `32/96/160` 三个离散边界和单窗口假设限制，已实现新的集成工作流：

```text
已有 Action-Risk bank CPU 预审
→ 4 dense-correct/source causal-grid smoke
→ 20 dense-correct/source full-trajectory pilot + grouped OOF
→ 隔离训练来源上的 fixed multi-window dev
→ 精确重放完整历史的 on-policy bank smoke
→ 最终 gate 与中文报告
```

自然决策边界从 token `32` 开始、每 `32` token 一次；每个动作持续 16 个 affected-token
decision，之后至少保持 16 token dense cooldown。`tail_anchor` 仅用于临近 EOS 的诊断，不进入
可部署模型。Stage 只作为 causal soft probability；低置信 hard stage 归为 `unknown`，依赖
完整轨迹位置的 checkpoint 会硬失败。

Smoke/Pilot 会先适度过采样输入，再分别固定选取每来源 `4/20` 个 full-window eligible
dense-correct problem，避免“20 道输入题”被误当作“20 个有效问题”。总控开始时还会先运行现有
单元测试；任一测试失败不会启动 GPU 采集。

固定多窗口 dev 比较 `r0.10/r0.20/r0.30` 的不同 cadence/max-window 组合，并仅选择在 GSM8K
train 与隔离 math_train 上同时通过风险—曝光方向门槛的 behavior policy。On-policy smoke
不会拼接 dense prefix，而会重放此前每个动作及 dense 恢复，并核对 token prefix、boundary
完整 logits hash/top-k、hidden、动作历史和 cooldown。

唯一入口：

```bash
mkdir -p logs/07_stage_aware/10_full_trajectory_multi_window
nohup env GPU_IDS=0,1,2,3,4,5,6,7 \
PYTHON=/home/cike/jjy/envs/rasp_qwen3/bin/python \
bash scripts/90_run_full_trajectory_multi_window_workflow.sh \
> logs/07_stage_aware/10_full_trajectory_multi_window/launcher.log 2>&1 &
```

结果统一写入 `runs/07_stage_aware/10_full_trajectory_multi_window/`。每个阶段有独立 gate；失败后
总控立即停止后续 GPU 工作，并仍生成 `workflow_gate.json`、`final_workflow_summary.json` 与
`final_workflow_report_zh.md`。本轮最终状态固定为
`learned_multi_window_allowed=false`；只有扩大 on-policy bank 并通过 problem-grouped OOF 后
才允许训练 learned multi-window controller。

首次八卡 smoke 在 `math_train_s04` 停止：该 shard 的两道输入均未被 dense 正确回答，因此
合法地产生 `0` 个可采集边界，但旧采集器将其误判为致命错误。现已修复为空 shard 可验证语义：
只有当 source trajectory 确实不产生任何预期边界时，validator 才接受空 shard；每来源
`4/20` 个有效 dense-correct problem 的总量要求仍由阶段 gate 严格检查。断点重跑会跳过已经
通过 validation 的非空 shard。

本次 smoke 同时观测到 math_train 的 dense-correct 率约为 `7/12`。因此 Pilot 的默认
math_train 输入过采样量已由 `32` 提高到 `48`；最终仍只固定保留 `20` 个 eligible
dense-correct problem，不改变两来源的分析样本量，只降低长时间采集后因有效题不足而失败的概率。

检查部分 smoke 产物时还发现，旧 counterfactual 实现把“续写最终以 EOS 结束”误记为
`action_terminal_eos`，导致完整 16-token 窗口也被标成 terminal。该字段现已严格改为“EOS 在
动作窗口完成前发生”，并单独记录 `continuation_ended_with_eos`。断点校验会拒绝旧 terminal
语义，因此首次 smoke 的 9 个已完成 shard 需要重采；旧产物的 final-answer flip 和 dense replay
检查仍可用于诊断，但其 terminal-EOS 统计不得使用。

Full-Trajectory Multi-Window 工作流现已端到端通过：40 个 Pilot 问题产生 `332` 个 causal
boundary 和 `1660` 个 action rows，ratio 风险从 `0.10` 的 `7.23%` 单调上升到 `0.50` 的
`18.07%`。固定多窗口 dev 唯一通过两来源方向 Gate 的策略为 `r020_c32_m4`，on-policy smoke
两来源均完成 4 个有效问题、8 个 prior-action boundary，精确 replay failure 为 `0`，因此允许
扩大 on-policy bank，但尚不允许训练 learned multi-window controller。

On-policy smoke 还暴露了标签语义问题：原 `candidate_flipped_from_on_policy_dense_control`
只表示答案变化，会把“错误 dense-control 被剪枝动作修正”也算作风险。本轮 16 个 boundary 中
正好存在 1 个这种情况。现已增加 `candidate_harmful_flip` 与
`candidate_beneficial_correction`，后续风险训练必须使用前者；旧汇总的 `flip_rate` 只能解释为
answer-change rate。

另一个扩大前必须消除的偏差是旧 smoke 只采集 dense 与 behavior 都正确的问题，这会排除 fixed
behavior 已造成错误的轨迹，形成幸存者偏差。新配置改为必须 `dense-correct`、但允许
`behavior-incorrect`，并显式记录两者 correctness。这样 on-policy bank 才包含需要 Controller
学会规避的失败状态；behavior correctness 不再作为进入 bank 的过滤条件。
