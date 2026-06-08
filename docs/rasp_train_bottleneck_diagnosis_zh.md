# RASP-Train 当前瓶颈诊断与解决路线

## 1. 结论

当前主要瓶颈不是网络容量或 threshold，而是训练目标与真实在线动作尚未完全对齐。

RASP-Train v1、v2、v2.1 已分别排除了单标签 imitation、预算伪相关和不稳定 calibration 等问题，
但现有结果仍不能证明 nonlinear policy 弱于 RASP-Zero，因为当前比较还不是同 split、同标签、
同动作语义的公平对照。

在修复数据和评估协议前，不应继续扩大模型、调整 loss 或运行在线 smoke。

## 2. 已确认的四个上游瓶颈

### 2.1 Action horizon 不一致

当前 counterfactual bank 在 segment 起点施加 ratio 后，让该 ratio 持续作用到剩余生成结束，
再用最终答案是否翻转作为标签：

```text
bank label = P(final answer flip | ratio 从 segment 起点永久作用)
```

线上 controller 每 `16` token 重新决策，单个 action 只作用一个窗口：

```text
online target = P(不可恢复损害 | ratio 仅作用下一个 16-token window)
```

永久干预标签会系统性高估短窗口动作风险，是 v2.1 过度保守的首要候选原因。

### 2.2 Neuron ranking 语义不一致

Bank 对每个 segment 使用“题目 + 当前 dense reasoning prefix”重新 dense prefill，并刷新 neuron
ranking。在线 runtime 只在最初题目 prefill 时建立 ranking，后续窗口不刷新。

因此同一个 ratio 在 bank 和 runtime 中可能剪掉不同 neuron，动作本身并不相同。

### 2.3 State distribution 不一致

Bank state 来自 dense trajectory 的 heuristic segment start；线上 state 来自 fixed-token window，
且可能已经受此前剪枝动作影响。

当前训练没有覆盖 policy-induced states，存在标准 offline-policy covariate shift。

### 2.4 RASP-Zero 对照不公平

现有 RASP-Zero 使用 raw `flipped` 标签和独立 75/25 split；v2.1 使用 monotonic
`candidate_unsafe` 标签和 70/15/15 split。

更严重的是，v2.1 的 120 个 test problems 中有 87 个被现有 RASP-Zero checkpoint 用于训练：

```text
v2.1 test overlap with RASP-Zero train = 87 / 120 = 72.5%
```

因此当前表格只能说明“v2.1 未超过这个现有 checkpoint”，不能作为严格方法优劣结论。

## 3. 其他已量化问题

### 3.1 非单调与标签保守化

```text
problem steps = 3479
non-monotonic steps = 407
non-monotonic rate = 11.7%
```

对非零动作：

```text
raw flip positive rate       = 19.63%
monotonic unsafe positive rate = 22.51%
额外保守标签                  = 2.88 percentage points
```

建议训练 raw flip risk，并在 controller 中对预测风险做 ratio 方向 cumulative-max，而不是修改监督
标签。这样可保留真实观测，同时维持保守单调选择。

### 3.2 数据规模与模型容量

当前有效训练规模约为：

```text
561 train problems
2444 train states
```

v2.1 nonlinear policy 约有几十万参数，而 linear probe 只有几千参数。当前属于明显的小样本场景；
在数据协议未修复前，增加模型容量更可能过拟合。

### 3.3 风险高度依赖推理位置

非零动作 raw flip rate：

```text
position 0.00-0.25: 37.9%
position 0.25-0.50: 26.8%
position 0.50-0.75: 14.4%
position 0.75-1.00:  4.2%
```

这支持 reasoning-aware motivation，但也意味着模型可能主要学习粗粒度 position，而没有真正学会
同一阶段内的细粒度 state sensitivity。必须加入 position-only baseline 验证增益来源。

## 4. 正确的解决顺序

### Phase A：先建立公平离线 benchmark

使用同一份 problem split manifest，让以下模型使用完全相同的 train/calibration/test：

```text
ratio-only
position + ratio
entropy/confidence + ratio
hidden + ratio linear probe
hidden + ratio nonlinear probe
```

分别训练两种标签：

```text
raw candidate_flipped
monotonic candidate_unsafe
```

所有模型使用相同 causal controller、相同 threshold calibration 和相同 test frontier。报告
problem-level bootstrap confidence interval，不再只比较一个 threshold。

这一阶段可以使用现有 bank，成本低，目的是判断 nonlinear 模型是否真的提供额外排序能力。

### Phase B：重建真正 deployment-aligned window bank

新 bank 必须满足：

```text
state boundary = 在线 fixed 16-token boundary
action duration = 仅下一个 16-token window
ranking semantics = 与 runtime 完全一致
after window = 回到 dense，再继续到最终答案
label = 单窗口干预是否最终导致答案 flip
```

同时记录辅助标签：

```text
窗口内 token divergence
窗口末 hidden-state drift
最终答案 flip
```

最终答案 flip 负责真实性，短期 drift 提供更密集的 credit assignment。

#### Phase B1 实现状态

Phase B1 aligned window bank 采集代码已完成：

```text
src/main_collect_aligned_window_bank.py
src/main_validate_aligned_window_bank.py
scripts/43_prepare_rasp_phase_b_aligned_bank_configs.py
scripts/44_collect_rasp_phase_b_aligned_bank.sh
```

实现语义：

1. 从原始 prompt 做一次 dense prefill，建立与 runtime 相同的固定 neuron ranking；
2. dense 强制重放 baseline assistant token 到固定 `16-token` boundary；
3. candidate ratio 仅作用下一个窗口；
4. 窗口后恢复 ratio `0`，继续生成到答案结束；
5. paired ratio-0 continuation 作为 dense control；
6. 记录 `window_token_divergence`、窗口末 hidden L2/cosine drift 和 paired final-answer flip。

Phase B 配置要求 dense trajectory 生成入口直接保存并重放模型原始 `generated_token_ids`，避免
decode/re-tokenize 改变 token boundary。采集器虽然保留文本重新 tokenize 的兼容回退，但正式
validation 会拒绝这种 fallback 数据。

Phase B bank 采集成本显著高于旧 bank，应先运行小规模 smoke：

```bash
export PYTHON=/home/cike/jjy/envs/rasp_qwen3/bin/python
export RASP_PHASE_B_LIMIT_PER_SOURCE=25
export RASP_PHASE_B_SHARD_SIZE=5
export RASP_PHASE_B_GPU_COUNT=8
bash scripts/44_collect_rasp_phase_b_aligned_bank.sh
```

确认所有 shard validation 为 `status=ok`、ratio-0 control 稳定且局部 drift 字段有效后，再将
`RASP_PHASE_B_LIMIT_PER_SOURCE` 提升到 `500` 正式采集。

#### Phase B1 Smoke 结果

每个数据源 25 题的 smoke 已完成。10/10 个 shard validation 均为 `status=ok`：

```text
dense-correct problems = 38 / 50
fixed-window boundaries = 228
counterfactual rows = 1596
dense replay flip rate = 0
所有 action duration = 16 tokens
所有 boundary 使用原始 generated_token_ids
```

非零 ratio 的 paired final-answer flip rate 为 `48 / 1368 = 3.51%`，明显低于旧永久干预 bank，
符合短窗口动作风险更低的预期。按 ratio 汇总：

| Ratio | Flip | Window token divergence | Hidden L2 drift |
|---:|---:|---:|---:|
| 0.02 | 0.0088 | 0.0779 | 24.99 |
| 0.05 | 0.0175 | 0.1058 | 35.45 |
| 0.10 | 0.0175 | 0.1395 | 46.82 |
| 0.20 | 0.0395 | 0.2385 | 71.38 |
| 0.30 | 0.0658 | 0.3353 | 94.12 |
| 0.40 | 0.0614 | 0.3983 | 109.76 |

局部 divergence 与 hidden drift 随 ratio 清晰增长，说明 aligned action 生效且辅助标签有信息量。
但最终 flip 标签稀疏并存在约 `8.3%` boundary-level 非单调，因此 Phase B2 应采用 raw flip +
连续 drift 多任务目标，而不能只训练最终 flip BCE。

Smoke 只覆盖每题前 6 个窗口，即前 96 tokens。正式采集必须显式决定覆盖范围：

```bash
# 建议先覆盖前 12 个窗口，即 192 tokens；设为 0 表示覆盖完整 dense trajectory。
export RASP_PHASE_B_MAX_BOUNDARIES_PER_EXAMPLE=12
```

由于正式 aligned bank 成本远高于旧 bank，建议先以每数据源 `100` 题、12 个窗口进行中型采集，
检查后半程标签和运行成本，再扩大到 `500` 题。

#### Phase B1 中型采集结果与配置修复

每数据源 100 题的中型采集已经完成，20/20 shard 严格 validation 均通过：

```text
dense trajectories = 200
dense-correct problems = 164
fixed-window boundaries = 984
counterfactual rows = 6888
dense replay flip rate = 0
nonzero-action positives = 236 / 5904 = 4.00%
positive problems = 60
positive boundaries = 138
```

相比 smoke，正例率从 `3.51%` 上升到 `4.00%`，ratio 与 token divergence/hidden L2 drift 的
Pearson correlation 分别为 `0.375/0.436`。但 drift 与最终 flip 的线性相关都只有约 `0.11`，
因此 drift 适合作为辅助多任务目标，不能直接替代 final flip 标签。

本次中型采集实际仍只覆盖 boundary `0-5`，即前 96 tokens，没有按计划覆盖 12 个窗口。数据长度
不是原因：全部 164 个 dense-correct trajectory 均超过 96 tokens，其中 140 个超过 192 tokens。
这是旧断点续跑逻辑只检查 `status=ok`、未检查当前配置是否改变导致的。

该问题已经修复：

- 默认 `RASP_PHASE_B_MAX_BOUNDARIES_PER_EXAMPLE` 从 `6` 改为 `12`；
- validation 记录 configured window/max-boundary 参数；
- worker 仅在已有 validation 与当前配置完全匹配时跳过；
- 配置变化时复用已有 dense trajectories，仅重采 counterfactual bank。

因此下一步应在当前 100 题数据上重采 12-window bank，不需要重新生成 dense trajectories：

```bash
export PYTHON=/home/cike/jjy/envs/rasp_qwen3/bin/python
export RASP_PHASE_B_LIMIT_PER_SOURCE=100
export RASP_PHASE_B_SHARD_SIZE=10
export RASP_PHASE_B_GPU_COUNT=8
export RASP_PHASE_B_MAX_BOUNDARIES_PER_EXAMPLE=12
bash scripts/44_collect_rasp_phase_b_aligned_bank.sh
```

新 worker 会输出 `RECOLLECT config changed` 并复用 `01_trajectories.jsonl`。12-window 重采完成后，
再决定是否扩到 500 题以及正式进入 Phase B2。多任务训练代码可以并行开发，但当前前 96-token
bank 只用于诊断，不能作为正式 Phase B2 方法结论。

### Phase C：覆盖 policy-induced states

第一轮 window bank 来自 dense trajectory。训练初版 policy 后，再从 policy rollout 中采集状态并加入
bank，执行一到两轮 DAgger-style 数据聚合。

否则模型只会在 dense state 上可靠，无法处理此前剪枝导致的状态漂移。

### Phase D：再选择模型结构

在 aligned bank 上按以下顺序尝试：

1. raw-flip linear action-risk probe；
2. raw-flip nonlinear probe；
3. RASP-Zero score + learned residual；
4. 多任务预测短期 drift 与最终 flip；
5. per-ratio 或 conformal risk calibration。

先从 raw flip 学习，再在 controller 中使用：

```text
conservative_risk(r_i) = max predicted_flip_risk(r_j), for r_j <= r_i
```

不再通过修改标签强制单调。

## 5. 验收与停止条件

离线方法只有同时满足以下条件才进入在线：

```text
同 split、同标签、同 controller 下公平比较
test problems 从未进入任何 comparator 训练
在 problem-level 95% CI 下形成相对 RASP-Zero 的 Pareto 优势
不是通过大幅降低 average ratio 换取安全性
跨至少 3 个训练 seed 结果稳定
```

若 aligned window bank 上 linear/nonlinear/residual 都无法超过 position-only 或 RASP-Zero，则应接受：

> 当前 hidden state 对短窗口安全动作的可预测信号不足，RASP-Train 不应作为主线贡献。

这时主线应回到更可解释的 conservative RASP-Zero，或转向更强的结构信号和真实 reduced-weight
backend，而不是继续堆叠 policy 网络。

## 6. 当前是否考虑全面

目前已经覆盖：

- 数据泄漏与 split 公平性；
- raw flip / monotonic unsafe 标签差异；
- action horizon；
- neuron ranking 语义；
- dense-state / policy-state covariate shift；
- segment boundary / fixed-window mismatch；
- 小样本与模型容量；
- calibration 泛化；
- logical mask 与真实加速边界。

仍需通过实验确认，而不能仅靠分析断言：

- 短窗口最终 flip 标签是否足够稳定；
- 短期 hidden drift 是否能改善 credit assignment；
- ranking 应固定、按窗口刷新，还是由轻量历史统计更新；
- aligned bank 的采集成本是否可接受；
- 改成真实 reduced-weight backend 后策略行为是否保持一致。
