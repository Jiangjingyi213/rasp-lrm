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

#### Phase B1 12-Window Bank 结果

12-window 重采已完成，当前结果暂存于 `runs/未命名/`。20/20 shard 均通过严格 validation：

```text
configured window tokens = 16
configured max boundaries = 12
dense trajectories = 200
dense-correct problems = 164
fixed-window boundaries = 1926
counterfactual rows = 13482
nonzero-action rows = 11556
nonzero final-flip positives = 561 (4.85%)
dense replay flip rate = 0
```

按 ratio 汇总：

| Ratio | Final flip | Token divergence | Hidden L2 drift |
|---:|---:|---:|---:|
| 0.02 | 0.0125 | 0.0525 | 20.52 |
| 0.05 | 0.0228 | 0.0891 | 33.79 |
| 0.10 | 0.0332 | 0.1311 | 48.78 |
| 0.20 | 0.0514 | 0.2223 | 71.56 |
| 0.30 | 0.0774 | 0.2937 | 89.44 |
| 0.40 | 0.0940 | 0.3617 | 105.77 |

后 6 个窗口提供了重要的新风险信号。非零动作 final-flip rate 从前 6 窗口的 `4.00%` 上升至
后 6 窗口的 `5.75%`；但后半程平均 token divergence 反而更低。这说明局部 drift 大小不能直接
替代最终风险，policy 必须结合 boundary state、位置与 action 预测 final flip。

当前共有：

```text
positive problems = 108 / 164
positive boundaries = 299 / 1926
boundary-level non-monotonic flip = 142 / 1926 (7.37%)
```

Final flip 与 token divergence/hidden L2 的相关系数仅为约 `0.114/0.128`。因此 Phase B2 应把
drift 作为辅助目标，而不是安全标签本身；主标签继续使用 raw paired final flip，并由 controller
对 ratio 风险做 cumulative-max。

当前数据已足够进入 Phase B2 多任务原型训练。由于只有 164 个问题，不足以作为最终论文规模，
训练评估必须使用 dataset/positive-rate 分层 problem split、至少三个 seed 和 out-of-fold
calibration；原型有效后再扩大至 500 题。

另有 `25/1926` 个 boundary 在完整 16-token action window 结束前遇到 EOS，对应 149 条
`action_duration_tokens < 16` 的 rows。这不是采集错误，但暴露时长与其余 action 不同。Phase B2
首轮训练与公平评估应过滤这些 incomplete-window boundaries，并将其保留为单独的尾部行为分析。

### Phase B2：多任务原型实现

Phase B2 多任务训练代码已完成：

```text
src/rasp/phase_b2.py
src/main_prepare_rasp_phase_b2_data.py
src/main_train_rasp_phase_b2.py
src/main_eval_rasp_phase_b2.py
scripts/45_prepare_rasp_phase_b2_data.sh
scripts/46_train_rasp_phase_b2.sh
scripts/47_eval_rasp_phase_b2.sh
scripts/48_summarize_rasp_phase_b2.py
```

训练目标：

```text
L = weighted BCE(final paired flip)
  + lambda_div * SmoothL1(window token divergence)
  + lambda_hidden * SmoothL1(window-end hidden cosine distance)
```

共享 state-action encoder 输出三个 head。Final flip 是 controller 唯一使用的风险；drift 只作为训练
辅助信号。ratio=0 不进入 flip BCE 和类别权重计算，避免大量恒定安全 control 淹没正例。

默认比较：

```text
hidden_multitask
hidden_flip_only
uncertainty_multitask
```

数据准备会把每个 boundary 的 7 条 action rows 合并成一个 multi-action 样本，并整体过滤
incomplete-window boundary。Split 使用 dataset 与 problem 是否含正例的分层 70/15/15
problem-level manifest。Threshold calibration 只使用训练未见的 calibration problems，并要求
calibration 内各 problem fold 的最坏 flip rate 同时满足约束。

执行：

```bash
mv runs/未命名 runs/05_phase_b/01_aligned_banks/rasp_phase_b_aligned_bank_12w
export PYTHON=/home/cike/jjy/envs/rasp_qwen3/bin/python
bash scripts/45_prepare_rasp_phase_b2_data.sh
```

三个 seed 可分别占用三张 GPU：

```bash
mkdir -p logs/05_phase_b
for item in "0 1" "1 2" "2 3"; do
  set -- ${item}
  nohup env CUDA_VISIBLE_DEVICES="$1" PHASE_B2_SEEDS="$2" \
    bash scripts/46_train_rasp_phase_b2.sh \
    > "logs/05_phase_b/rasp_phase_b2_seed_$2.log" 2>&1 &
done
```

训练全部完成后：

```bash
bash scripts/47_eval_rasp_phase_b2.sh
```

输出汇总为 `runs/05_phase_b/02_phase_b2/rasp_phase_b2/comparison_summary.csv`。验收重点不是单一 AUC，而是：

1. `hidden_multitask` 是否稳定优于 `hidden_flip_only`；
2. hidden 是否稳定优于 uncertainty-only；
3. calibration constraints 是否跨三 seed 全部满足；
4. B15/B20 在保持低 flip 时能否获得有效 ratio，而不是退化到接近 dense。

### Phase B2 三 seed 结果

`runs/05_phase_b/02_phase_b2/rasp_phase_b2/` 已完整产生 9 个 checkpoint、9 个 train metrics 和 9 个 test eval。数据为
164 个问题、1901 个完整 16-token boundary、11406 个非零 action rows，其中 558 个 final-flip
正例。

| variant | ROC-AUC | PR-AUC | B15 ratio / flip | B20 ratio / flip |
|---|---:|---:|---:|---:|
| hidden-flip-only | 0.6099 | 0.0694 | 0.0923 / 0.0180 | 0.1647 / 0.0236 |
| hidden-multitask | 0.6202 | 0.0736 | 0.0826 / 0.0202 | 0.1659 / 0.0247 |
| uncertainty-multitask | 0.6450 | 0.0802 | 0.0808 / 0.0236 | 0.1195 / 0.0257 |

结论不是“多任务成功”，而是：

1. drift 辅助目标只给 hidden 风险排序带来小幅提升，没有稳定改善 controller；
2. uncertainty-only 的平均风险排序最好，尚无证据证明 hidden state 提供稳定额外信息；
3. hidden-flip-only 当前 controller trade-off 最好，但 seed 间方差过大。seed 2 在 B15/B20
   的 test flip 为 `0.0407/0.0441`，而 seed 1/3 明显更低；
4. 无风险限制、只耗尽 causal budget 的 test baseline，B15/B20 平均 flip 约为
   `0.0446/0.0513`；安全 oracle 几乎可在零 flip 下用满预算，说明 action 空间存在潜力，但当前
   policy 尚未学会稳定识别安全 action。

所有 calibration constraints 都显示满足，但这不表示 test constraints 满足。更重要的是，当前
训练循环每个 epoch 使用 calibration loss 选择 best checkpoint，之后又用同一 calibration split
选择 threshold，导致 calibration reuse。下一轮执行顺序必须是：

1. manifest 增加独立 validation split，validation 只用于 checkpoint selection；
2. calibration 只用于 threshold / risk constraint，test 始终只评估一次；
3. 增加 position-only、ratio-only、RASP-Zero residual 对照；
4. 无泄漏重跑三 seed 后，再决定是否扩大 bank 或进入 Phase C。

在完成上述修复前，不进入在线 rollout。若无泄漏实验中 hidden 仍不能稳定超过简单特征，应触发
停止条件，接受当前 hidden signal 不足。

### Phase B2 v2 无泄漏重跑实现

上述修复已经实现，默认输出目录改为 `runs/05_phase_b/02_phase_b2/rasp_phase_b2_v2/`，不会覆盖第一轮诊断结果：

- manifest schema 升级为 `rasp_phase_b2_multitask_v2`；
- split 改为 problem-level、dataset/positive 分层的 `60/10/15/15`
  train/validation/calibration/test；当前 164 题实际得到 `97/17/25/25`；
- validation 只负责 best-checkpoint selection；
- calibration 只负责 controller threshold 与 problem-fold constraint；
- test 只在训练和校准完成后评估；
- 新增 `ratio_only_flip_only`、`position_flip_only`、`uncertainty_flip_only`，用于判断 hidden 与
  drift 辅助目标是否真的超过简单特征。

aligned bank 当前没有保存可直接复用的 RASP-Zero score，因此本轮不伪造 residual 对照。若要做
RASP-Zero residual，必须先定义并保存与 aligned boundary 一致的 zero-score。

服务器运行：

```bash
export PYTHON=/home/cike/jjy/envs/rasp_qwen3/bin/python
bash scripts/45_prepare_rasp_phase_b2_data.sh

mkdir -p logs/05_phase_b
variants=(
  hidden_multitask hidden_flip_only
  uncertainty_multitask uncertainty_flip_only
  position_flip_only ratio_only_flip_only
)
for i in "${!variants[@]}"; do
  variant="${variants[$i]}"
  nohup env CUDA_VISIBLE_DEVICES="$i" PHASE_B2_VARIANTS="$variant" PHASE_B2_SEEDS="1 2 3" \
    bash scripts/46_train_rasp_phase_b2.sh \
    > "logs/05_phase_b/rasp_phase_b2_v2_${variant}.log" 2>&1 &
done
```

六个日志均完成后执行：

```bash
bash scripts/47_eval_rasp_phase_b2.sh
```

只有 `comparison_summary.csv` 中 `all_checkpoints_selected_on_validation=True`，且 hidden 在三 seed
上稳定超过 ratio/position/uncertainty 基线并形成 controller Pareto 增益，才进入 Phase C。

首次 v2 训练同步后、test eval 前发现 `position_flip_only` 与 `ratio_only_flip_only` 完全相同。
审计确认原因是 `PhaseB2MultiTaskNet` 对一维 state 使用 `LayerNorm(1)`，会把 position 恒定归零。
代码已改为一维 state 跳过 LayerNorm。必须重训这两个一维 variant，再执行 test eval；其余四个
variant checkpoint 不受该修复影响。

### Phase B2 v2 结果异常的根因审查

进一步端到端审查后，v2 不再作为最终裁决实验，原因分为任务变化与代码问题两类。

任务变化解释了大部分 AUC 落差：

- 旧永久干预 bank 的 nonzero raw-flip 正例率为 `19.63%`，`50.62%` boundary 至少含一个正例；
- aligned 16-token bank 的对应数值只有 `4.89%` 和 `15.57%`；
- 旧任务中一次 action 长期影响后续生成，更容易由当前状态预测；aligned 任务只干预一个短窗口，
  随后恢复 dense，最终答案是否 flip 更稀疏、更容易恢复，也更接近真实 credit assignment 难题；
- v2 hidden nonlinear 有约 52 万个输入投影参数，但只有 97 个训练问题，最佳 validation epoch
  通常为 1–2；uncertainty 模型最佳 epoch 约 33–40，符合 hidden 模型快速过拟合。

同时确认了四个实现问题：

1. single-window collector 的第一个生成 token 来自 action 前的 dense logits；真正受 action
   影响的是后续第 `2–17` 个 token，但旧 `window_ids` 记录的是第 `1–16` 个 token。因此 final
   flip 主标签仍反映 action 后果，但 token-divergence 辅助标签错位；
2. Phase B2 把 position 写成 `boundary_index / 11`，而 runtime 使用
   `generated_tokens / max_new_tokens`。以当前 `max_new_tokens=768` 为例，第 12 个 boundary
   实际 position 约为 `176/768=0.229`，旧数据却写成 `1.0`；
3. nonlinear encoder 对整段输入做 LayerNorm，既会让一维 position 恒为零，也会把 hidden、
   entropy、confidence、position 和 domain one-hot 混合归一化，导致 hidden 与简单特征对照
   难以解释；
4. 旧 manifest 所谓 positive-rate 分层实际只区分“是否含任意正例”。三个 seed 的 validation
   action-positive rate 在 `3.96%–7.48%` 之间，近乎翻倍，放大了 early stopping 与 calibration
   方差。

上述问题已按 Phase B2 v3 修复：

- collector 记录真正由 action 产生的后续 token decisions；
- row 与 validator 强制保存并核对 runtime-aligned position、max-new-tokens 和 window alignment；
- worker 断点检查会拒绝旧 alignment，避免错误 shard 被静默跳过；
- Phase B2 schema 升级为 `rasp_phase_b2_multitask_v3`，移除混合输入 LayerNorm；
- 增加 hidden/uncertainty/position/ratio-only 的 linear 与 nonlinear 对照，区分“hidden 无信号”
  与“小样本 nonlinear 过拟合”；
- split 与 calibration folds 改按 dataset 内 `zero / positive-low / positive-high` 风险负担分层，
  不再只按是否含正例二分。

因此下一步必须先重采 `runs/05_phase_b/01_aligned_banks/rasp_phase_b_aligned_bank_v2/`，再训练
`runs/05_phase_b/02_phase_b2/rasp_phase_b2_v3/`；不要继续评估或引用尚未完成的 v2 test。

服务器执行顺序：

```bash
export PYTHON=/home/cike/jjy/envs/rasp_qwen3/bin/python
export RASP_PHASE_B_LIMIT_PER_SOURCE=100
export RASP_PHASE_B_SHARD_SIZE=10
export RASP_PHASE_B_GPU_COUNT=8
export RASP_PHASE_B_MAX_BOUNDARIES_PER_EXAMPLE=12
export RASP_PHASE_B_RUN_ROOT=runs/05_phase_b/01_aligned_banks/rasp_phase_b_aligned_bank_v2
bash scripts/44_collect_rasp_phase_b_aligned_bank.sh
```

所有新版 shard validation 为 `ok` 后：

```bash
SOURCE_ROOT=runs/05_phase_b/01_aligned_banks/rasp_phase_b_aligned_bank_v2 \
OUTPUT_ROOT=runs/05_phase_b/02_phase_b2/rasp_phase_b2_v3 \
bash scripts/45_prepare_rasp_phase_b2_data.sh
```

训练优先先跑六个裁决 variant，而不是一次跑满十个：

```bash
mkdir -p logs/05_phase_b
variants=(
  hidden_flip_linear hidden_flip_only
  uncertainty_flip_linear uncertainty_flip_only
  position_flip_linear ratio_only_flip_linear
)
for i in "${!variants[@]}"; do
  variant="${variants[$i]}"
  nohup env CUDA_VISIBLE_DEVICES="$i" PHASE_B2_VARIANTS="$variant" PHASE_B2_SEEDS="1 2 3" \
    bash scripts/46_train_rasp_phase_b2.sh \
    > "logs/05_phase_b/rasp_phase_b2_v3_${variant}.log" 2>&1 &
done
```

只有 hidden 至少在 linear 或受控容量 nonlinear 中稳定超过 uncertainty/position，才继续跑
multitask 与 Phase C；否则停止扩张 learned hidden policy。

六个裁决 variant 训练完成后，使用同一列表评估：

```bash
PHASE_B2_VARIANTS="hidden_flip_linear hidden_flip_only uncertainty_flip_linear uncertainty_flip_only position_flip_linear ratio_only_flip_linear" \
bash scripts/47_eval_rasp_phase_b2.sh
```

数据准备还会强制检查同一 boundary 的所有 candidate action 是否共享完全相同的 pre-action hidden、
entropy、confidence 与 position，避免 hidden/action 对齐错误被静默吞掉。

### Phase B2 v3 数据验收

新版 aligned bank 与 Phase B2 v3 数据准备已完成，可以启动裁决训练：

- `runs/05_phase_b/01_aligned_banks/rasp_phase_b_aligned_bank_v2/` 的 20/20 shard validation 均为 `ok`；
- 全部 shard 使用 `affected_next_token_decisions_v2`，`max_new_tokens=768`，dense paired flip 与
  dense replay flip 均为 `0`；
- 原始 bank 为 164 个 dense-correct problems、1926 个 boundary、13482 条 action rows；
- 过滤 27 个不足完整 affected window 的 boundary 后，v3 数据为 1899 个 boundary、
  11394 个 nonzero action rows、558 个 final-flip 正例；
- hidden/action 强一致检查通过，ratio grid 完整，dense action flip 为 `0`；
- 修复 window alignment 后，`36.2%` 的 nonzero token-divergence 标签发生变化，平均 divergence
  从 `0.1925` 升至 `0.2150`；position 最大值从错误的 `1.0` 修正为 `176/768=0.2292`；
- final-flip 正例仍为 558，说明主标签稳定，修复主要影响局部辅助标签与 runtime feature 定义；
- ratio `0.02 -> 0.40` 时，flip `1.26% -> 9.43%`、token divergence
  `0.0597 -> 0.4027`、hidden cosine drift `0.0684 -> 0.4348`，action 强度关系健康。

三 seed 均为 `100/16/24/24` 个 train/validation/calibration/test problems，数据源组成一致。
train action-positive rate 已稳定到 `4.49%–4.79%`；validation/test 因仅有 16/24 个问题仍存在
波动，最终必须报告三 seed 均值与方差。

四张 GPU 可用以下队列一次完成六个裁决 variant：

```bash
mkdir -p logs/05_phase_b

nohup env CUDA_VISIBLE_DEVICES=0 \
  PHASE_B2_VARIANTS="hidden_flip_linear position_flip_linear" PHASE_B2_SEEDS="1 2 3" \
  bash scripts/46_train_rasp_phase_b2.sh > logs/05_phase_b/rasp_phase_b2_v3_gpu0.log 2>&1 &

nohup env CUDA_VISIBLE_DEVICES=1 \
  PHASE_B2_VARIANTS="hidden_flip_only" PHASE_B2_SEEDS="1 2 3" \
  bash scripts/46_train_rasp_phase_b2.sh > logs/05_phase_b/rasp_phase_b2_v3_gpu1.log 2>&1 &

nohup env CUDA_VISIBLE_DEVICES=2 \
  PHASE_B2_VARIANTS="uncertainty_flip_linear ratio_only_flip_linear" PHASE_B2_SEEDS="1 2 3" \
  bash scripts/46_train_rasp_phase_b2.sh > logs/05_phase_b/rasp_phase_b2_v3_gpu2.log 2>&1 &

nohup env CUDA_VISIBLE_DEVICES=3 \
  PHASE_B2_VARIANTS="uncertainty_flip_only" PHASE_B2_SEEDS="1 2 3" \
  bash scripts/46_train_rasp_phase_b2.sh > logs/05_phase_b/rasp_phase_b2_v3_gpu3.log 2>&1 &
```

### Phase B2 v3 裁决结果

六个裁决 variant 已完整产生 18/18 checkpoint、18/18 train metrics 和 18/18 test eval，
且全部 checkpoint 均由 validation split 选择。三 seed 测试集聚合结果如下：

| variant | ROC-AUC | PR-AUC | budget 0.15 utilization / flip | budget 0.20 utilization / flip |
|---|---:|---:|---:|---:|
| uncertainty nonlinear | `0.627 ± 0.057` | `0.098 ± 0.022` | `0.527 / 0.0169` | `0.587 / 0.0241` |
| uncertainty linear | `0.591 ± 0.121` | `0.070 ± 0.014` | `0.400 / 0.0131` | `0.513 / 0.0240` |
| hidden/combined nonlinear | `0.555 ± 0.060` | `0.083 ± 0.019` | `0.802 / 0.0276` | `0.751 / 0.0278` |
| hidden/combined linear | `0.510 ± 0.051` | `0.063 ± 0.017` | `0.811 / 0.0314` | `0.877 / 0.0408` |
| position linear | `0.553 ± 0.113` | `0.070 ± 0.030` | `0.267 / 0.0060` | `0.605 / 0.0207` |
| ratio-only linear | `0.568 ± 0.118` | `0.067 ± 0.031` | `0.000 / 0.0000` | `0.333 / 0.0147` |

这里的 `hidden` feature 实际为 raw hidden、entropy、confidence、position 和 dataset one-hot
的 combined 输入，并非纯 hidden。即使包含 uncertainty 信息，combined nonlinear 仍低于
uncertainty nonlinear；combined linear 更接近随机。当前表示下，raw hidden 没有提供稳定的
跨问题泛化增益。

训练行为也支持该判断：三个 hidden nonlinear checkpoint 均在 epoch `1–2` 被 validation loss
选中，hidden linear 为 epoch `1/2/10`；相比之下，uncertainty nonlinear 在 epoch `31/36/37`
才达到最佳点。2055 维 combined 输入相对 100 个训练问题明显更容易早期过拟合。

controller 结果不能只看平均 utilization：

- budget `0.15` 下 hidden nonlinear 的 seed 3 test flip 为 `5.13%`，超过目标上限 `4%`；
- `all_calibration_constraints_satisfied=True` 仅表示 calibration split 满足约束，不表示 test
  split 也满足约束；零剪枝同样会被记为满足约束；
- 部分 seed 的 budget `0.20` utilization 反而低于 `0.15`，说明只有 24 个 calibration problems
  时阈值选择仍不稳定；
- action-cell AUC 中同一 boundary 的六个 action 高度相关，因此三 seed 波动比单个 AUC 更值得重视。

**裁决：当前不进入 Phase C，不运行 multitask 扩张。** Phase C 会采集当前 policy-induced
states；在 policy 尚未稳定、raw hidden 尚未证明增益时执行，只会放大一个未通过裁决的策略。
下一步应先做一个受控的 Phase B2.5：固定 uncertainty nonlinear 为有效基线，验证低维/正则化
hidden residual 是否能在同一 split 上稳定增加 test ROC-AUC、PR-AUC，并改善受约束 controller
Pareto。只有 raw hidden 的增量价值通过三 seed 裁决后，才进入 Phase C。

该结果不推翻 Motivation 的原始结论。Motivation hidden probe 使用 problem-level OOF，覆盖
1342 个问题，五折 ROC-AUC 均约为 `0.812–0.843`，没有发现 problem leakage；它支持的是
“hidden 编码长期、强结构扰动下的 state fragility”。Phase B2 v3 检验的是只作用一个
16-token window、随后恢复 dense 的 MLP action 是否仍造成最终答案 flip。两者的主要差异为：

- Motivation 同时包含 layer/attention/MLP 等强扰动与 `0.2/0.4/0.6` ratio，正例率约 `45.8%`；
- aligned Phase B2 只使用 MLP intermediate action 与较短 horizon，正例率仅约 `4.9%`；
- Motivation 有 1342 个 problem-level OOF 问题；Phase B2 每个 seed 只有 100 个训练问题、
  16 个 validation、24 个 calibration 和 24 个 test；
- 长期 fragility 可以被当前状态稳定预测，不代表短期扰动恢复后仍会造成不可恢复答案错误。

当前 aligned-bank **数据与评估协议可接受**：action 确实生效、dense replay 为零 flip、
pre-action hidden 在同 boundary 各 ratio 完全一致、problem splits 互斥、test 不参与训练或
checkpoint selection。当前尚不足以最终裁决的是 **hidden 模型设计**：v3 没有纯 hidden 对照，
没有 train-only 标准化/低维投影，也没有 uncertainty + hidden residual；2055 维 combined 输入
直接对比 3 维 uncertainty，容易把“小样本下未学到”误判成“hidden 没有信息”。

Phase B2.5 应按以下顺序完成后再决定是否停止 hidden 路线：

1. 固定同一 manifest，增加纯 hidden、uncertainty-only、uncertainty + hidden residual；
2. 所有标准化与降维仅在 train problems 拟合，比较强正则 linear、低维投影和小容量 residual；
3. 同时报告 action-level final-flip 与 boundary-level any-flip，区分 state fragility 与 ratio 风险；
4. 使用 validation 选模型、calibration 选阈值、test 只做一次报告，并给出 problem bootstrap CI；
5. 若当前 bank 上出现稳定增量，再扩大 dense-correct problem 数；否则不直接投入昂贵 Phase C。

### Phase B2.5 实现与执行

Phase B2.5 已作为独立链路实现，不覆盖 `runs/05_phase_b/02_phase_b2/rasp_phase_b2_v3/`：

```text
src/rasp/phase_b25.py
src/main_train_rasp_phase_b25.py
src/main_eval_rasp_phase_b25.py
scripts/49_train_rasp_phase_b25.sh
scripts/50_eval_rasp_phase_b25.sh
scripts/51_summarize_rasp_phase_b25.py
```

固定使用 v3 的 dataset 与三个 manifest，比较四个受控 variant：

- `uncertainty_nonlinear`：标准化后的 entropy/confidence/position 小容量基线；
- `hidden_pca_linear`：纯 hidden，train-only 标准化 + PCA + 强正则线性 action-risk；
- `hidden_pca_nonlinear`：纯 hidden，train-only PCA + 小容量 nonlinear action-risk；
- `uncertainty_hidden_residual`：uncertainty/action 基线加低容量 hidden-PCA/action residual。

默认 PCA/model dim 均为 `32`。所有标准化和 PCA 仅由 manifest 的 train rows 拟合，变换状态随
checkpoint 保存；eval 会拒绝非 train-only transform。测试同时报告：

```text
action ROC-AUC / PR-AUC
boundary any-flip ROC-AUC / PR-AUC
problem-level bootstrap 95% CI
B15/B20 controller utilization / flip
```

服务器更新代码后先运行两 epoch smoke：

```bash
OUTPUT_ROOT=runs/05_phase_b/03_phase_b25/rasp_phase_b25_smoke \
PHASE_B25_VARIANTS="uncertainty_hidden_residual" PHASE_B25_SEEDS="1" PHASE_B25_EPOCHS="2" \
bash scripts/49_train_rasp_phase_b25.sh

OUTPUT_ROOT=runs/05_phase_b/03_phase_b25/rasp_phase_b25_smoke \
PHASE_B25_VARIANTS="uncertainty_hidden_residual" PHASE_B25_SEEDS="1" \
bash scripts/50_eval_rasp_phase_b25.sh
```

确认 smoke 产生 `policy.pt`、`train_metrics.json`、`eval.json` 和 `comparison_summary.csv` 后，
再启动正式四卡实验。

四张 GPU 可分别运行一个 variant：

```bash
mkdir -p logs/05_phase_b

nohup env CUDA_VISIBLE_DEVICES=0 PHASE_B25_VARIANTS="uncertainty_nonlinear" \
  bash scripts/49_train_rasp_phase_b25.sh > logs/05_phase_b/rasp_phase_b25_uncertainty.log 2>&1 &

nohup env CUDA_VISIBLE_DEVICES=1 PHASE_B25_VARIANTS="hidden_pca_linear" \
  bash scripts/49_train_rasp_phase_b25.sh > logs/05_phase_b/rasp_phase_b25_hidden_linear.log 2>&1 &

nohup env CUDA_VISIBLE_DEVICES=2 PHASE_B25_VARIANTS="hidden_pca_nonlinear" \
  bash scripts/49_train_rasp_phase_b25.sh > logs/05_phase_b/rasp_phase_b25_hidden_nonlinear.log 2>&1 &

nohup env CUDA_VISIBLE_DEVICES=3 PHASE_B25_VARIANTS="uncertainty_hidden_residual" \
  bash scripts/49_train_rasp_phase_b25.sh > logs/05_phase_b/rasp_phase_b25_residual.log 2>&1 &
```

四项训练结束后评估：

```bash
bash scripts/50_eval_rasp_phase_b25.sh
```

预期产生 12 个 checkpoint、12 个 train metrics、12 个 eval，以及：

```text
runs/05_phase_b/03_phase_b25/rasp_phase_b25/comparison_raw.csv
runs/05_phase_b/03_phase_b25/rasp_phase_b25/comparison_summary.csv
runs/05_phase_b/03_phase_b25/rasp_phase_b25/comparison_summary.json
```

准入 Phase C 的必要条件是 `uncertainty_hidden_residual` 在三 seed 上相对
`uncertainty_nonlinear` 稳定改善 action/boundary 排序，并在相近 test flip 下改善 controller
utilization；单 seed 提升、bootstrap CI 大量重叠或依赖更低 ratio 均不算通过。

### Phase B2.5 结果与裁决

Phase B2.5 已完整产生 12/12 checkpoint、12/12 train metrics 和 12/12 test eval。全部
checkpoint 由 validation 选择，全部标准化/PCA 仅在 train rows 拟合。三 seed 聚合结果：

| variant | action ROC / PR | boundary ROC / PR | B15 utilization / flip | B20 utilization / flip |
|---|---:|---:|---:|---:|
| uncertainty nonlinear | `0.553 / 0.077` | `0.508 / 0.196` | `0.601 / 0.0182` | `0.649 / 0.0255` |
| hidden PCA linear | `0.459 / 0.046` | `0.450 / 0.152` | `0.670 / 0.0276` | `0.781 / 0.0434` |
| hidden PCA nonlinear | `0.565 / 0.079` | `0.544 / 0.214` | `0.867 / 0.0265` | `0.881 / 0.0314` |
| uncertainty + hidden residual | `0.520 / 0.068` | `0.501 / 0.191` | `0.817 / 0.0288` | `0.741 / 0.0338` |

降维缓解了 raw hidden 的立即过拟合：hidden PCA nonlinear 的最佳 epoch 为 `6/8/31`，相比 v3
高维 combined nonlinear 的 `1/2/2` 更健康。它相对本轮 uncertainty baseline 的平均 action ROC
提高约 `0.011`、boundary ROC 提高约 `0.035`，说明 hidden 并非完全没有短窗口 fragility 信号。
但增益不稳定：

- action ROC 仅 seed 1 明显胜出，seed 2/3 均低于 uncertainty；
- action PR 平均只提高约 `0.003`；
- 所有 problem-bootstrap 95% CI 都很宽且大量重叠；
- 更高 utilization 同时伴随更高 flip，没有形成稳定 Pareto 优势。

用于检验增量价值的 residual 更明确失败：相对同轮 uncertainty baseline，三个 seed 的 action
ROC 分别下降约 `0.049/0.019/0.032`，平均 action/boundary ROC 分别下降约 `0.033/0.007`；
B15/B20 虽使用更多 ratio，但 flip 也分别增加约 `0.0106/0.0084`。因此当前
`uncertainty_hidden_residual` 不通过 Phase C 准入条件。

另外，B2.5 的小容量/full-batch uncertainty baseline action ROC 为 `0.553`，弱于 v3 已有最佳
uncertainty nonlinear 的 `0.627`。所以 B2.5 只用于同轮 hidden 增量诊断，不能替代 v3 最强
baseline；B2.5 最好的 hidden PCA nonlinear `0.565` 仍明显低于项目当前最强 uncertainty router。

该结果仍有一个实验边界：当前 residual 与 uncertainty 主分支联合从零训练，而不是冻结完全相同的
uncertainty checkpoint 后仅训练 residual。因此它可靠地否定当前联合 residual 结构，但还不是
hidden 增量信息的最终统计检验。下一步只应做一次低成本的 B2.5b：

1. 固定每 seed 已训练的 uncertainty baseline；
2. 冻结 baseline，只在 train split 学习零初始化的小容量 hidden/action residual；
3. 使用 validation 选择 residual 强度，并在同一 test predictions 上做 paired problem bootstrap；
4. 若三 seed paired delta 仍不为正，则停止 hidden router 路线，不进入 Phase C。

**当前裁决：不进入 Phase C。** 不扩大 bank、不运行 multitask，也不继续调 PCA/model dim。

### Phase B2.5b 冻结基线残差实验

最终低成本裁决链路已实现：

```text
src/rasp/phase_b25b.py
src/main_train_rasp_phase_b25b.py
src/main_eval_rasp_phase_b25b.py
scripts/52_train_rasp_phase_b25b.sh
scripts/53_eval_rasp_phase_b25b.sh
scripts/54_summarize_rasp_phase_b25b.py
```

每个 seed 直接加载并冻结 v3 的 `uncertainty_flip_only/policy.pt`。B2.5b 只训练
train-only hidden PCA + ratio residual，最后一层严格零初始化；validation 同时选择 epoch 与
`alpha ∈ {0, 0.25, 0.5, 0.75, 1}`。`alpha=0` 始终作为候选，因此 residual 无法通过 validation
选择时会精确退回原始 uncertainty baseline。正式 residual checkpoint 会内嵌冻结 baseline，
确保 eval 使用与训练时完全相同的基线权重。

Test 使用同一组 problems 和同一组 bootstrap 重采样，直接报告 combined 相对 frozen baseline 的：

```text
paired action ROC/PR delta + 95% CI
paired boundary ROC/PR delta + 95% CI
B15/B20 baseline vs combined ratio / flip
```

先运行单 seed 两 epoch smoke：

```bash
OUTPUT_ROOT=runs/05_phase_b/03_phase_b25/rasp_phase_b25b_smoke PHASE_B25B_SEEDS="1" PHASE_B25B_EPOCHS="2" \
bash scripts/52_train_rasp_phase_b25b.sh

OUTPUT_ROOT=runs/05_phase_b/03_phase_b25/rasp_phase_b25b_smoke PHASE_B25B_SEEDS="1" \
bash scripts/53_eval_rasp_phase_b25b.sh
```

Smoke 完成后，三张 GPU 并行正式运行：

```bash
mkdir -p logs/05_phase_b

nohup env CUDA_VISIBLE_DEVICES=0 PHASE_B25B_SEEDS="1" \
  bash scripts/52_train_rasp_phase_b25b.sh > logs/05_phase_b/rasp_phase_b25b_seed1.log 2>&1 &

nohup env CUDA_VISIBLE_DEVICES=1 PHASE_B25B_SEEDS="2" \
  bash scripts/52_train_rasp_phase_b25b.sh > logs/05_phase_b/rasp_phase_b25b_seed2.log 2>&1 &

nohup env CUDA_VISIBLE_DEVICES=2 PHASE_B25B_SEEDS="3" \
  bash scripts/52_train_rasp_phase_b25b.sh > logs/05_phase_b/rasp_phase_b25b_seed3.log 2>&1 &
```

训练完成后统一评估：

```bash
bash scripts/53_eval_rasp_phase_b25b.sh
```

结果写入 `runs/05_phase_b/03_phase_b25/rasp_phase_b25b/`，正式验收文件为：

```text
comparison_raw.csv
comparison_summary.csv
comparison_summary.json
```

只有三个 seed 的 paired action delta 稳定为正，且 controller 在相近或更低 flip 下增加 ratio，
才允许进入 Phase C。否则执行预先定义的停止条件：停止 hidden router，转向 uncertainty /
conservative RASP-Zero 在线主线。

### Phase B2.5b 最终结果与停止裁决

B2.5b 已完整产生 3/3 checkpoint、3/3 train metrics 和 3/3 test eval，全部冻结 v3
`uncertainty_flip_only`，PCA 只在 train rows 拟合，epoch/alpha 只由 validation 选择。

| seed | alpha | action ROC delta | paired 95% CI | boundary ROC delta | paired 95% CI |
|---|---:|---:|---:|---:|---:|
| 1 | `0.00` | `0.0000` | `[0.0000, 0.0000]` | `0.0000` | `[0.0000, 0.0000]` |
| 2 | `0.25` | `+0.0121` | `[-0.0248, +0.0643]` | `+0.0278` | `[-0.0136, +0.0704]` |
| 3 | `0.75` | `-0.0644` | `[-0.1756, +0.1178]` | `+0.0068` | `[-0.1085, +0.1525]` |

三 seed 平均 action ROC delta 为 `-0.0174 ± 0.0411`，boundary ROC delta 为
`+0.0115 ± 0.0145`，但所有 paired CI 均未形成稳定正增量。seed 1 validation 直接选择
`alpha=0`，说明 hidden residual 无法改善该 seed；seed 2 有小幅正增量但 CI 跨零；seed 3
action ROC 明显下降。

Controller 同样不形成稳定 Pareto：

- B15 ratio `0.0790 -> 0.0928`，但 flip `0.0169 -> 0.0182`；
- B20 ratio `0.1174 -> 0.1202`，flip 基本不变，但增量极小且跨 seed 不稳定；
- seed 3 B15 residual 将 flip 从 `2.93%` 提高到 `4.03%`。

**最终裁决：执行停止条件。** 当前 aligned short-window 任务上，没有证据证明 hidden 能为
uncertainty router 提供稳定、可泛化的增量价值。停止 hidden router，不进入 Phase C，不再扩大
hidden bank、调 PCA/model dim 或训练 multitask。Motivation 中 hidden 对长期/强扰动 fragility
的结论继续保留，但不将其外推为短窗口在线 action router。

后续主线转向：

1. 以 v3 `uncertainty_flip_only` 和 conservative RASP-Zero 作为在线候选；
2. 在相同预算下进行 paired online accuracy/flip/ratio 验证；
3. 明确 logical masking 仅报告理论 reduction，随后推进真实 reduced-weight backend；
4. 将 hidden fragility probe 保留为分析与解释模块，而不是在线控制器。

### Phase B3：uncertainty paired online 验证

当前第一优先级不是继续训练 router，而是检查 v3 uncertainty 离线信号是否能在真实自回归 rollout
中保持安全。现有旧 `rasp_train_policy` 在线入口依赖 hidden checkpoint，不能回答这个问题；
现已新增 `phase_b2_uncertainty` controller，严格复用 v3 checkpoint 的特征、校准 threshold、
单调风险包络和因果 prefix budget。v3 bank 只覆盖前 12 个窗口，因此 smoke 设置
`policy_horizon_tokens: 192`，超出已观测范围后强制恢复 dense，禁止无依据外推。

先在两张 GPU 上并行运行 20 题 paired smoke：

```bash
mkdir -p logs/06_phase_b3_online

nohup env CUDA_VISIBLE_DEVICES=0 PHASE_B2_ONLINE_DATASETS="gsm8k" \
  bash scripts/55_eval_phase_b2_uncertainty_online_smoke.sh \
  > logs/06_phase_b3_online/phase_b2_uncertainty_online_gsm8k.log 2>&1 &

nohup env CUDA_VISIBLE_DEVICES=1 PHASE_B2_ONLINE_DATASETS="math500" \
  bash scripts/55_eval_phase_b2_uncertainty_online_smoke.sh \
  > logs/06_phase_b3_online/phase_b2_uncertainty_online_math500.log 2>&1 &
```

结果位于 `runs/06_phase_b3_online/rasp_phase_b2_uncertainty_online_smoke/{gsm8k,math500}/`。每个数据集检查：

```text
dense/00_runtime_summary.json
b15/00_runtime_summary.json
b15/14_paired_dense_comparison.json
b20/00_runtime_summary.json
b20/14_paired_dense_comparison.json
```

Smoke 只做故障与方向检查。若 dense-correct flip 没有明显失控且实际 average ratio 非零，再扩到
三个 checkpoint seed 和更大 paired sample；若 uncertainty 在线同样没有 Pareto，则主线回到
conservative RASP-Zero，并开始真实 reduced-weight backend，不再继续 learned router。

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
