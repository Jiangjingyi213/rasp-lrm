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
mv runs/未命名 runs/rasp_phase_b_aligned_bank_12w
export PYTHON=/home/cike/jjy/envs/rasp_qwen3/bin/python
bash scripts/45_prepare_rasp_phase_b2_data.sh
```

三个 seed 可分别占用三张 GPU：

```bash
mkdir -p logs
for item in "0 1" "1 2" "2 3"; do
  set -- ${item}
  nohup env CUDA_VISIBLE_DEVICES="$1" PHASE_B2_SEEDS="$2" \
    bash scripts/46_train_rasp_phase_b2.sh \
    > "logs/rasp_phase_b2_seed_$2.log" 2>&1 &
done
```

训练全部完成后：

```bash
bash scripts/47_eval_rasp_phase_b2.sh
```

输出汇总为 `runs/rasp_phase_b2/comparison_summary.csv`。验收重点不是单一 AUC，而是：

1. `hidden_multitask` 是否稳定优于 `hidden_flip_only`；
2. hidden 是否稳定优于 uncertainty-only；
3. calibration constraints 是否跨三 seed 全部满足；
4. B15/B20 在保持低 flip 时能否获得有效 ratio，而不是退化到接近 dense。

### Phase B2 三 seed 结果

`runs/rasp_phase_b2/` 已完整产生 9 个 checkpoint、9 个 train metrics 和 9 个 test eval。数据为
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

上述修复已经实现，默认输出目录改为 `runs/rasp_phase_b2_v2/`，不会覆盖第一轮诊断结果：

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

mkdir -p logs
variants=(
  hidden_multitask hidden_flip_only
  uncertainty_multitask uncertainty_flip_only
  position_flip_only ratio_only_flip_only
)
for i in "${!variants[@]}"; do
  variant="${variants[$i]}"
  nohup env CUDA_VISIBLE_DEVICES="$i" PHASE_B2_VARIANTS="$variant" PHASE_B2_SEEDS="1 2 3" \
    bash scripts/46_train_rasp_phase_b2.sh \
    > "logs/rasp_phase_b2_v2_${variant}.log" 2>&1 &
done
```

六个日志均完成后执行：

```bash
bash scripts/47_eval_rasp_phase_b2.sh
```

只有 `comparison_summary.csv` 中 `all_checkpoints_selected_on_validation=True`，且 hidden 在三 seed
上稳定超过 ratio/position/uncertainty 基线并形成 controller Pareto 增益，才进入 Phase C。

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
