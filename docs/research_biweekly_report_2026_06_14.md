# RASP-LRM 项目双周进展汇报

**汇报周期：** 2026-05-30 至 2026-06-15
**项目方向：** Reasoning-Aware Dynamic Structured Pruning for Large Reasoning Models

## 1. 本阶段工作概览

Motivation 实验表明，LRM 在不同推理状态下对结构化剪枝的敏感程度不同，固定剪枝策略难以同时
兼顾推理正确率与剪枝强度。过去两周的核心任务，是将该观察推进为能够在真实生成过程中执行的
动态剪枝方法。

本阶段先后完成了：

1. 检查 uncertainty、hidden、stage 等特征能否预测局部剪枝风险；
2. 建立严格的 Action-Risk 离线采集、problem-level OOF 和在线 paired evaluation 框架；
3. 定位单窗口 learned controller 总在 token 32 执行动作的原因；
4. 从固定 `32/96/160` 边界扩展到完整因果轨迹；
5. 实现可重复执行多个 16-token 剪枝窗口的 runtime；
6. 实现能够精确重放历史剪枝动作的 on-policy bank collector；
7. 完成 Full-Trajectory Multi-Window 自动工作流的端到端验证。

目前项目主线已经重新收束为：

```text
完整轨迹 causal state
  + candidate pruning action
  + soft reasoning-stage information
  + prior pruning history
→ 预测动作造成 harmful flip 的风险
→ 在线选择何时剪枝、剪多少、是否继续剪枝
```

当前不再将 hard stage 直接作为剪枝开关，也不再限制每题只能执行一次剪枝。

## 2. 前期失败实验与得到的教训

### 2.1 Hidden 不能直接承担短窗口风险判断

Motivation 证明 hidden state 包含推理阶段与状态差异信息，但后续实验表明，hidden 无法稳定地
直接预测某个 16-token 剪枝窗口是否会导致最终答案变化。

因此当前对 hidden 的客观定位是：

```text
可以用于表示推理状态、长期脆弱性或辅助风险判断；
尚不能作为独立的局部剪枝开关。
```

这并不推翻 Motivation，而是限制了 hidden 在在线 Controller 中的具体角色。

### 2.2 Hard Stage Gate 未通过

当前 stage 分类结果高度集中于 reasoning，非 reasoning stage 样本过少。Stage-Action-Risk v2
虽然修复了训练候选边界不完整的问题，但 stage 相对 causal context 只在 `1/5` folds 中同时改善
ROC-AUC 与 PR-AUC，未通过 stage-controller 准入条件。

因此：

- hard stage 仅保留为诊断；
- 在线模型使用 soft-stage probability 作为辅助特征；
- 不再默认某个 stage 一定安全或一定不能剪枝。

### 2.3 单窗口策略无法充分验证最终方法

旧 learned single-window controller 几乎总在 token 32 立即执行动作。它主要学习了“剪多少”，
没有真正学习“是否等待”。同时，每题只剪一个 16-token 窗口的理论 exposure 很低，无法代表最终
动态剪枝方法。

因此当前实验已经扩展为完整轨迹、多次决策和多窗口执行。

## 3. Full-Trajectory Multi-Window 工作流

为避免继续被 `32/96/160` 三个边界和单窗口假设限制，本阶段实现了新的自动工作流：

```text
已有 Action-Risk Bank CPU 预审
→ Causal-Grid Dense Bank Smoke
→ Full-Trajectory Dense Bank Pilot + Grouped OOF
→ Fixed Multi-Window Dev
→ On-Policy 精确重放 Smoke
→ 阶段 Gate 与最终报告
```

### 3.1 完整轨迹采集

自然决策边界定义为：

```text
decision_start  = 32 tokens
decision_stride = 32 tokens
window_tokens   = 16 tokens
```

每个 boundary 测试完整 action grid：

```text
ratio = 0 / 0.10 / 0.20 / 0.30 / 0.40 / 0.50
```

模型可部署输入只包含当前已经生成的因果信息：

- generated token count；
- entropy / confidence；
- causal hidden；
- causal soft-stage probabilities；
- candidate ratio；
- 在 on-policy 阶段加入历史动作信息。

完整轨迹长度、相对位置、hard stage 和 tail-anchor 均只用于诊断，不进入在线模型。

### 3.2 Fixed Multi-Window Runtime

Runtime 允许每题执行多个剪枝窗口：

- 每次动作持续 16 token；
- 动作结束后恢复 dense；
- 至少经过 16 token dense cooldown 后才能再次动作；
- 达到最大窗口数量后永久恢复 dense；
- 每次动作记录完整历史和累计理论 pruning exposure。

### 3.3 On-Policy 精确重放

On-policy collector 不使用 dense prefix 替代真实剪枝轨迹，而是重放此前所有历史动作。只有以下
状态全部一致，boundary 才允许进入数据集：

- action schedule；
- forced token prefix；
- boundary next token；
- top-k logits、logits L2 和完整 logits hash；
- entropy / confidence；
- hidden state；
- cooldown 与历史动作数量。

## 4. 当前有效实验结果

### 4.1 完整轨迹 Pilot 数据通过严格验证

Full-Trajectory Pilot 最终保留：

```text
GSM8K dense-correct problems      20
MATH train dense-correct problems 20
causal boundaries                332
tail diagnostic boundaries        40
nonzero action rows             1660
positive answer-change labels    210
dense replay mismatch rate         0
```

所有采集 Gate 均通过：

- ratio grid 完整；
- dense replay 一致；
- 16-token 动作和 dense 恢复语义正确；
- causal feature audit 通过；
- 未使用 GSM8K test 或 MATH500。

### 4.2 剪枝风险具有清晰剂量响应

完整轨迹上的 final-answer change rate 随 ratio 单调上升：

| Candidate Ratio | Answer-Change Rate |
|---:|---:|
| 0.10 | 7.23% |
| 0.20 | 10.84% |
| 0.30 | 12.35% |
| 0.40 | 14.76% |
| 0.50 | 18.07% |

该结果说明：

1. 当前真实 runtime action 确实具有稳定风险监督信号；
2. 高 ratio 并非不能使用，但必须识别脆弱状态；
3. Controller 需要联合选择动作时机和动作强度。

两个来源存在明显差异：

| Ratio | GSM8K Answer Change | MATH Answer Change |
|---:|---:|---:|
| 0.10 | 2.22% | 10.66% |
| 0.20 | 6.67% | 13.71% |
| 0.50 | 16.30% | 19.29% |

MATH 对剪枝更敏感，后续必须报告分数据集 calibration，不能只使用合并风险阈值。

### 4.3 Dense-State Grouped OOF

使用 problem-level 5-fold OOF 比较不同特征：

| Variant | ROC-AUC | PR-AUC |
|---|---:|---:|
| action-only | 0.5858 | 0.1672 |
| causal context + action | 0.5919 | 0.1641 |
| soft-stage + context + action | **0.6063** | **0.1712** |
| hidden + context + action | 0.5592 | 0.1567 |
| hidden + soft-stage + context + action | 0.5567 | 0.1543 |

客观结论：

- soft-stage 在合并结果上提供小幅增益，但只在 `2/5` folds 稳定胜过 context，当前只能作为辅助；
- hidden 在当前局部 answer-change 预测任务上没有稳定增益；
- GSM8K 的 causal context 信号较明显，但 MATH 上各模型接近随机；
- 当前风险建模仍不足以直接训练可部署的 learned multi-window controller。

### 4.4 多次剪枝具有可行性

Fixed Multi-Window Dev 比较了不同 ratio、cadence 和最大窗口数。唯一同时通过两个训练来源方向
Gate 的策略为：

```text
r020_c32_m4
ratio          = 0.20
cadence        = 32 tokens
maximum windows = 4
```

| 数据集 | 理论 Exposure | Accuracy Delta | Dense-Correct Flip |
|---|---:|---:|---:|
| GSM8K | 4.48% | +5% | 0% |
| MATH train | 2.46% | 0% | 10% |

该实验的价值不是证明固定策略已经足够安全，而是证明：

- 多窗口 runtime 能够正确执行和恢复 dense；
- 每题执行多次剪枝可以产生非零理论 exposure；
- `r020_c32_m4` 可以作为采集真实 on-policy 状态的 behavior policy。

由于每来源只有 20 题，准确率变化与置信区间仍不稳定，不能宣称性能提升或最终安全。

### 4.5 On-Policy 精确重放通过

On-policy Smoke 结果：

```text
每来源有效问题             4
每来源 prior-action boundary 8
replay failures             0
invalid candidate boundaries 0
```

Action schedule、prefix、logits、hidden 和 cooldown 均能够精确复现。因此，当前实现已经具备采集
真实多窗口状态分布的能力，工作流允许扩大 on-policy bank。

## 5. On-Policy 扩大前发现并修复的问题

### 5.1 Answer Change 不等于 Harmful Flip

旧 on-policy 标签只记录 candidate 答案是否与 dense-control 不同。但当 dense-control 本身错误时，
答案变化可能是有益纠正，而不是风险。

Smoke 的 16 个 boundary 中确实出现了 1 个该情况：`ratio=0.10` 的唯一 answer change 将错误答案
修正为正确答案。

当前已经显式拆分：

```text
candidate_harmful_flip:
  dense-control 正确，但 candidate 错误

candidate_beneficial_correction:
  dense-control 错误，但 candidate 正确
```

后续安全风险模型必须使用 `candidate_harmful_flip`，不能继续使用一般 answer-change 标签。

### 5.2 排除 Behavior 错误会产生幸存者偏差

旧 Smoke 只保留 dense 与 fixed behavior 都正确的问题。这会系统性排除 fixed policy 已经破坏答案
的轨迹，使训练数据看不到最需要避免的失败状态。

新采集条件改为：

```text
dense trajectory 必须正确；
behavior trajectory 可以正确或错误；
behavior correctness 作为显式标签保留。
```

因此，扩大后的 on-policy bank 将同时包含安全轨迹和 behavior 失败轨迹。

## 6. 当前得到的项目结论

### 6.1 已经得到支持的结论

1. **动态剪枝问题具有可学习结构。**
   风险随 ratio 呈稳定剂量响应，并且同一 ratio 在不同数据集和状态下差异明显。

2. **多次剪枝是可行方向。**
   当前 runtime 已经能够执行多个剪枝窗口、恢复 dense 并产生非零 exposure，不必继续限制为每题
   只能剪一次。

3. **完整轨迹与 on-policy 数据是必要的。**
   Dense-state bank 只能描述首次动作风险；真实 Controller 必须学习历史动作已经改变状态后的风险。

4. **Reasoning-aware 信息仍然有价值，但不能使用 hard stage。**
   Soft-stage 提供了小幅辅助信号；stage 应作为连续状态特征，而不是人工规定哪些阶段允许剪枝。

5. **GSM8K 与 MATH 的风险机制不同。**
   MATH 更脆弱，且当前风险模型在 MATH 上泛化较弱，后续必须进行分数据集分析与 calibration。

### 6.2 尚未得到支持的结论

当前还不能证明：

- hidden 能稳定预测短窗口 harmful flip；
- 某个 hard reasoning stage 一定更安全；
- `r020_c32_m4` 是最终安全策略；
- learned multi-window controller 已经可以训练或部署；
- logical MLP mask 能带来真实 wall-clock 加速。

## 7. 下一阶段具体方向

下一阶段不再继续扩大 dense-state bank，也不直接训练最终 Controller。当前优先级是建立可信的
on-policy Action-Risk 数据与 grouped OOF。

### Step 1：重新运行修正后的 On-Policy Smoke

使用新的 harmful/beneficial 标签和无幸存者偏差准入条件，确认：

- behavior 错误轨迹能够进入 bank；
- 所有 retained problem 均来自 dense-correct trajectory；
- harmful flip 和 beneficial correction 能正确区分；
- 精确 replay 仍为 0 failure。

### Step 2：扩大 On-Policy Bank

使用 `r020_c32_m4` 作为数据采集 behavior policy，扩大两个训练来源上的真实多窗口状态：

- 优先达到每来源至少约 50 个有效 dense-correct problem；
- 每题采集多个包含 prior action 的 boundary；
- 保留完整 candidate ratio grid；
- 若 harmful positive 数量不足，再扩展至每来源约 100 个问题。

### Step 3：On-Policy Problem-Grouped OOF

比较：

```text
action-only
causal context + action
soft-stage + context + action
history + context + action
hidden + history + context + action
```

主要目标不是预测一般 answer change，而是预测：

```text
P(harmful flip | current on-policy state, action history, candidate action)
```

必须分别报告 GSM8K 与 MATH 的 ROC-AUC、PR-AUC、风险分桶和跨 fold 稳定性。

### Step 4：训练 Learned Multi-Window Controller

只有当 on-policy grouped OOF 证明风险模型相对 action-only 稳定提升后，才实现 learned controller：

- 每 32 token 评估一次；
- 根据当前状态、历史动作和候选 ratio 预测 harmful risk；
- 风险过高则保持 dense；
- 风险可接受时选择最高安全 ratio；
- 达到风险预算或最大窗口数后永久恢复 dense；
- soft-stage 作为辅助特征，hidden 仅在 OOF 证明增益后启用。

### Step 5：Paired Online 验收

最终对比：

```text
dense
fixed r020_c32_m4
learned multi-window
conservative fixed multi-window
```

报告 accuracy delta、dense-correct harmful flip、实际动作数量、理论 exposure、bootstrap 95% CI，
并分别在 GSM8K 与 MATH500 上验证。

## 8. 阶段性总结

本阶段最重要的进展不是已经得到最终 Controller，而是完成了从离线单窗口实验到真实多窗口
on-policy 学习问题的转换：

```text
完整轨迹风险信号已验证
→ 多窗口 runtime 已验证
→ behavior policy 已选定
→ on-policy 精确重放已验证
→ 可以开始采集训练最终 Controller 所需的数据
```

当前主线不再依赖“预先规定某个 reasoning stage 可以剪枝”，而是让 Controller 使用推理状态、
soft-stage、历史动作和 candidate ratio，学习每次动作是否会造成 harmful flip。

因此，下一阶段最明确的方向是：

> 扩大无幸存者偏差的 on-policy bank，并验证当前因果状态和历史动作能否稳定预测 harmful pruning
> risk；通过后再训练真正的 learned multi-window controller。
