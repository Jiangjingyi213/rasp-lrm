# RASP-LRM 项目双周进展汇报

**汇报周期：** 2026-05-30 至 2026-06-14  
**项目方向：** Reasoning-Aware Dynamic Structured Pruning for Large Reasoning Models

## 1. 本阶段工作概览

Motivation 实验表明，大语言模型在不同推理状态下对结构化剪枝的敏感程度存在差异，固定剪枝策略难以同时兼顾推理准确率和剪枝强度。因此，过去两周的主要任务是将这一观察推进为可执行的动态剪枝方法。

本阶段围绕以下问题进行了验证：

1. Hidden state 能否表示当前推理阶段；
2. 当前推理上下文能否预测执行某个剪枝动作的风险；
3. 离线训练得到的风险模型能否在新题上进行在线剪枝决策；
4. 当前实验设计中还有哪些因素阻碍控制器真正学习剪枝时机。

经过多轮离线和在线实验，目前项目已经从宽泛的“根据 uncertainty 或 hidden 决定是否剪枝”，收束为更明确的 **Stage-Aware Action-Risk Controller**：

```text
hidden state
  → 识别当前推理状态

causal context + stage + candidate action
  → 预测执行具体剪枝动作的风险

online controller
  → 选择何时剪枝，以及采用多大的剪枝率
```

## 2. 前期实验的主要结论

### 2.1 有效结论

前期实验得到以下对当前方法设计仍然有效的结论：

- **Hidden state 包含明显的推理阶段信息。** Stage probe 能够较好地区分 `setup / reasoning / final`，说明从 reasoning stage 层面研究剪枝行为是可行的。
- **剪枝风险具有明显的状态差异和剂量响应。** 随着剪枝率增加，最终答案错误率整体上升；但相同剪枝率在不同题目和不同上下文中的风险并不相同。
- **因果上下文能够帮助预测剪枝风险。** 将当前上下文与候选剪枝动作共同输入风险模型，整体优于只根据剪枝率做判断。
- **动态选择剪枝率具有潜力。** GSM8K 在线 pilot 中，learned ratio selection 明显优于相近曝光量的固定剪枝动作。

这些结果共同说明：项目不应回到固定剪枝策略，而应继续研究能够感知推理状态并预测具体动作风险的在线控制器。

### 2.2 未通过实验的简要说明

部分实验未通过，但帮助明确了方法边界：

- Hidden state 无法稳定地直接预测单个 16-token 剪枝窗口是否会导致最终答案错误，因此 hidden 更适合作为 stage 和风险辅助特征，而不是独立的剪枝开关。
- 连续在线剪枝会积累误差，与单窗口离线训练存在分布差异，因此当前先限制为每题最多执行一个剪枝窗口。
- 第一版 learned controller 在线运行时几乎总在 token 32 执行动作，说明它主要学到了“剪多少”，尚未真正学会“何时等待、何时剪枝”。

## 3. 当前核心问题

在线控制器需要使用离线采集的数据学习通用决策规律：

```text
当前已经生成的因果信息
+ 当前推理状态
+ 候选剪枝率
→ 执行该动作的风险
```

随后，控制器才能在未见过的新题上，仅根据当前已经生成的信息进行实时判断。

上一版离线 Action-Risk bank 存在候选边界不完整的问题：

| 数据来源 | Problem 数 | 包含 token 32 | 包含 token 96 | 包含 token 160 |
|---|---:|---:|---:|---:|
| GSM8K train | 118 | 83 | 70 | 70 |
| MATH train | 160 | 93 | 83 | 75 |

由于同一道题不一定同时拥有 token `32 / 96 / 160` 的训练样本，离线模拟中的“等待到 token 96”可能只是因为该题没有 token 32 样本，而不是模型真正判断 token 32 不安全。

在线运行时，每道题都一定会经过 token 32，因此控制器最终对几乎所有题目都立即执行动作。这使上一轮实验只能验证动态剪枝率选择，不能验证 reasoning-aware 时机选择。

## 4. 当前计划：Stage-Action-Risk v2

当前最重要的任务是构建 **Stage-Action-Risk v2 精确边界 bank**，修复训练数据中的时机监督问题。

这一步并不是简单增加数据量，而是为控制器提供公平且完整的候选动作训练数据，使其能够真正学习：

```text
同一道题应该在什么时候剪，以及剪多少。
```

### 4.1 精确边界数据采集

对于每个保留的 dense-correct problem，强制采集完全相同的候选边界和动作：

```text
boundary = 32 / 96 / 160
ratio    = 0 / 0.05 / 0.10 / 0.20 / 0.30 / 0.40 / 0.50
window   = 16 tokens
```

每次非零剪枝动作只持续 16 token，窗口结束后立即恢复 dense。

每条样本记录：

- 当前已生成 token 数；
- 当前 entropy 和 confidence；
- 动作执行前的 causal hidden；
- stage probability；
- 候选剪枝率；
- 剪枝窗口内的输出变化；
- 最终答案是否发生 flip。

完整性验证要求：

- 每道题必须同时存在 `32 / 96 / 160` 三个边界；
- 每个边界必须拥有完整 ratio grid；
- 每个剪枝窗口必须完整执行 16 token；
- stage 和 hidden 必须来自动作执行前的信息；
- 缺失任何一项时，整道题不进入训练和评估。

### 4.2 离线模型比较

使用 problem-level 5-fold cross-validation，比较以下模型：

1. `action-only`：只根据候选剪枝率预测风险；
2. `causal context + action`：加入当前因果上下文；
3. `stage + causal context + action`：进一步加入推理阶段；
4. `hidden + stage + causal context + action`：加入 hidden 风险信息。

这一实验重点回答：

> 在所有题目都拥有相同候选时机的公平条件下，stage 信息是否能够稳定改善动作风险预测？

只有当 stage 模型在跨 fold 和跨数据集条件下稳定优于 context-only，才允许继续实现 stage-gated 在线控制器。

### 4.3 在线 Stage-Aware Controller

如果离线分析通过，下一步实现能够真正等待的单窗口控制器：

```text
生成到 token 32
  → 评估当前 stage 和每个候选 ratio 的风险
  → 风险过高则保持 dense，继续生成

生成到 token 96
  → 使用新的因果状态重新评估
  → 接受安全动作，或继续等待

生成到 token 160
  → 进行最后一次评估
```

控制器执行规则：

- 每道题最多执行一个 16-token 剪枝窗口；
- 控制器可以拒绝所有动作并全程保持 dense；
- hidden 和 stage 只能基于当前已经生成的信息；
- 剪枝窗口结束后永久恢复 dense；
- 优先选择风险可接受条件下更高的剪枝率；
- 不允许读取正确答案、未来 token 或完整轨迹信息。

### 4.4 在线验收

在线实验将在 GSM8K 和 MATH500 上与 dense、固定剪枝策略进行 paired comparison，主要报告：

- 题目级 accuracy delta；
- dense-correct flip rate；
- 实际执行剪枝窗口的题目比例；
- 平均 action ratio；
- 平均理论 MLP pruning exposure；
- 不同数据集和边界下的风险分布；
- bootstrap 95% confidence interval。

当前阶段只报告 logical mask 对应的理论 MLP reduction，不宣称真实 wall-clock 加速。

## 5. 阶段准入与停止条件

为了避免继续扩大无效路线，后续采用明确的阶段门槛。

### 离线阶段通过条件

Stage-Action-Risk v2 必须证明：

- stage 特征相对 context-only 在至少 4/5 folds 中改善风险预测；
- 改进同时出现在 GSM8K train 和隔离的 MATH train；
- 风险预测随 ratio 增大保持合理趋势；
- 模拟 first-accepted policy 能够真正选择不同边界，而不是全部集中在 token 32；
- 在保持足够剪枝曝光的同时，不增加两个数据集的 flip。

### 在线阶段通过条件

Stage-aware controller 必须：

- 在 GSM8K 和 MATH500 上均保持可接受的准确率；
- 相比相近曝光量的固定剪枝策略降低 dense-correct flip；
- 实际产生非零剪枝曝光；
- 能够根据题目状态选择不同剪枝时机；
- 多 seed 下保持稳定。

如果 stage 在精确边界数据上仍不能改善风险预测，则停止 stage-gated controller，不继续扩大数据规模。届时保留已经验证有效的 Action-Risk 结论，并重新考虑其他时机建模方法。

## 6. 当前进度

目前已经完成：

- 真实 runtime 单窗口 MLP-channel 剪枝和恢复流程；
- Hidden stage probe 与阶段标签审核；
- Action-Risk 离线训练和 OOF 评估框架；
- 固定单窗口与 learned single-window 在线诊断；
- 第一版 learned controller 失败原因定位；
- Stage-Action-Risk v2 精确边界采集、校验和分析代码实现。

Stage-Action-Risk v2 精确边界 bank 已完成：

```text
394 个 complete dense-correct problems
1182 个精确 boundary
7092 个非零 action rows
54 / 54 shard validation 通过
```

本轮成功修复了旧 bank 的候选边界缺失问题，但 stage 相对 causal context 仅在 `1/5` folds 中
同时改善 ROC-AUC 和 PR-AUC，未通过 stage-controller 准入条件。进一步分析发现，
first-accepted policy 即使拥有完整边界，仍会在 token 32 存在低风险动作时立即执行，因而没有
显式学习“等待的价值”。

当前正在推进的任务调整为：

```text
Stage-Action-Risk v2 paired exact-boundary bank
  → 构建 act-now vs wait 的 timing-value 标签
  → problem-level OOF 判断当前因果状态能否预测等待价值
  → 通过后再训练 stage-aware waiting controller
```

当前结果目录：

```text
runs/07_stage_aware/09_stage_action_risk_v2/
```

## 7. 下一阶段任务

1. 基于 v2 exact-boundary bank 定义当前动作效用和未来安全动作价值。
2. 训练并评估 `act-now vs wait` timing-value 模型。
3. 判断 hidden/stage/context 是否能稳定预测等待价值，而不仅是局部动作风险。
4. 若 timing-value 离线准入通过，实现新的 stage-aware single-window waiting controller。
5. 在线跨数据集验收通过后，再扩大数据规模、增加 seed，并考虑双窗口或连续动态剪枝。

## 8. 阶段性总结

项目目前尚未得到最终可部署的在线剪枝控制器，但已经明确了下一步需要解决的核心问题：

> 不再只判断某个固定时刻应该剪多少，而是让控制器根据推理状态，在多个候选时机之间等待，并选择风险可接受的最高剪枝率。

当前证据仍然支持 reasoning-aware 动态剪枝主线：

- hidden state 能够表示推理阶段；
- 推理上下文能够提供局部剪枝风险信号；
- 动态剪枝率选择已经显示出优于固定动作的潜力；
- 当前主要瓶颈是公平、完整地学习剪枝时机。

因此，下一阶段将以 Stage-Action-Risk v2 为核心，先验证 stage 是否真正帮助在线时机选择，再决定是否进入更大规模和连续剪枝实验。
