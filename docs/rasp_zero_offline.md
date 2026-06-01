# RASP-Zero Offline Evaluation：中文详细说明

## 1. 这一步实验到底在做什么？

这一阶段是我们开始构建自己方法的第一步。

此前，我们已经完成了 motivation 实验：

1. 使用未剪枝的 `Qwen/Qwen3-1.7B` 生成 GSM8K 与 MATH500 的完整推理轨迹。
2. 将每条推理轨迹自动切分成若干 reasoning segments。
3. 在每个 segment 后临时施加不同类型、不同强度的结构剪枝。
4. 观察剪枝后的模型是否仍然能够得到原本正确的答案。

motivation 实验告诉我们：

> 不同推理步骤对剪枝的敏感性不同。某些步骤可以承受剪枝，某些步骤一旦剪枝就容易导致答案错误。

因此，我们希望进一步回答：

> 能否根据当前推理状态，自动判断哪些步骤风险较低，并优先把剪枝预算分配给这些步骤？

`RASP-Zero Offline Evaluation` 就是在回答这个问题。

它是一个 **离线策略模拟实验**：

- 不需要重新运行 Qwen3 生成模型；
- 不会真正导出一个压缩后的模型；
- 不会测量真实 GPU 加速；
- 会复用此前已经得到的反事实剪枝结果；
- 会模拟不同剪枝策略在相同预算下会造成多少答案翻转。

这里的目标不是立刻获得可部署系统，而是先验证：

> reasoning-aware 动态剪枝策略是否值得继续开发？

---

## 2. 什么是 RASP-Zero？

`RASP` 是：

```text
Reasoning-state Adaptive Structured Pruning
```

可以翻译为：

```text
基于推理状态的自适应结构化剪枝
```

其中：

- `Reasoning-state`：当前模型正在进行哪一步推理，以及它内部的 hidden state 是什么；
- `Adaptive`：不是所有步骤使用固定剪枝强度，而是根据风险动态调整；
- `Structured Pruning`：剪掉具有明确结构的计算单元，例如 layer、attention block、MLP block 或通道；
- `Zero`：第一版不训练复杂路由器，主要通过已有风险信号和规则完成决策。

RASP-Zero 的最朴素思路是：

```text
低风险 reasoning step  -> 可以多剪一些
高风险 reasoning step  -> 少剪或保持 dense
```

这类似于做题时分配注意力：

- 简单的解释性步骤可以快速处理；
- 关键推导、回看和验证步骤需要保留更多计算资源。

---

## 3. 为什么叫 Offline Evaluation？

`Offline` 表示离线。

离线实验与真实运行时实验的区别如下：

| 对比项 | 离线策略模拟 | 真实运行时剪枝 |
|---|---|---|
| 是否重新运行大模型生成 | 否 | 是 |
| 是否真正减少运行时计算 | 否 | 是 |
| 是否能够测量真实 latency | 否 | 是 |
| 是否能够快速比较大量策略 | 是 | 较慢 |
| 当前阶段是否已经完成 | 是 | 尚未开始 |

为什么先做离线实验？

因为我们已经提前运行了大量反事实剪枝实验。现在可以利用这些数据快速判断：

```text
哪些策略值得继续投入工程时间？
哪些手写规则其实效果不好？
hidden state 是否比 entropy 更有用？
step-level 动态路由是否真的存在收益空间？
```

如果离线模拟都不能优于静态剪枝，就不应该急着实现复杂运行时系统。

---

## 4. 什么是反事实剪枝？

### 4.1 反事实是什么意思？

`Counterfactual` 可以翻译为“反事实”。

它表示：

> 已知模型原本能够正确完成某道题，现在人为改变某个条件，观察结果会不会变化。

例如，Dense 模型原本正确回答：

```text
16 - 3 - 4 = 9
9 x 2 = 18
Final answer: 18
```

我们在中间某个推理步骤后临时关闭部分 MLP block，再让模型继续生成。

如果新答案变成：

```text
Final answer: 16
```

那么就说明：

```text
该推理步骤 + 该剪枝动作
```

是危险的。

### 4.2 什么是 answer flip？

`Answer flip` 表示答案翻转：

\[
\text{flip}
=
\mathbb{1}
[
\text{pruned answer}
\ne
\text{dense answer}
]
\]

其中：

- `dense answer`：未剪枝模型的原始答案；
- `pruned answer`：施加剪枝后的答案；
- \(\mathbb{1}[\cdot]\)：如果条件成立则为 1，否则为 0。

简单来说：

```text
答案没变 -> flipped = false
答案变了 -> flipped = true
```

在我们的 motivation 实验中，只对 Dense 模型原本答对的题执行反事实分析。因此，answer flip 可以理解为：

> 剪枝是否破坏了原本正确的推理结果？

### 4.3 一条反事实数据包含什么？

每条紧凑数据来自：

```text
05_probe_dataset.jsonl
```

它大致对应：

```text
(dataset, problem, segment, module, ratio, pruned layers)
```

例如：

```text
dataset       = gsm8k
problem       = gsm8k-test-0
segment       = 第 2 个推理步骤
module        = mlp_block
ratio         = 0.4
pruned_layers = [4, 12]
flipped       = true
```

它的含义是：

> 对 GSM8K 第 0 题，在第 2 个推理步骤后，对指定层的 MLP block 施加剪枝。模型最终答案发生了翻转。

---

## 5. 什么是 reasoning segment？

一条完整推理链通常比较长。我们不会只把整道题看成一个整体，而是将它切成多个 reasoning segments。

`Segment` 可以翻译为“推理片段”或“推理步骤”。

例如：

```text
Step 1: 理解题目给出的数字
Step 2: 计算剩余鸡蛋数量
Step 3: 计算收入
Step 4: 输出最终答案
```

当前仓库使用 rule-based heuristic 自动分配 `segment_type`：

| Segment type | 中文解释 |
|---|---|
| `understanding` | 理解题目、提取条件 |
| `planning` | 制定解题计划 |
| `derivation` | 执行计算、公式推导 |
| `verification` | 回看、检查、验证 |
| `final` | 整理并输出最终答案 |

需要注意：

> 当前 stage 并不是人工标注，也不是 LLM classifier 标注，而是基于规则的自动划分。

因此它适合做 motivation 分析和初步策略实验，但论文中必须如实说明，并在后续抽样人工检查其合理性。

---

## 6. 什么是结构化剪枝？

剪枝的目标是减少模型计算量。

### 6.1 非结构化剪枝

非结构化剪枝会移除零散权重：

```text
某些矩阵元素置零
```

它可能减少理论参数量，但普通 GPU 未必能够直接获得明显加速。

### 6.2 结构化剪枝

结构化剪枝会移除具有明确形状的计算单元：

```text
完整 layer
完整 attention block
完整 MLP block
部分 attention heads
部分 MLP channels
```

结构化剪枝更接近实际部署需求，因为硬件更容易跳过整块计算。

### 6.3 当前 motivation action bank

当前反事实 action bank 包括：

| Module | 中文解释 | 当前实现的作用 |
|---|---|---|
| `layer` | 完整 decoder layer | 临时跳过若干层 |
| `attention_block` | 完整注意力模块 | 将若干层 attention 输出置零 |
| `mlp_block` | 完整前馈网络模块 | 将若干层 MLP 输出置零 |
| `attention_heads` | 部分注意力头 | 对部分 attention 输出维度做 mask |
| `mlp_channels` | 部分 MLP 通道 | 对部分 MLP 输出维度做 mask |

当前 RASP-Zero v1 只选择：

```text
mlp_block
```

也就是说，v1 先回答一个更简单的问题：

> 如果只考虑 MLP block，能否根据 reasoning state 决定哪些推理步骤可以剪枝？

多模块选择已经在 `RASP-Zero Offline Evaluation v2` 中继续扩展。

---

## 7. 什么是 pruning ratio？

`Pruning ratio` 表示剪枝强度。

当前 v1 使用：

```text
0.2 / 0.4 / 0.6
```

可以近似理解为：

| Ratio | 中文解释 |
|---:|---|
| 0.2 | 较轻剪枝 |
| 0.4 | 中等剪枝 |
| 0.6 | 较强剪枝 |

需要注意：

> 当前 ratio 是反事实压力测试强度，不等价于模型真实减少了完全相同比例的 FLOPs 或 latency。

不同 module 的 ratio 也不能直接横向等价。例如：

```text
layer ratio=0.2
```

通常比：

```text
mlp_channels ratio=0.2
```

更激进。

v1 只比较同一种 `mlp_block`，因此可以直接用 ratio 做预算匹配。

---

## 8. 什么是 pruning budget？

`Budget` 可以翻译为“预算”。

在这里，它不是指金钱，而是指：

> 平均允许使用多少剪枝强度。

例如：

```text
target_average_pruning_ratio = 0.2
```

表示所有 reasoning segments 平均下来，剪枝 ratio 希望接近 `0.2`。

一种静态策略可能是：

```text
每个 segment 都使用 ratio=0.2
```

一种动态策略可能是：

```text
1/3 的低风险 segment 使用 ratio=0.6
2/3 的高风险 segment 保持 dense
```

二者平均 ratio 都约为：

\[
\frac{1}{3} \times 0.6
+
\frac{2}{3} \times 0
=
0.2
\]

这就是 matched-budget comparison：

> 在平均剪枝预算相近的情况下，比较哪种策略更少破坏答案。

如果不匹配预算，比较就不公平。例如，一个策略完全不剪枝，flip rate 肯定更低，但它没有节省任何计算。

---

## 9. 什么是 hidden state？

`Hidden state` 可以翻译为“隐藏状态”或“内部表征”。

当模型读取题目并逐步生成推理文本时，每个 token 都会在模型内部形成一个向量：

\[
h_t \in \mathbb{R}^{d}
\]

其中：

- \(t\)：当前 token 或当前 prefix 的位置；
- \(d\)：隐藏维度；
- \(h_t\)：模型当前内部状态。

这个向量不是自然语言，但它浓缩了模型当前掌握的信息，例如：

```text
当前正在做哪一步计算
是否已经接近最终答案
是否处于逻辑分支
当前状态是否容易受到剪枝干扰
```

当前代码使用：

```text
当前 segment prefix 最后一个 token 的 hidden state
```

作为 probe 输入。

---

## 10. 什么是 risk probe？

`Probe` 可以翻译为“探针”。

它是一个轻量模型，用来检测大型模型内部是否存在某种可预测信号。

当前 probe 是一个线性分类器：

\[
z = w^\top h_t + b
\]

\[
\text{risk score}
=
\sigma(z)
=
\frac{1}{1 + e^{-z}}
\]

其中：

- \(h_t\)：当前 hidden state；
- \(w, b\)：probe 学习到的参数；
- \(\sigma\)：Sigmoid 函数；
- `risk score`：范围在 `[0, 1]` 内的风险分数。

它希望估计：

\[
\text{risk score}
\approx
P(
\text{answer flip after pruning}
\mid
\text{current reasoning state}
)
\]

可以直观理解为：

```text
risk_score 接近 0 -> 当前步骤相对稳健
risk_score 接近 1 -> 当前步骤容易被剪枝破坏
```

### 为什么只训练一个小 probe？

因为这一阶段的目的不是训练新大模型，而是验证：

> 剪枝风险是否已经存在于 LRM 的内部 reasoning state 中？

如果一个简单线性 probe 就能达到较好效果，说明信号确实存在，而且具有可解释价值。

---

## 11. 什么是 entropy 和 confidence？

### 11.1 Confidence

`Confidence` 可以翻译为“置信度”。

模型生成下一个 token 时，会对所有候选 token 给出概率分布。最大概率为：

\[
\text{confidence}_t
=
\max_i p(y_{t+1}=i \mid y_{\le t})
\]

如果最大概率很高，说明模型对下一个 token 比较确定。

### 11.2 Entropy

`Entropy` 可以翻译为“熵”或“不确定性”。

\[
H_t
=
-
\sum_i
p_i \log p_i
\]

其中 \(p_i\) 是下一个 token 为第 \(i\) 个候选 token 的概率。

直观理解：

```text
entropy 低  -> 概率集中，模型比较确定
entropy 高  -> 概率分散，模型比较犹豫
```

### 11.3 为什么不能只看 entropy？

因为模型即使对下一个 token 很确定，也不代表当前整个推理步骤适合剪枝。

例如：

```text
模型可能非常确定地输出一个数学符号，
但该步骤仍然依赖完整 MLP 计算。
```

因此：

```text
next-token uncertainty
```

与：

```text
structural pruning risk
```

并不是同一个概念。

这正是我们比较 entropy-only、confidence-only 和 hidden-state probe 的原因。

---

## 12. 什么是 problem-level OOF？

这是实验严谨性中非常关键的一部分。

### 12.1 为什么不能随机按 row 切分？

同一道题会产生很多相关数据：

```text
多个 reasoning segments
x 多种 modules
x 多种 ratios
x 多组 layers
```

如果简单随机划分 row，可能出现：

```text
同一道题的部分 action 在训练集
同一道题的其他 action 在验证集
```

这会产生数据泄漏。

Probe 可能只是记住某道题的特征，而不是真正学会泛化判断风险。

### 12.2 Problem-level split

我们按照：

```text
(dataset, problem id)
```

分组。

同一道题的全部 rows 只能进入：

```text
训练集
```

或者：

```text
验证集
```

不能同时出现。

### 12.3 什么是 OOF？

`OOF` 是：

```text
Out-of-Fold
```

可以翻译为“折外预测”。

我们使用 problem-level 5-fold：

1. 将全部题目分成 5 份；
2. 每次使用其中 4 份训练 probe；
3. 在剩余 1 份题目上预测风险；
4. 重复 5 次；
5. 拼接所有未见题目的预测结果。

这样，每一道题对应的 `risk_score` 都来自：

> 没有在训练阶段见过该题的 probe。

因此，OOF risk score 更适合用于离线策略评估。

---

## 13. 什么是 ROC-AUC 和 PR-AUC？

Probe 输出的是连续风险分数。我们需要判断它是否真的能够把高风险样本排在低风险样本前面。

### 13.1 ROC-AUC

`ROC-AUC` 衡量：

> 随机取一个 flipped 样本和一个 non-flipped 样本，probe 将 flipped 样本排在更高风险位置的概率。

直观理解：

| ROC-AUC | 解释 |
|---:|---|
| 0.5 | 接近随机猜测 |
| 0.6 - 0.7 | 有一定信号 |
| 0.7 - 0.8 | 信号较明显 |
| 0.8 以上 | 排序能力较强 |

### 13.2 PR-AUC

`PR-AUC` 更关注正类，也就是：

```text
flipped = true
```

它适合观察：

> Probe 能否较准确地找出真正危险的剪枝样本？

### 13.3 当前结果

当前 v1 hidden-state OOF probe 结果：

| 指标 | 数值 |
|---|---:|
| Counterfactual rows | 104,280 |
| Problems | 1,342 |
| Positive rate | 45.8% |
| ROC-AUC | 0.812 |
| PR-AUC | 0.782 |

说明：

> 当前 hidden state 中确实包含明显的剪枝风险信号。

---

## 14. RASP-Zero v1 比较了哪些策略？

### 14.1 `dense_lrm`

```text
完全不剪枝
```

用途：

- 作为安全上界；
- 确认 flip rate 应为 0；
- 但它不节省任何计算。

### 14.2 `static_mlp_block`

```text
所有 reasoning segments 使用完全相同的 MLP-block 剪枝强度
```

例如：

```text
static_mlp_block_r0.20
```

表示每个 segment 都使用 `ratio=0.2`。

用途：

> 代表最简单的一刀切静态策略。

### 14.3 `entropy_only`

```text
只根据 entropy 判断风险
```

模型越犹豫，策略越保守；模型越确定，策略越激进。

用途：

> 检查 next-token uncertainty 是否足够指导剪枝。

### 14.4 `confidence_only`

```text
只根据 1 - confidence 判断风险
```

用途与 entropy-only 类似。

### 14.5 `hidden_probe`

```text
根据 hidden-state linear probe 的 OOF risk score 分配预算
```

低风险 segment 优先获得剪枝预算，高风险 segment 保持 dense。

用途：

> 检验 reasoning internal state 是否比简单输出不确定性更有用。

### 14.6 `rasp_zero_stage`

在 hidden probe 基础上加入手写 stage 上限：

| Stage | 最大 ratio |
|---|---:|
| `verification` | 0.2 |
| `final` | 0.2 |
| `derivation` | 0.4 |
| `planning` | 0.6 |
| `understanding` | 0.6 |

原始假设：

```text
verification 与 final 阶段应更加保守
```

但实验结果表明：

> 手写 hard stage cap 未必稳定，特别是在较高预算下反而会恶化结果。

因此它现在应被视为：

```text
hard-stage-cap ablation
```

也就是一个有价值但失败的消融实验，而不是最终方法。

### 14.7 `prompt_router_safe_oracle`

`Oracle` 可以翻译为“理想上界”。

该策略假设：

> 对于每道题，我们提前知道哪一种固定 action 最安全。

它为每道题选择一次 action，但题目内部不同 segment 使用同一个 action。

用途：

> 衡量 prompt-level 路由的理论空间。

### 14.8 `step_safe_oracle`

该策略假设：

> 对每个 reasoning segment，我们提前知道哪种 action 最安全。

它可以对每个 segment 单独选择 action。

用途：

> 衡量 step-level 动态路由的理论安全上界。

它不是真正可部署方法，因为真实运行时不可能提前知道某次 action 是否会导致答案翻转。

---

## 15. 动态预算具体如何分配？

以 hidden-state probe 为例。

对于每个 reasoning segment，probe 给出：

```text
risk_score
```

策略会：

1. 按照风险从低到高排序；
2. 优先给低风险 segment 分配剪枝；
3. 在总预算约束下逐步增加 ratio；
4. 高风险 segment 可以保持 dense。

例如，平均预算为 `0.2` 时，策略可能选择：

```text
约 1/3 segments -> ratio=0.6
约 2/3 segments -> dense
```

平均值仍约为：

\[
\frac{1}{3} \times 0.6
=
0.2
\]

这种策略表达了一个重要思想：

> 与其对所有推理步骤平均剪一点，不如集中剪枝低风险步骤，保护高风险步骤。

---

## 16. 输出文件分别是什么？

运行脚本后会生成：

```text
runs/rasp_zero_offline/
├── hidden_probe_oof_scores.jsonl
├── hidden_probe_oof_summary.json
├── rasp_zero_offline_summary.json
├── rasp_zero_offline_summary.csv
├── rasp_zero_selected_actions.jsonl
└── figures/
    ├── rasp_zero_policy_frontier.pdf
    └── rasp_zero_policy_frontier.png
```

### 16.1 `hidden_probe_oof_scores.jsonl`

保存每条反事实 row 的 OOF 风险分数。

重要字段：

| 字段 | 含义 |
|---|---|
| `dataset` | 数据集 |
| `id` | 题目编号 |
| `segment_id` | 推理步骤编号 |
| `module` | 剪枝模块 |
| `ratio` | 剪枝强度 |
| `flipped` | 实际是否导致答案翻转 |
| `risk_score` | Probe 预测风险 |

### 16.2 `hidden_probe_oof_summary.json`

保存 probe 总体指标：

```text
ROC-AUC
PR-AUC
positive rate
每个 fold 的训练题数与验证题数
```

### 16.3 `rasp_zero_offline_summary.csv`

这是最适合快速查看的策略汇总表。

字段含义：

| 字段 | 含义 |
|---|---|
| `policy` | 策略名称 |
| `target_average_pruning_ratio` | 目标平均剪枝预算 |
| `n_problem_steps` | 参与模拟的 reasoning segments 数量 |
| `selected_action_flip_rate` | 策略选中的 action 中，最终导致答案翻转的比例 |
| `average_pruning_ratio` | 实际平均剪枝比例 |
| `average_pruning_strength_proxy` | 模块加权强度近似值 |

### 16.4 `rasp_zero_selected_actions.jsonl`

保存每个策略为每个 reasoning segment 具体选择了什么 action。

它适合用于进一步分析：

```text
哪些 stage 被剪得更多？
哪些题更脆弱？
哪些策略在 MATH500 上更保守？
```

### 16.5 `rasp_zero_policy_frontier.pdf`

该图横轴是：

```text
Realized average pruning ratio
```

也就是实际平均剪枝比例。

纵轴是：

```text
Answer flip rate
```

也就是答案翻转率。

在相同横轴位置下：

> 曲线越低，表示该策略越安全。

---

## 17. 当前正式结果如何解读？

### 17.1 OOF probe 结果

| 指标 | 数值 |
|---|---:|
| Counterfactual rows | 104,280 |
| Reasoning segments | 6,952 |
| Problems | 1,342 |
| Positive rate | 45.8% |
| ROC-AUC | 0.812 |
| PR-AUC | 0.782 |

说明：

> Hidden state 可以较好地区分高风险与低风险剪枝状态。

### 17.2 同预算策略比较

| 策略 | 20% budget flip rate | 40% budget flip rate | 60% budget flip rate |
|---|---:|---:|---:|
| Static MLP block | 47.3% | 50.7% | 61.2% |
| Entropy-only | 19.7% | 40.5% | 61.2% |
| Confidence-only | 19.6% | 40.7% | 61.2% |
| Hidden-state probe | **10.4%** | **31.5%** | 61.2% |
| Hard stage cap | 15.3% | 51.9% | 51.9% |

可以得到三个结论。

#### 结论 1：动态剪枝优于静态剪枝

在 `20%` 预算下：

```text
Static MLP block: 47.3%
Hidden-state probe: 10.4%
```

说明：

> 剪枝预算不应平均分配，而应集中给低风险 reasoning segments。

#### 结论 2：hidden state 比 entropy 和 confidence 更有用

在 `20%` 预算下：

```text
Entropy-only:    19.7%
Confidence-only: 19.6%
Hidden probe:    10.4%
```

说明：

> 结构剪枝风险不仅体现在下一个 token 的不确定性里，还存在于模型内部 reasoning representation 中。

#### 结论 3：手写 stage cap 不够可靠

在 `40%` 预算下：

```text
Hidden probe:   31.5%
Hard stage cap: 51.9%
```

说明：

> Stage 信息可能有用，但不能简单地通过硬编码规则决定剪枝率。

这也是我们继续开发 v2 的原因：

```text
将 stage 作为 soft feature 输入 probe
而不是直接写死 pruning cap
```

---

## 18. 什么是 strength proxy？

当前输出中还有：

```text
average_pruning_strength_proxy
```

`Proxy` 可以翻译为“近似指标”或“替代指标”。

它用于粗略表达：

> 某种剪枝动作相对有多激进。

当前权重为：

| Module | Weight |
|---|---:|
| `attention_heads` | 0.25 |
| `mlp_channels` | 0.25 |
| `attention_block` | 0.50 |
| `mlp_block` | 0.50 |
| `layer` | 1.00 |

计算公式：

\[
\text{strength proxy}
=
\text{module weight}
\times
\text{ratio}
\]

例如：

```text
mlp_block ratio=0.4
```

对应：

\[
0.5 \times 0.4 = 0.2
\]

需要特别强调：

> strength proxy 不是实测 FLOPs，也不是实测 latency，更不是 GPU 加速比。

它只是离线比较不同 action 时使用的诊断性近似值。

---

## 19. Oracle 为什么有两种口径？

这是很容易混淆的地方。

### 19.1 Motivation risk oracle

此前 motivation 文件：

```text
03_counterfactuals.oracles.json
```

中的 oracle 会寻找：

> 最容易导致答案翻转的 action。

它故意最大化 flip rate，用于证明：

```text
不同 problem 和不同 reasoning step 的结构敏感性确实不同
```

### 19.2 Deployment safe oracle

RASP-Zero 离线策略中的：

```text
prompt_router_safe_oracle
step_safe_oracle
```

会寻找：

> 尽量不导致答案翻转的 action。

它们用于衡量：

```text
理想安全路由还有多大提升空间
```

### 19.3 为什么不能混用？

因为二者目的相反：

| Oracle | 目标 |
|---|---|
| Motivation risk oracle | 最大化破坏，揭示敏感性 |
| Deployment safe oracle | 最小化破坏，衡量安全上界 |

论文中必须明确区分这两种 oracle。

---

## 20. 如何运行？

在远程服务器执行：

```bash
cd /home/cike/jjy/rasp-lrm
mkdir -p logs

export PYTHON=/home/cike/jjy/envs/rasp_qwen3/bin/python
export CUDA_VISIBLE_DEVICES=2
export TOKENIZERS_PARALLELISM=false

nohup bash scripts/19_rasp_zero_offline.sh \
  > logs/rasp_zero_offline.log 2>&1 &
```

查看日志：

```bash
tail -f logs/rasp_zero_offline.log
```

OOF probe 相比重新生成大模型轨迹要轻量得多。Probe 训练可以使用一张 GPU，策略模拟本身主要在 CPU 上运行。

---

## 21. 输入完整性检查在做什么？

运行脚本首先执行：

```text
src.main_validate_rasp_inputs
```

它会检查：

1. 每个正式 shard 的 `05_probe_dataset.jsonl` 是否可以完整解析；
2. JSONL 是否在传输过程中被截断；
3. 行数是否与 `03_counterfactuals.oracles.json` 记录一致；
4. 每个已执行 reasoning segment 是否具有完整的：

```text
5 modules x 3 ratios
```

action grid；

5. hidden-state tensor 是否存在。

如果文件损坏，脚本会停止，并指出具体文件和行号。

这样可以避免：

> 在输入不完整的情况下悄悄得到一个看似正常、实际错误的结果。

---

## 22. v1 的局限是什么？

RASP-Zero v1 是重要的第一步，但它不是最终方法。

### 局限 1：只使用 `mlp_block`

v1 还没有真正根据 reasoning state 选择：

```text
attention heads
MLP channels
attention block
MLP block
layer
```

中的最佳 action。

### 局限 2：Probe 只预测 step-level 风险

当前 probe 回答：

```text
这个 reasoning segment 整体脆弱吗？
```

但它不能回答：

```text
该 segment 可以剪 MLP channels，
但是否不能跳 layer？
```

### 局限 3：Stage cap 是手写规则

Stage cap 在较高预算下表现不好，说明：

> heuristic stage 不能直接决定剪枝比例。

### 局限 4：仍然不是运行时系统

当前 action 是诊断性 hook，不是已经优化的物理剪枝实现。

因此，v1 的正确定位是：

> 验证 reasoning-state-aware budget allocation 是否有效。

---

## 23. 与 RASP-Zero Offline v2 的关系

v2 在 v1 基础上继续推进。

核心变化：

| v1 | v2 |
|---|---|
| 只预测 reasoning step 是否危险 | 预测具体 action 在当前 step 是否危险 |
| 只选择 `mlp_block` | 在多个 module 中动态选择 |
| 手写 stage cap | Stage 作为 soft feature |
| 用 ratio 匹配预算 | 用 module-weighted strength proxy 匹配预算 |

v2 学习：

\[
q(s_t, a)
=
P(
\text{answer flip}
\mid
\text{reasoning state } s_t,
\text{ action } a
)
\]

其中：

- \(s_t\)：当前 reasoning state；
- \(a\)：候选剪枝动作。

详细说明见：

```text
docs/rasp_zero_offline_v2_zh.md
```

---

## 24. 最后用一句话总结

RASP-Zero Offline v1 做的事情是：

> 利用已有反事实剪枝数据训练一个轻量 hidden-state risk probe，并在相同平均剪枝预算下，将剪枝集中分配给低风险 reasoning steps。实验结果表明，这种 reasoning-state-aware 动态策略明显优于静态剪枝、entropy-only 和 confidence-only 策略，因此值得继续推进到 action-conditioned、多模块的 RASP-Zero v2。
