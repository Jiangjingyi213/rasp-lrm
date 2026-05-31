# Motivation 实验细节解释：从零理解每个结果是怎么来的

本文档用于解释当前 LRM pruning motivation 实验中每一个重要结果的来源、计算方式、公式和直觉含义。目标是让自己能够从头到尾讲清楚：

- 为什么要做反事实剪枝？
- 每种 module pruning 到底剪了什么？
- flip rate 是怎么来的？
- oracle analysis 到底是什么？
- entropy / confidence / activation / hidden state / combined probe 分别是什么？
- heatmap 和图像里的每个数字怎么解释？
- 当前结果为什么能支持 reasoning-aware pruning？

当前实验主模型为 `Qwen/Qwen3-1.7B`，主数据集为 GSM8K 和 MATH500。

## 1. 整个实验到底想证明什么？

我们想证明的核心 motivation 是：

> LRM 的剪枝风险不是一个固定的全局属性，而是随着推理过程变化。不同题目、不同 reasoning stage、不同 module、不同 pruning ratio、不同 hidden state 下，剪枝造成答案错误的风险都不同。

换句话说，传统剪枝方法常常假设：

```text
某个结构不重要 -> 可以一直剪掉
某个剪枝率安全 -> 对所有问题都安全
某个 prompt 适合剪枝 -> 整个生成过程都适合剪枝
```

但 reasoning model 的推理过程不是这样。一个模型在解数学题时可能经历：

```text
读题理解 -> 制定计划 -> 逐步推导 -> 检查验证 -> 给出最终答案
```

这些阶段依赖的模型结构可能不同。某些 FFN / attention / layer 在普通解释句子里不太重要，但在关键计算步骤里非常重要。因此我们需要实验回答：

1. 剪枝是否真的会让原本正确的推理变错？
2. 哪些 module 更危险？
3. 哪些 reasoning stage 更危险？
4. 是否存在 step-level dynamic pruning 的必要性？
5. 能不能用模型内部状态预测当前剪枝风险？

这就是整个 motivation 实验的出发点。

## 2. Dense Generation：为什么先跑未剪枝模型？

### 2.1 Dense 是什么意思？

`Dense` 指的是不剪枝的原始模型，也就是完整的 Qwen3。

在我们的实验里，Dense Qwen3 的作用是建立基线：

```text
原始模型能不能答对？
如果原始模型答对了，剪枝后会不会答错？
```

输出文件是：

```text
01_trajectories.jsonl
```

每一行大致包含：

```json
{
  "id": "...",
  "dataset": "gsm8k",
  "question": "...",
  "gold": "...",
  "prompt": "...",
  "completion": "...",
  "prediction": "...",
  "correct": true
}
```

### 2.2 为什么后续只分析 dense 原本答对的样本？

因为我们关心的是：

> 剪枝是否破坏了原本正确的 reasoning trajectory？

如果 dense 本来就答错，那么剪枝后仍然答错，不能说明剪枝破坏了正确推理。为了让风险定义更清晰，我们只在 dense-correct 样本上做 counterfactual pruning。

形式化写法：

设第 `i` 道题的 dense 输出为：

```text
y_i^dense
```

标准答案为：

```text
y_i^*
```

如果：

```text
y_i^dense = y_i^*
```

那么这道题进入后续 counterfactual 分析。

如果 dense 本来错了：

```text
y_i^dense != y_i^*
```

则不用于主要 flip risk 分析。

### 2.3 当前 Dense 结果

| 数据集 | 总题数 | Dense 答对 | Dense Acc |
|---|---:|---:|---:|
| GSM8K | 1319 | 1062 | 80.5% |
| MATH500 | 500 | 280 | 56.0% |

这说明 Qwen3-1.7B 本身有可观推理能力，因此后续观察到的剪枝风险有意义。

## 3. Answer Extraction：模型答案怎么判断对错？

模型输出通常是一整段推理文本，不是单个数字。因此需要从 `completion` 中抽取最终答案。

实现位置：

```text
src/metrics/answer_match.py
```

### 3.1 抽取优先级

代码大致按以下顺序抽取：

1. 如果有 `\boxed{...}`，优先抽取最后一个 boxed 内容。
2. 如果有 `Final answer:`，抽取其后的内容。
3. 如果有 GSM8K 标准格式 `#### answer`，抽取 `####` 后的内容。
4. 如果文本很短，直接清洗文本。
5. 否则抽取最后一个数字。

例如：

```text
Final answer: $18
```

会被清洗为：

```text
18
```

又例如：

```text
\boxed{\frac{14}{3}}
```

会被清洗为：

```text
14/3
```

### 3.2 判断对错

代码会尽量把答案转成数字或分数比较：

```text
Decimal(pred) == Decimal(gold)
```

或：

```text
Fraction(pred) == Fraction(gold)
```

如果不能转成数值，则做字符串归一化后比较。

这一点很重要，因为 MATH500 里有 LaTeX、分数、boxed answer 等格式。如果 parser 不够强，会低估模型准确率。

## 4. Reasoning Segmentation：为什么要切分推理过程？

### 4.1 为什么需要 segment？

我们关心的是：

> 在推理过程的哪个阶段剪枝更危险？

如果只看整段回答，就只能知道“这道题剪了以后对不对”。但 reasoning-aware pruning 需要更细：

```text
读题阶段剪枝安全吗？
规划阶段剪枝安全吗？
推导阶段剪枝安全吗？
验证阶段剪枝安全吗？
最终答案阶段剪枝安全吗？
```

所以我们把 dense completion 切成多个 segment。

输出文件：

```text
02_segments.jsonl
```

### 4.2 当前 segmentation 是怎么做的？

实现位置：

```text
src/segmentation/rule_segmenter.py
```

当前是 rule-based，不是人工标注。

分段逻辑大致是：

1. 优先找显式 step：

```text
Step 1:
Step 2:
Step 3:
```

2. 如果没有足够显式 step，就找更宽泛的列表/标题：

```text
### ...
1. ...
- ...
* ...
```

3. 如果还找不到，就按空行或句子切分。

4. 太短的片段会合并，默认 `min_chars=20`。

每个 segment 记录：

```json
{
  "segment_id": 0,
  "text": "...",
  "start_char": 123,
  "end_char": 456,
  "segment_type": "derivation"
}
```

### 4.3 segment_type 是怎么分的？

当前分类规则：

| 条件 | segment_type |
|---|---|
| 包含 `final answer` 或 `\boxed`，或最后一段且有 `answer` | `final` |
| 包含 `check`, `verify`, `therefore`, `sanity`, `confirm` | `verification` |
| 第一段且包含 `given`, `understand`, `we need`, `determine`, `find` | `understanding` |
| 前两段且包含 `plan`, `strategy`, `approach`, `steps`, `first` | `planning` |
| 其他 | `derivation` |

### 4.4 这有什么局限？

这是 heuristic，可能会错。例如：

- 一个真正的 verification 句子不一定写 `verify`。
- 一个 derivation 句子可能包含 `therefore`，被归为 verification。
- final 段可能不仅仅是答案，也包含最后一步计算。

所以论文里必须写：

```text
We use automatic rule-based stage assignment.
```

最好后续抽样人工检查 20-50 条。

## 5. Counterfactual Pruning：反事实剪枝到底是什么？

### 5.1 “反事实”是什么意思？

反事实的意思是：

> 假设模型在某个推理阶段被剪枝了，它接下来还会得到正确答案吗？

我们不是从头重新跑一个永久剪枝模型，而是在某个 segment 边界处构造一个“如果这里剪枝会怎样”的实验。

### 5.2 具体流程

对于一条 dense completion：

```text
Segment 0: ...
Segment 1: ...
Segment 2: ...
...
Final answer: 18
```

假设我们要在 `Segment 1` 后测试剪枝风险。

步骤是：

1. 取 dense completion 到 Segment 1 结束为止的前缀：

```text
prefix = completion[:segment.end_char]
```

2. 构造新 prompt：

```text
题目 + Reasoning so far: prefix + Continue reasoning
```

3. 临时对模型施加某种 pruning action。

4. 让模型从这个 prefix 后继续生成。

5. 抽取 counterfactual answer。

6. 比较它是否和 baseline answer 一样。

### 5.3 为什么要这样做？

因为我们想隔离问题：

> 在同一个已知正确推理前缀下，如果模型结构在此处被扰动，后续推理是否会偏离？

这比直接比较“剪枝模型从头生成”更细。它能告诉我们：

- 哪个 segment 之后更脆弱？
- 哪种 module pruning 更容易破坏后续推理？
- 哪个 action 在哪个 reasoning state 更危险？

### 5.4 Flip 是什么？

代码位置：

```text
src/metrics/flip_rate.py
```

定义为：

```python
extract_answer(baseline_text) != extract_answer(counterfactual_text)
```

数学写法：

设 dense baseline 答案为：

```text
a_i
```

counterfactual pruning 后答案为：

```text
a_i^{cf}
```

则：

```text
flipped_i = 1[a_i^{cf} != a_i]
```

注意这里比较的是 counterfactual answer 和 baseline answer，而不是直接和 gold 比。因为我们只对 dense-correct 样本做分析，所以 baseline answer 本身应当等于 gold。这样定义更直接地衡量“剪枝是否改变了原本正确答案”。

### 5.5 Flip Rate 是什么？

对一组 counterfactual rows：

```text
FlipRate = (# flipped rows) / (# all rows)
```

公式：

```text
FlipRate(G) = (1 / |G|) * Σ_{r in G} 1[flipped_r = true]
```

例如 module=layer 的 flip rate：

```text
所有 layer pruning 的 counterfactual rows 中，有多少比例发生 answer flip
```

## 6. Module Pruning：每种 module 到底剪了什么？

实现位置：

```text
src/pruning/contexts.py
src/pruning/layer_skipper.py
src/pruning/mlp_pruner.py
src/pruning/attention_pruner.py
```

当前 motivation counterfactual pruning 不是永久导出模型，而是通过 PyTorch forward hook 在生成时临时改变某些模块输出。

### 6.1 layer / layer_skip

代码逻辑：

```text
把指定 decoder layer 的输出替换为输入 hidden_states
```

直觉上，相当于跳过这一层：

```text
h_{l+1} = Layer_l(h_l)
```

变为：

```text
h_{l+1} = h_l
```

也就是该层不做任何变换。

为什么危险？

因为 transformer layer 同时包含 attention、MLP、norm、residual 等结构。跳过整层相当于删掉一整段计算，对 reasoning state 的影响最大。

### 6.2 attention_block

代码逻辑：

```text
把指定层 attention module 的输出置零
```

直觉上：

```text
Attention_l(h) -> 0
```

它模拟的是该层无法进行 attention 信息混合。

为什么重要？

Attention 负责 token 与 token 之间的信息交互。在推理中，当前 token 可能需要回看题目条件、前面计算结果、变量定义等。如果 attention block 被置零，模型可能失去上下文整合能力。

### 6.3 attention_heads

代码逻辑：

1. 获取 attention 输出 tensor。
2. 根据 hidden dimension 和 head 数，把最后一维 reshape 成：

```text
[..., num_heads, head_dim]
```

3. 将前 `ratio` 比例的 heads 输出置零。

如果无法确定 head 结构，则退化为 mask 最后一维前一部分 hidden dimensions。

公式化表示：

```text
head_j(h) = 0, for j in selected heads
```

这比 attention_block 温和，因为不是整个 attention 都没了，而是只屏蔽部分 heads。

### 6.4 mlp_block

代码逻辑：

```text
把指定层 MLP module 的输出置零
```

直觉上：

```text
MLP_l(h) -> 0
```

在 transformer 中，MLP/FFN 往往负责非线性变换、知识存储、局部计算模式等。对数学 reasoning 来说，MLP block 可能对公式转换、数值计算、模式匹配很重要。

当前结果中 mlp_block flip rate 很高，说明完整 MLP block 对 reasoning 非常关键。

### 6.5 mlp_channels

代码逻辑：

1. 获取 MLP 输出 tensor。
2. 取最后一维 hidden dimension。
3. 将前 `ratio` 比例的 channels 置零。

形式上：

```text
MLP_l(h)[..., 0:k] = 0
```

其中：

```text
k = round(width * ratio)
```

这比 mlp_block 更细粒度，因为不是整个 MLP 置零，而是只屏蔽部分输出 channel。

### 6.6 select_layers 是怎么选择层的？

代码位置：

```text
src/pruning/contexts.py
```

函数：

```python
select_layers(candidate_layers, ratio)
```

如果候选层是：

```text
[4, 8, 12, 16, 20, 24]
```

ratio=0.2，则选择大约 20% 的候选层，至少 1 层。若只选 1 层，取中间层。

ratio=0.6，则选择更多层，并尽量均匀分布。

所以在 counterfactual 中，ratio 有两层含义：

1. 对 `layer/attention_block/mlp_block`：表示从候选层集合里选多少层施加 block-level pruning。
2. 对 `attention_heads/mlp_channels`：既影响选哪些层，也传入模块内部决定 mask 多少 heads/channels。

这点解释时要小心。

## 7. 为什么有些结果不是严格随 ratio 单调？

直觉上，剪得越多应该越危险。但当前结果中：

```text
r=0.2 flip rate = 43.2%
r=0.4 flip rate = 41.6%
r=0.6 flip rate = 52.7%
```

0.4 不比 0.2 高。

原因是这里的平均 flip rate 是多个因素混合后的结果：

```text
module × ratio × layer group × stage × dataset
```

ratio=0.2 可能对应某些特别敏感的中间层，ratio=0.4 可能选到的组合平均反而稍微低一些。再加上不同 module 对 ratio 的使用方式也不同，所以不能把总表理解成纯粹的一维函数。

正确解释是：

> ratio 是风险因素之一，但不是唯一因素。module、stage、layer group 和 dataset difficulty 同样重要。

## 8. Oracle Analysis：到底是什么？

### 8.1 Oracle 是什么意思？

Oracle 在实验里指“知道真实结果后，选择最优策略的上限”。它不是一个实际可部署的方法，而是一个 theoretical upper bound / diagnostic upper bound。

简单说：

```text
如果我们事先知道哪个 action 最容易导致 flip，那么最佳选择能做到多好？
```

在我们的 motivation 实验里，oracle 不是为了部署，而是为了证明：

> 如果能更细粒度地选择 pruning action，潜在收益/差异会更大。

### 8.2 这里为什么 oracle 看的是 flip rate？

在 counterfactual motivation 中，我们关注的是“某种 action 是否会改变答案”。因此 oracle 选择的是 flip rate 最高的 action。

这听起来有点反直觉，因为正式剪枝当然想避免 flip。但这里的 oracle analysis 是用来度量：

> 不同粒度的 action choice 是否真的存在差异？

如果 step oracle 远高于 static oracle，说明不同 step 的敏感 action 不一样，step-level decision 有价值。

换句话说，这里的 oracle gap 是在证明“风险分布有结构”，不是在说部署时要最大化 flip。

### 8.3 Static Oracle

Static oracle 表示：

> 全数据集、所有题、所有 step 都固定选择同一个 action，看哪个 action 的平均 flip rate 最高。

定义：

设 action 为 `a`，所有 rows 为 `R`。

```text
score_static(a) = mean_{r in R, action(r)=a} flipped_r
```

Static oracle 选择：

```text
a_static* = argmax_a score_static(a)
```

Static oracle flip rate：

```text
max_a score_static(a)
```

当前结果：

```text
static oracle = 69.1%
```

说明全局最危险的固定 action 可以导致 69.1% 的 flip。

### 8.4 Prompt Oracle

Prompt oracle 表示：

> 每道题可以选择一个最适合这道题的 action，但这道题内部所有 segment 都用同一个 action。

对每个 problem `i`：

```text
a_i* = argmax_a mean_{segments s in problem i} flipped(i, s, a)
```

然后把每道题的最佳 action 统计起来。

当前结果：

```text
prompt oracle = 77.4%
```

说明如果允许每道题选择不同 pruning action，比全局固定 action 更能捕捉风险差异。

### 8.5 Step Oracle

Step oracle 表示：

> 每道题的每个 reasoning segment 都可以选择自己的最优 action。

对每个 problem-step `(i, s)`：

```text
a_{i,s}* = argmax_a flipped(i, s, a)
```

更准确地说，因为同一个 `(problem, segment, action)` 可能有不同 module/ratio/layer组合，代码按 choice group 计算该 step 下 action 的平均 flip，然后选最高。

当前结果：

```text
step oracle = 83.0%
```

### 8.6 Oracle Gap 怎么解释？

当前结果：

| Policy | Flip Rate |
|---|---:|
| static oracle | 69.1% |
| prompt oracle | 77.4% |
| step oracle | 83.0% |

解释：

1. Static oracle 到 prompt oracle 的提升说明：

```text
不同问题适合/敏感的 action 不同。
```

2. Prompt oracle 到 step oracle 的提升说明：

```text
同一道题内部，不同 reasoning step 的敏感 action 也不同。
```

这正是 RASP 的核心依据：

> 剪枝策略不应该只由全局模型结构决定，也不应该只由 prompt 决定，而应该跟随 reasoning step 动态变化。

### 8.7 Micro vs Macro 是什么？

代码里有：

- `prompt_oracle_flip_rate`
- `macro_prompt_oracle_flip_rate`
- `step_oracle_flip_rate`
- `macro_step_oracle_flip_rate`

micro 是按 row 数加权：

```text
总 flipped 数 / 总 row 数
```

macro 是先对每个 problem 或 step 求 rate，再平均：

```text
每个 problem/step 权重相同
```

如果不同题的 segment 数差异很大，micro 和 macro 会不同。

## 9. Entropy 与 Confidence：它们是什么？

实现位置：

```text
src/models/hooks.py
next_token_stats()
```

在每个 conditioned prompt 上，模型会输出最后一个位置的 logits：

```text
z ∈ R^V
```

其中 `V` 是词表大小。

通过 softmax 得到下一个 token 的概率分布：

```text
p_j = exp(z_j) / Σ_k exp(z_k)
```

### 9.1 Entropy

Entropy 衡量模型对下一个 token 的不确定性：

```text
H(p) = - Σ_j p_j log(p_j)
```

如果模型非常确定下一个 token，概率集中在少数 token 上，entropy 低。

如果模型很不确定，概率分散，entropy 高。

直觉：

```text
entropy 高 -> 模型不确定 -> 可能更容易被剪枝扰动
```

但实验发现 entropy 的预测能力有限。

### 9.2 Confidence

Confidence 是模型下一个 token 最大概率：

```text
Conf(p) = max_j p_j
```

如果 confidence 高，说明模型对某个下一个 token 很确信。

直觉：

```text
confidence 高 -> 模型更确定 -> 可能更安全
```

但实验中 confidence 也不能很好预测 pruning flip。

### 9.3 为什么 entropy/confidence 不够？

因为它们只看“下一个 token 的输出分布”，而剪枝风险可能来自更深的内部 reasoning state。

一个模型可能下一个 token 很确定，但内部表示已经处在一个很脆弱的推理状态。例如：

```text
它很确定要输出一个数字，但这个数字依赖前面复杂计算。
```

剪枝可能不会立即让下一个 token 看起来不确定，但会破坏后续多步推理。

这就是为什么 hidden-state probe 明显强于 entropy/confidence。

## 10. Activation Features：activation 是什么？

实现位置：

```text
src/models/hooks.py
activation_summary()
```

Activation 指模型某些层在 forward pass 中产生的中间输出。

在我们的实现中，对配置中的 layer_ids 注册 forward hook，捕获每个 layer 的输出 tensor：

```text
H_l ∈ R^{seq_len × hidden_dim}
```

然后取最后一个 token 的 hidden vector：

```text
h_l = H_l[-1]
```

对每个 layer 计算 4 个统计量：

1. L2 norm：

```text
||h_l||_2 = sqrt(Σ_j h_{l,j}^2)
```

2. mean：

```text
mean(h_l) = (1/d) Σ_j h_{l,j}
```

3. standard deviation：

```text
std(h_l) = sqrt((1/d) Σ_j (h_{l,j} - mean(h_l))^2)
```

4. max absolute value：

```text
max_j |h_{l,j}|
```

如果有 `L` 个 layer，每层 4 个值，则 activation feature 维度为：

```text
4L
```

这些是比较粗的内部激活统计。它们比 entropy/confidence 更接近模型内部状态，但仍然是低维摘要，因此效果不如完整 hidden state。

## 11. Hidden State：hidden feature 是什么？

实现位置：

```text
src/models/hooks.py
token_hidden_states()
```

Hidden state 是 transformer 某一层输出的完整 token representation。

当前实现默认取指定 hidden layer，通常是最后一层：

```text
hidden_layer = -1
```

对 conditioned prompt 做 forward：

```text
out = model(..., output_hidden_states=True)
```

得到：

```text
hidden_states[layer] ∈ R^{seq_len × hidden_dim}
```

然后取最后一个 token：

```text
h = hidden_states[layer][-1] ∈ R^{hidden_dim}
```

这就是 hidden feature。

对 Qwen3-1.7B 来说，hidden_dim 通常是 2048，因此 hidden feature 是一个 2048 维向量。

### 11.1 为什么 hidden state 有用？

因为它不是只看下一个 token 的概率，而是包含了模型到当前 reasoning prefix 为止的内部表示。

它可能编码了：

- 当前题目类型
- 已经走到哪一步
- 当前变量关系
- 前面计算是否稳定
- 当前 token 是否处在关键推理点
- 模型对后续推理路径的内部准备状态

因此 hidden state 更适合作为 reasoning-criticality estimator 的输入。

## 12. Combined Feature 是什么？

`combined` 是把多种 feature 拼接起来：

```text
combined = [hidden ; activation ; entropy ; confidence]
```

也就是：

```text
x_combined = concat(x_hidden, x_activation, x_entropy, x_confidence)
```

直觉上，combined 应该包含最多信息。但当前结果中 combined 和 hidden 很接近，有时略低于 hidden。这可能说明：

1. hidden state 本身已经包含主要信息。
2. activation/entropy/confidence 维度少，增益有限。
3. 线性 probe 对拼接特征的利用能力有限。

## 13. Risk Probe：风险探针到底在训练什么？

实现位置：

```text
src/probes/train_probe.py
src/probes/dataset.py
```

### 13.1 训练目标

每一条 counterfactual row 都有标签：

```text
y = 1 if flipped else 0
```

Probe 的目标是根据 feature `x` 预测：

```text
P(flipped = 1 | x)
```

也就是当前 reasoning state/action 是否高风险。

### 13.2 当前模型：Linear Risk Probe

当前 probe 是一个线性模型：

```text
logit = w^T x + b
```

通过 sigmoid 得到风险分数：

```text
score = σ(logit) = 1 / (1 + exp(-logit))
```

其中：

- `score` 越接近 1，表示预测越可能 flip。
- `score` 越接近 0，表示预测越安全。

### 13.3 Loss 函数

训练使用 binary cross entropy with logits：

```text
L = - [ y log σ(logit) + (1-y) log(1 - σ(logit)) ]
```

如果真实 `y=1`，模型应该输出高 score。

如果真实 `y=0`，模型应该输出低 score。

### 13.4 为什么要 problem-level split？

如果 row-level split，同一道题的不同 segment/action 可能同时出现在 train 和 validation 中。这样模型可能记住题目特征，而不是真正泛化到新题。

因此当前使用 problem-level split：

```text
同一道题的所有 rows 只能在 train 或 val 一边。
```

这更严格，也更接近真实泛化能力。

### 13.5 ROC-AUC 是什么？

ROC-AUC 衡量模型把正例排在负例前面的能力。

可以理解为：

```text
随机抽一个 flipped=1 的样本和一个 flipped=0 的样本，
模型给正例更高 score 的概率。
```

如果 ROC-AUC = 0.5，表示和随机猜差不多。

如果 ROC-AUC = 1.0，表示完美区分。

当前结果：

| Dataset | Feature | ROC-AUC |
|---|---|---:|
| GSM8K | entropy | 0.566 |
| GSM8K | confidence | 0.565 |
| GSM8K | activation | 0.641 |
| GSM8K | hidden | 0.831 |
| GSM8K | combined | 0.828 |
| MATH500 | entropy | 0.624 |
| MATH500 | confidence | 0.515 |
| MATH500 | activation | 0.579 |
| MATH500 | hidden | 0.850 |
| MATH500 | combined | 0.847 |

解释：

- entropy/confidence 只有弱预测能力。
- activation 有一定提升。
- hidden/combined 明显最好。

这说明 hidden state 中确实包含与 pruning risk 相关的信息。

### 13.6 PR-AUC 是什么？

PR-AUC 是 precision-recall curve 下的面积。

它在类别不平衡时很有用。例如 flipped 比例不是 50%，那么只看 ROC-AUC 可能不够。PR-AUC 更关注模型找到 positive class，即 flipped cases 的能力。

在我们的任务里，positive class 是：

```text
flipped = true
```

也就是“高风险样本”。

PR-AUC 越高，表示模型越能把真正高风险的 segment/action 排在前面。

## 14. Heatmap 是怎么算的？

Heatmap 本质上都是 group-by 后求 flip rate。

实现位置：

```text
src/main_heatmap_summary.py
```

### 14.1 group-by 公式

假设我们要算 module × ratio 的 heatmap。

对每个组合：

```text
G_{m,r} = { rows | row.module = m and row.ratio = r }
```

对应 flip rate：

```text
FlipRate(m,r) = (1 / |G_{m,r}|) Σ_{row in G_{m,r}} 1[row.flipped]
```

stage × module、stage × ratio、dataset × module 都是同理。

### 14.2 为什么 heatmap 有意义？

因为它能展示风险是否均匀。

如果所有格子颜色都差不多，说明剪枝风险主要是全局统一的。

但当前 heatmap 显示不同格子差异明显，例如：

- `layer` 在很多 stage 都高风险。
- `mlp_block` 也高风险。
- `attention_heads` 和 `mlp_channels` 相对温和。
- `verification` 对 layer / mlp_block 很敏感。

这说明 pruning policy 应该根据 state 和 module 变化。

## 15. 当前几张图应该怎么讲？

### 15.1 Stage × Module Heatmap

这张图回答：

```text
不同推理阶段对不同 module 剪枝是否一样敏感？
```

解释重点：

- planning / derivation / verification 更敏感。
- layer / mlp_block 在多个 stage 中最危险。
- final 阶段较低，但要注意 heuristic stage assignment 的局限。

可以说：

> 这张图证明 pruning risk is stage-dependent and module-dependent。

### 15.2 Stage × Ratio Heatmap

这张图回答：

```text
不同推理阶段对不同剪枝强度是否一样敏感？
```

解释重点：

- 高 ratio 通常更危险，但不是唯一因素。
- 某些 stage 即使低 ratio 也可能敏感。

可以说：

> 剪枝预算不能只全局固定，而应该按 reasoning stage 调整上限。

### 15.3 Module × Ratio Heatmap

这张图回答：

```text
剪哪个 module 和剪多少之间是否存在交互？
```

解释重点：

- layer + high ratio 最危险。
- mlp_block + high ratio 也危险。
- heads/channels 相对温和。

这支持后续 action router 同时选择：

```text
module + ratio + layer group
```

### 15.4 Oracle Gap 图

这张图回答：

```text
step-level decision 是否比 static/prompt-level decision 更有价值？
```

解释重点：

- static oracle = 69.1%
- prompt oracle = 77.4%
- step oracle = 83.0%

可以说：

> 由于 step oracle 明显高于 prompt oracle，同一道题内部不同 reasoning step 的敏感性确实不同。

### 15.5 Entropy Not Enough 图

这张图回答：

```text
只用输出不确定性指标能否预测剪枝风险？
```

解释重点：

- entropy/confidence ROC-AUC 接近 0.5-0.6。
- hidden-state probe ROC-AUC 达到 0.83-0.85。

可以说：

> 剪枝风险不是简单的 next-token uncertainty，而是更深层的 internal reasoning state fragility。

## 16. Baseline 结果应该怎么解释？

### 16.1 Dense Qwen3

Dense 是原始模型上界：

| 数据集 | Acc |
|---|---:|
| GSM8K | 80.52% |
| MATH500 | 56.00% |
| Combined | 73.78% |

这说明模型本身足够强。

### 16.2 GRIFFIN-style

GRIFFIN-style 是动态 FFN pruning，但不是 reasoning-aware。

结果：

| 方法 | Combined Acc |
|---|---:|
| GRIFFIN p20 | 35.57% |
| GRIFFIN p40 | 6.82% |
| GRIFFIN p60 | 2.58% |

解释：

> 动态剪枝不等于安全剪枝。只根据 sequence-level FFN activation 做 neuron selection，仍然可能破坏关键 reasoning steps。

### 16.3 FLAP-MLP

FLAP-MLP 是静态结构化 pruning。

结果：

| 方法 | Combined Acc |
|---|---:|
| FLAP-MLP p05 | 57.12% |
| FLAP-MLP p20 | 1.98% |
| FLAP-MLP p40 | 0.00% |
| FLAP-MLP p60 | 0.11% |

解释：

> 静态 MLP width pruning 对 reasoning model 很脆弱，尤其在 MATH500 上即使 p05 也明显下降。

这说明静态结构重要性不能完整刻画 reasoning-time importance。

### 16.4 LLM-Pruner-style 为什么排除？

我们尝试的 naive LLM-Pruner-style MLP port 在 p05 就出现 repeated-token collapse。

这说明当前实现不是可信 baseline。官方 LLM-Pruner 涉及：

- dependency graph
- group discovery
- Taylor importance
- pruning schedule
- LoRA recovery

不能用一个简单的 MLP L2 magnitude pruning 代替。

因此正式报告中应写：

> We exclude the naive Qwen3 MLP-only LLM-Pruner-style port due to generation collapse and leave faithful LLM-Pruner adaptation as future work.

## 17. 最终如何把这些结果串成论文 motivation？

可以按以下逻辑叙述：

### Step 1：Dense 模型有推理能力

Qwen3 在 GSM8K/MATH500 上有较强 dense accuracy。

### Step 2：正确推理轨迹对结构扰动敏感

Counterfactual pruning 导致大量 answer flip。

### Step 3：风险不是均匀分布

不同 dataset、module、ratio、stage 的 flip rate 不同。

### Step 4：step-level decision 有额外价值

Step oracle 明显高于 static/prompt oracle。

### Step 5：输出不确定性不够

Entropy/confidence 不能很好预测风险，hidden-state probe 明显更强。

### Step 6：已有 baseline 不够

GRIFFIN-style 和 FLAP-MLP 都显示非 reasoning-aware 剪枝会显著破坏 accuracy。

### Step 7：引出 RASP

因此需要：

```text
Reasoning-aware Dynamic Structured Pruning
```

它应该包含：

- reasoning-criticality estimator
- module/ratio router
- conservative verification policy
- budget-aware objective

## 18. 你可以如何口头解释这项实验？

可以这样讲：

> 我们首先用未剪枝的 Qwen3 在 GSM8K 和 MATH500 上生成完整推理轨迹，只保留原本答对的题。然后把推理过程自动切成若干 reasoning segments，例如理解、规划、推导、验证和最终答案。接着，在每个 segment 后面临时施加不同类型的结构剪枝，例如跳过 layer、置零 attention block、置零 MLP block、mask attention heads 或 MLP channels，再让模型继续生成。若剪枝后的最终答案与原本正确答案不同，就记为一次 answer flip。通过统计不同 module、ratio、stage 下的 flip rate，我们可以测量不同 reasoning state 对结构剪枝的敏感性。

然后继续：

> 结果显示，剪枝风险并不均匀。MATH500 比 GSM8K 更敏感，layer 和 MLP block 比 heads/channels 更危险，planning/derivation/verification 阶段比 final 阶段更脆弱。更重要的是，step oracle 明显强于 static 和 prompt oracle，说明同一道题内部不同推理步骤需要不同剪枝策略。最后，hidden-state probe 显著优于 entropy/confidence，说明剪枝风险不能只通过输出不确定性判断，而需要利用模型内部 reasoning state。

最后总结：

> 因此，LRM pruning 不应该是静态的，也不应该只根据 prompt 或 FFN 激活做动态选择，而应该根据当前 reasoning step 的风险动态决定剪哪个 module、剪多少，以及什么时候不剪。

## 19. 目前最需要记住的几个核心数字

Dense：

```text
GSM8K 80.5%
MATH500 56.0%
Combined 73.8%
```

Counterfactual：

```text
Total actions: 104280
Overall flip rate: 45.8%
GSM8K flip: 41.8%
MATH500 flip: 61.1%
```

Module risk：

```text
layer 57.4%
mlp_block 53.1%
attention_block 44.1%
mlp_channels 38.0%
attention_heads 36.6%
```

Stage risk：

```text
verification 60.4%
planning 50.8%
derivation 48.8%
understanding 44.8%
final 21.2%
```

Oracle：

```text
static 69.1%
prompt 77.4%
step 83.0%
```

Probe：

```text
entropy ROC-AUC: 0.566 / 0.624
hidden ROC-AUC: 0.831 / 0.850
```

Baselines：

```text
GRIFFIN p20 combined: 35.57%
FLAP-MLP p05 combined: 57.12%
Dense combined: 73.78%
```

## 20. 目前最需要谨慎的表述

1. 不要说 stage 是人工标注。
   - 应说：rule-based automatic stage assignment。

2. 不要说 counterfactual pruning 等价于真实加速。
   - 应说：counterfactual ablation for pruning-risk analysis。

3. 不要说 FLAP-MLP 是完整 FLAP。
   - 应说：Qwen3 MLP-only FLAP-style / FLAP-MLP static pruning baseline。

4. 不要说 GRIFFIN-style 是完整官方 GRIFFIN 复现。
   - 应说：GRIFFIN-style FFN dynamic pruning adapter。

5. 不要使用 naive LLM-Pruner-style MLP port 作为正式 baseline。
   - 它已经发生 repeated-token collapse。

6. 不要只用 p20/p40/p60 的崩溃说“所有剪枝都不行”。
   - 更准确说法是：非 reasoning-aware 的静态或 sequence-level pruning 在 LRM reasoning tasks 上风险很高。

## 21. 下一步应该怎么推进？

最自然的下一步是 RASP-Zero。

RASP-Zero 可以先不训练复杂模型，而是基于已有实验结果做规则策略：

```text
输入：
  segment_type
  risk_probe_score
  module risk table
  pruning ratio budget

输出：
  当前 step 是否剪枝
  剪哪个 module
  剪多少
```

示例规则：

```text
if risk_score > 0.7:
    no pruning or p05
elif segment_type in {verification, final}:
    cap ratio at p05 or p20
elif segment_type in {planning, derivation}:
    avoid layer/mlp_block; prefer heads/channels
else:
    allow stronger pruning
```

然后在已有 counterfactual table 上做 offline policy simulation，比较：

- static policy
- entropy-only policy
- confidence-only policy
- hidden-probe policy
- RASP-Zero
- step oracle

这一步成本低，但能直接判断 RASP-Zero 是否真的比简单 baseline 更接近 step oracle。
