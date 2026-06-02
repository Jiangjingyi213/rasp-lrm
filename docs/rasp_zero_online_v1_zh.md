# RASP-Zero Online v1：从反事实银行到在线动态路由

## 1. 当前版本要解决什么问题

前面的 motivation 实验已经说明：

> 同一个模型在不同 reasoning step 上，对剪枝的容忍度并不相同。

如果整段推理固定使用同一个剪枝率，容易在关键逻辑步骤上误剪。RASP-Zero Online v1
的目标是：在生成过程中定期观察当前 reasoning state，再动态决定下一段生成使用多强的
FFN 剪枝。

当前版本是 correctness-first 原型。它优先验证动态策略是否有效，不声称已经获得真实加速。

## 2. 当前 mask bank 是什么

候选动作集合为：

```text
ratio in {0.00, 0.02, 0.05, 0.10, 0.20, 0.30, 0.40}
module = mlp_intermediate_channels
layers = Qwen3-1.7B 的全部 28 层
```

`ratio=0.00` 表示 dense 对照。其他 ratio 表示：在 FFN 中间维度上，临时 mask 掉对应比例的
neurons。

模型先使用 dense prefill 读取题目和已有推理前缀。随后根据 prefix FFN activation 对 neurons
排序。排序靠前的 neuron 被视为更重要；剪枝时优先 mask 排名靠后的 neurons。不同 ratio 的
mask 是嵌套的。

这和本地 GRIFFIN adapter 的 activation ranking 思路一致，但当前 router 决定的是 reasoning
过程中动态变化的 ratio。

## 3. 反事实银行是什么

对 dense 模型原本答对的题目：

1. 生成完整 dense reasoning trajectory。
2. 自动切分 reasoning segments。
3. 在每个 segment 的起点恢复已有前缀。
4. 分别施加不同 ratio 的 MLP intermediate-channel mask。
5. 让模型继续生成答案。
6. 如果最终答案相对 dense 正确答案发生变化，记录：

```text
flipped = true
```

每一行训练数据表达的问题是：

> 在当前 reasoning state 下，如果使用这个具体 ratio，最终答案会不会被破坏？

因此，风险模型学习的是：

```text
P(answer flip | reasoning state, candidate pruning action)
```

## 4. Budget-aware safe oracle

### 4.1 最大安全 ratio

对每个 reasoning step，遍历候选 ratio，找出不会导致答案翻转的最大值：

```text
max_safe_ratio = max {r | flipped(step, r) = false}
```

这是理想化的离线上限，因为线上系统不知道真实答案，也不能提前知道某次剪枝是否翻转。
它用于回答：

> 如果存在一个完美 router，这个 step 最多可以安全剪多少？

### 4.2 为什么还记录 monotonic safe ratio

生成式模型不保证风险严格单调。有时 `ratio=0.20` 导致模型走入错误 token 路径，而
`ratio=0.40` 反而进入另一条正确路径。

因此额外记录：

```text
monotonic_safe_ratio
```

它表示：从 `ratio=0` 开始连续保持安全的最大 ratio。这个口径更保守。

### 4.3 预算约束

给定目标平均剪枝率 `B`，budget-aware safe oracle 在满足：

```text
average(selected_ratio) <= B
```

的前提下，只分配离线已知安全的动作，并尽量利用预算。它是线上 router 的参考上界。

## 5. Action-conditioned risk dataset

训练集排除 `ratio=0.00`，因为 dense fallback 不需要模型预测。每条 row 的标签仍为：

```text
y = 1  if answer flipped
y = 0  otherwise
```

第一版特征为：

```text
x = [
  hidden_state,
  entropy,
  confidence,
  generation_position,
  candidate_ratio,
  pruning_strength,
  module_one_hot,
  dataset_one_hot,
  layer_mask
]
```

其中：

- `hidden_state`：当前前缀最后一个 token 的模型内部表示。
- `entropy`：下一 token 概率分布的不确定性。
- `confidence`：下一 token 最大概率。
- `generation_position`：当前处于生成过程的相对位置。
- `candidate_ratio`：正在评估的候选剪枝率。
- `layer_mask`：哪些 Transformer 层参与当前动作。

当前 Online v1 不使用 heuristic stage 作为硬规则。原因是线上生成时还没有可靠的实时
stage classifier。后续可以将 stage 作为 soft feature 做消融。

## 6. Problem-level split 到底是什么意思

同一道题会产生很多 rows：

```text
problem
  -> 多个 reasoning segments
  -> 每个 segment 对应多个 candidate ratios
```

如果随机按 row 切分，某道题的 `ratio=0.05` row 可能进入训练集，而同一道题的 `ratio=0.20`
row 进入验证集。模型实际上已经见过这道题的 hidden state，验证指标会虚高。

因此必须按题目分组：

```text
同一道题的所有 segment、所有 ratio rows
只能整体进入 train 或整体进入 validation
```

代码使用：

```text
(dataset, id)
```

作为题目唯一键，避免不同数据集的 id 偶然冲突。

## 7. 第一版 router 如何训练

第一版使用线性风险 probe：

```text
risk_score = sigmoid(Wx + b)
```

损失函数：

```text
L = BCE(risk_score, flipped)
```

它不是最终 RASP-Train，而是可解释、轻量的 RASP-Zero 风险估计器。

## 8. 在线 router 如何工作

在线生成时：

1. 题目 prefill 始终 dense。
2. 每隔 `window_tokens=16` 个生成 token 更新一次路由。
3. 为每个候选 ratio 构造特征并预测风险。
4. 从大到小检查 ratio。
5. 选择满足以下条件的最大 ratio：

```text
predicted_risk <= risk_threshold
ratio <= 当前可用预算
```

如果没有候选动作满足条件，则回退到：

```text
ratio = 0.00
```

预算不会在推理开始时一次耗尽。每个 routing window 获得一小段额度；前面采用保守动作时，
额度可以积累到后面使用。

## 9. 当前版本的边界

Online v1 仅实现：

```text
MLP intermediate-channel logical mask
```

尚未实现：

- Attention 动态剪枝。
- Layer 动态跳过。
- 真实 reduced-weight kernel。
- 真实加速声明。
- 在线 stage classifier。
- RASP-Train 的联合优化。

因此当前可以严谨报告：

- accuracy
- answer flip
- selected ratio 分布
- 理论 FFN activated-parameter reduction
- 理论 FFN FLOPs reduction

当前不能把 logical mask wall-clock latency 当作最终加速结果。

## 10. 执行流程

正式 router 训练银行与最终 benchmark 测试集必须隔离：

```text
GSM8K train[:500]                    -> router bank
rasbt/math_full_minus_math500[:500]  -> router bank
GSM8K test                           -> 最终评估
HuggingFaceH4/MATH-500 test          -> 最终评估
```

正式银行默认拆成 20 个 shard，每个 50 题。四卡采集：

```bash
export PYTHON=/home/cike/jjy/envs/rasp_qwen3/bin/python
bash scripts/30_collect_runtime_bank_formal_four_gpu.sh
```

采集结果：

```text
runs/rasp_zero_runtime_bank_formal/
├── gsm8k_train_s00 ... gsm8k_train_s09
└── math_train_s00  ... math_train_s09
```

每个 shard 都会生成 `07_runtime_bank_validation.json`。再次运行 launcher 时，已经通过校验的
shard 会自动跳过，失败 shard 会单独重跑。

这里的 `500 + 500` 是第一版正式 router bank 的合理规模，但不是一次快速 smoke。上界情况下，
每道 dense 正确题最多产生：

```text
6 segments x 7 actions = 42 次 continuation
```

实际数量会因 dense 错题过滤和 segment 数不同而下降。P100 上仍可能需要连续运行多个晚上。
launcher 支持断点续跑，因此建议直接启动正式任务，并通过校验文件观察进度。

`math_train` 只表示训练题来源与 MATH500 test 隔离。构造 router 特征时，它会映射到共同的
math-domain feature，避免训练与评估之间出现人为的 one-hot 域偏移。

全部 shard 完成后，准备数据并训练 router：

```bash
export PYTHON=/home/cike/jjy/envs/rasp_qwen3/bin/python
bash scripts/27_train_runtime_router.sh
```

关键输出：

```text
runs/rasp_zero_runtime_router/
├── 00_runtime_router_data_summary.json
├── 08_safe_oracle_steps.jsonl
├── 08_budget_aware_safe_oracle_*.jsonl
├── 09_action_conditioned_risk_dataset.jsonl
├── 09_action_conditioned_risk_hidden_states.pt
├── 10_router_metrics.json
└── router.pt
```

在线小样本评估：

```bash
bash scripts/28_eval_rasp_zero_runtime_router_smoke.sh
```

它会在 GSM8K-2 与 MATH500-2 上分别运行：

```text
Dense logical-mask control
Online RASP-Zero router
```

在线轨迹中的 `runtime.router_events` 会记录每次更新时间、候选动作风险、最终 ratio 和预算状态。
