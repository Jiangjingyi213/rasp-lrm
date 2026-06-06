# RASP-Train v1：从分析型 RASP-Zero 到可训练 Router

## 1. 为什么要从 RASP-Zero 转向 RASP-Train

目前的 RASP-Zero 已经完成了它最重要的分析任务：证明 reasoning step 的剪枝风险不是均匀的，而且可以用 hidden state、entropy、confidence 等信号进行预测。

但是 RASP-Zero 在线版仍然是一个手写规则系统：它先预测某个候选 ratio 的风险，再用阈值和 budget 规则选择动作。这个策略在离线风险预测上有信号，但在线生成时容易发生 reasoning drift，即模型在某个关键推理位置被剪枝后，后续推理路径偏离 dense 模型。

因此，RASP-Zero 更适合定位为 analysis prototype：

- 证明 step-level pruning sensitivity 存在；
- 证明安全剪枝动作存在；
- 证明风险可以被预测；
- 暴露简单阈值 router 不足。

RASP-Train 的目标是把这些 observation 变成一个正式方法：训练一个轻量 router，让它直接学习在当前 reasoning state 下应该选择哪个 pruning ratio。

## 2. RASP-Train v1 的核心定义

RASP-Train v1 不训练 Qwen3 本体，只训练一个轻量 ratio policy。

输入是当前 reasoning state：

- hidden state；
- next-token entropy；
- next-token confidence；
- 当前 step/token 位置；
- dataset/domain 信息；
- target budget，例如 0.15 或 0.20。

输出是一个 pruning ratio 分类：

```text
0.00, 0.02, 0.05, 0.10, 0.20, 0.30, 0.40
```

v1 仍然只控制 Qwen3 MLP intermediate channel 的 logical mask，不做 attention/head/channel 物理剪枝。因此当前效率指标仍然是 activated-MLP proxy，而不是实际 wall-clock speedup。

## 3. Oracle imitation 是什么

我们已有 runtime counterfactual bank。对每个 reasoning step，我们都试过多个 ratio，并记录剪枝后答案是否翻转。

例如某个 step 的结果可能是：

| ratio | flipped |
|---:|:---|
| 0.00 | false |
| 0.02 | false |
| 0.05 | false |
| 0.10 | true |
| 0.20 | true |
| 0.30 | true |
| 0.40 | true |

这说明这个 step 最多只能安全剪到 0.05。

RASP-Train v1 使用 `monotonic_safe_ratio`。它从 0 开始向上看，只要遇到第一个 flipped，就停止。这样比 `max_safe_ratio` 更保守，可以避免 counterfactual 噪声导致的非单调现象。

然后在给定 target budget 的情况下，构造 deployment-aligned causal safe oracle：

- 每个 step 的 ratio 不能超过自己的 `monotonic_safe_ratio`；
- 每道题独立分配预算，不允许不同题目互相借用预算；
- 按 reasoning step 顺序分配，不使用未来 step 的预算；
- 任意前缀上的平均 ratio 都不能超过目标 budget；
- 在上述约束和安全上限内选择当前最大的可执行 ratio。

第 \(t\) 个决策前可用的 ratio 为：

```text
available_t = B * (t + 1) - sum(previous_selected_ratios)
```

训练特征中同时加入 `available_budget_before_selection`，使 teacher oracle、训练 policy 和在线
controller 使用同一预算状态。

这个 oracle 不是最终方法，而是训练监督信号。RASP-Train 要学习模仿它。

对于非单调 counterfactual，例如 `0.10` 已翻转但 `0.20` 偶然未翻转，v1 采用保守解释：
所有高于 `monotonic_safe_ratio` 的动作都记为 `candidate_unsafe=true`。原始
`candidate_flipped` 仍被保留，用于区分真实 answer flip 与保守风险。

## 4. Loss 设计

RASP-Train v1 使用三部分损失：

```text
loss = cross_entropy + unsafe_weight * unsafe_penalty + budget_weight * budget_penalty
```

其中：

- `cross_entropy`：让 policy 预测 oracle ratio；
- `unsafe_penalty`：如果模型把概率分给真实翻转或超过 monotonic-safe 上限的 ratio，就增加惩罚；
- `budget_penalty`：让 batch 平均 ratio 接近 target budget。

安全惩罚优先级高于预算惩罚。也就是说，不能为了用满 B=0.20 而鼓励明显危险的剪枝动作。

## 5. 离线评估指标

RASP-Train 必须先通过离线评估，再进入在线生成。

离线评估直接使用 counterfactual bank 中的真实 flip label，不需要重新调用大模型。

核心指标：

- `flip_rate`：被 policy 选中的动作中，有多少会导致答案翻转；
- `conservative_unsafe_rate`：有多少动作超过 monotonic-safe 上限；
- `average_selected_ratio`：平均剪枝 ratio；
- `budget_utilization`：实际平均 ratio / 目标 budget；
- `oracle_match_rate`：预测 ratio 和 oracle ratio 完全一致的比例；
- `unsafe_over_oracle_rate`：policy 选择了超过 oracle 安全上限且导致 flip 的比例。

我们希望看到：

- 在相近 `average_selected_ratio` 下，RASP-Train 的 `flip_rate` 低于 RASP-Zero threshold router；
- `unsafe_over_oracle_rate` 明显下降；
- B=0.15 比 B=0.20 更稳，B=0.20 可作为 aggressive setting。

由于 ratio 是离散动作且部分 step 的安全上限较低，causal oracle 不保证精确用满预算。正式
runtime bank 上的前置检查得到 B15/B20 oracle 平均 ratio 约为 `0.1403/0.1872`；这类低于 1 的
budget utilization 是真实安全约束的结果，不应强行补齐。

离线评估中的 `rasp_train` 与 `rasp_zero_risk_budget` 均按每道题逐 step 回放预算历史。
`offline_noncausal_entropy_budget` 和 `offline_noncausal_confidence_budget` 只作为利用完整
验证集排序的诊断基线，不能描述为可直接部署的方法。

## 6. 当前实现产物

默认输出目录：

```text
runs/rasp_train_v1/
```

主要文件：

```text
runs/rasp_train_v1/common/
  05_probe_dataset_merged.jsonl
  05_probe_hidden_states_merged.pt

runs/rasp_train_v1/b15/
  11_rasp_train_policy_dataset.jsonl
  11_rasp_train_policy_hidden_states.pt
  11_rasp_train_policy_data_summary.json
  rasp_train_policy.pt
  13_rasp_train_metrics.json
  offline_eval/

runs/rasp_train_v1/b20/
  ...
```

运行顺序：

```bash
bash scripts/35_prepare_rasp_train_v1_data.sh
bash scripts/36_train_rasp_train_v1.sh
bash scripts/37_eval_rasp_train_v1_offline.sh
```

在线 smoke 会先运行同题、同 greedy decoder、`ratio=0` 的 control，再运行 B15/B20 policy。
每个 policy run 会生成：

```text
14_paired_dense_comparison.json
```

其中直接统计 `dense_correct_policy_wrong_count`，并保存对应题目和两种预测。

## 7. 2026-06-06 前置代码审查与修复

正式训练前已修复：

1. 保存最佳 checkpoint 时错误读取 policy row 的 `flipped` 字段导致的 `KeyError`。
2. dataset-level budget oracle 与 runtime per-problem budget 不一致。
3. offline 独立 argmax 与 online causal controller 不一致。
4. monotonic-safe oracle 与 unsafe penalty 定义不一致。
5. online smoke 缺少 paired ratio-0 control。

checkpoint 现在记录 feature schema。旧 RASP-Train dataset/checkpoint 必须重新生成和训练，
不能与当前版本混用。

仍需如实保留的边界：

- bank state 来自 dense reasoning segment 边界，在线 router 则每 `window_tokens` 更新，存在分布差异；
- 在线剪枝会改变后续 hidden state，离线 oracle imitation 不能消除 reasoning drift；
- logical mask 仍执行 dense projection，不代表 wall-clock speedup；
- v1 只路由 MLP intermediate-channel ratio，尚未实现多 module router。

## 8. 如何解释 RASP-Zero 与 RASP-Train 的关系

RASP-Zero 是分析型 prototype：

```text
risk score + 手写阈值 + budget 规则
```

RASP-Train 是正式方法雏形：

```text
reasoning state -> trainable ratio policy -> dynamic pruning action
```

也就是说，RASP-Zero 用来证明问题和提供 oracle/risk bank；RASP-Train 用这些数据训练一个真正的 reasoning-aware router。

论文表述上可以写成：

> RASP-Zero reveals that pruning safety varies across reasoning steps and can be estimated from reasoning states. However, threshold-based routing is insufficient for robust online generation. Therefore, we introduce RASP-Train, a lightweight oracle-imitation router trained to select budget-aware safe pruning actions from reasoning states.
