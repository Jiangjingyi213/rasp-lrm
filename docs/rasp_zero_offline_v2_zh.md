# RASP-Zero Offline Evaluation v2：中文实验说明

## 1. 为什么需要 v2？

`RASP-Zero Offline v1` 已经证明：推理过程中的 hidden state 能够预测剪枝风险。在 GSM8K 与 MATH500 的完整反事实数据上，problem-level 5-fold OOF hidden-state probe 达到：

| 指标 | 数值 |
|---|---:|
| Counterfactual rows | 104,280 |
| Problems | 1,342 |
| ROC-AUC | 0.812 |
| PR-AUC | 0.782 |

在相同的平均 MLP-block 剪枝预算下，hidden-state 路由明显优于 static、entropy-only 和 confidence-only：

| 策略 | 20% budget flip rate | 40% budget flip rate |
|---|---:|---:|
| Static MLP block | 47.3% | 50.7% |
| Entropy-only | 19.7% | 40.5% |
| Confidence-only | 19.6% | 40.7% |
| Hidden-state probe | **10.4%** | **31.5%** |
| Hard stage cap | 15.3% | 51.9% |

但是，v1 仍有两个局限：

1. hidden-state probe 只判断“当前 reasoning step 总体是否脆弱”，不知道具体 action 的风险。
2. 手写 hard stage cap 在高预算下反而恶化结果，说明 heuristic stage 不应直接决定剪枝率。

因此，v2 将问题改写为：

> 对于当前 reasoning state 和候选剪枝 action，预测该 action 导致答案翻转的概率，再在全局预算约束下选择风险最低的 action。

---

## 2. v2 的核心问题

对每个 reasoning segment \(s_t\) 和候选剪枝动作 \(a\)，学习：

\[
q(s_t, a)
=
P(\text{answer flip} \mid s_t, a)
\]

其中：

- \(s_t\)：当前推理状态；
- \(a\)：候选结构化剪枝动作；
- `answer flip`：剪枝后最终答案与 dense 正确答案不同。

这与 v1 的差别非常重要：

```text
v1: 这个推理步骤整体危险吗？
v2: 在这个推理步骤上，执行某个具体 action 危险吗？
```

v2 才真正对应论文中的 `Structure-sensitive Router`。

---

## 3. 输入数据来自哪里？

v2 不需要重新运行 Qwen3 生成模型。它复用已经完成的四个正式 motivation shard：

```text
runs/formal_qwen3_gsm8k_full_s0
runs/formal_qwen3_gsm8k_full_s1
runs/formal_qwen3_math500_full_s0
runs/formal_qwen3_math500_full_s1
```

每条 `05_probe_dataset.jsonl` 行代表一次已执行的 counterfactual ablation：

```text
(dataset, problem, reasoning segment, module, ratio, pruned layers)
```

当前 action bank 包含：

| Module | 含义 |
|---|---|
| `attention_heads` | 对部分 attention 输出 head 做 mask |
| `attention_block` | 将若干层的完整 attention block 输出置零 |
| `mlp_channels` | 对部分 MLP 输出通道做 mask |
| `mlp_block` | 将若干层的完整 MLP block 输出置零 |
| `layer` | 跳过若干完整 decoder layer |

每种 module 对应三种压力测试比例：

```text
0.2 / 0.4 / 0.6
```

注意：这些 action 是用于分析敏感性的 counterfactual hooks，不等价于已经实现硬件加速的真实压缩模型。

---

## 4. Action-conditioned 特征如何构造？

### 4.1 State 特征

每条数据首先包含当前 reasoning segment 对应的模型内部状态：

| 特征 | 含义 |
|---|---|
| `hidden state` | 当前 prefix 最后一个 token 的隐藏表示 |
| `entropy` | 下一 token 分布的不确定性 |
| `confidence` | 下一 token 最大概率 |
| `segment position` | 当前 segment 在整条推理链中的相对位置 |
| `dataset one-hot` | 当前样本来自 GSM8K、MATH500 或其他数据集 |

相对位置定义为：

\[
p_t = \frac{\text{segment index}}{\max(1,\text{number of segments}-1)}
\]

### 4.2 Action 特征

每个候选 action 额外编码：

| 特征 | 含义 |
|---|---|
| `module one-hot` | 剪枝模块类型 |
| `ratio` | 剪枝比例 |
| `module-weighted strength` | 模块加权强度 proxy |
| `layer multi-hot` | 哪些层被干预 |

### 4.3 Soft stage 特征

v2 同时训练两个 action-conditioned probe：

```text
action_hidden
action_hidden_stage
```

二者区别是：后者额外将自动划分的 `segment_type` 作为 one-hot 特征输入。

这是一种 **soft stage usage**：

- stage 只是风险预测特征之一；
- stage 不再直接写死剪枝上限；
- probe 可以自动学习 stage 是否有补充价值。

这比 v1 的 hard cap 更稳健。

---

## 5. 为什么必须使用 problem-level OOF？

同一道题会产生大量相关样本：

```text
多个 reasoning segments
x 多种 modules
x 多种 ratios
x 多组 layers
```

如果随机按 row 切分，训练集与验证集可能同时包含同一道题，产生信息泄漏。

v2 使用 problem-level 5-fold out-of-fold：

1. 按 `(dataset, problem id)` 分组；
2. 同一道题的所有行只能进入一个 fold；
3. 每一行的 risk score 都由“没有看过该题”的 probe 生成；
4. 最终将五个 fold 的预测拼接成完整 OOF risk table。

因此，离线 router 使用的是严格 OOF 风险分数。

---

## 6. 多模块 Router 如何选择 action？

### 6.1 剪枝强度 proxy

不同 module 的 `ratio=0.2` 不能直接视为相同计算成本。例如，跳过完整 layer 通常比 mask 部分 channel 更激进。

当前 v2 沿用 motivation 阶段的近似权重：

| Module | Strength weight |
|---|---:|
| `attention_heads` | 0.25 |
| `mlp_channels` | 0.25 |
| `attention_block` | 0.50 |
| `mlp_block` | 0.50 |
| `layer` | 1.00 |

对于动作 \(a\)：

\[
\text{strength}(a)
=
\text{module weight}(a)
\times
\text{ratio}(a)
\]

例如：

```text
mlp_channels, ratio=0.4 -> strength=0.25 x 0.4 = 0.10
mlp_block, ratio=0.4    -> strength=0.50 x 0.4 = 0.20
layer, ratio=0.4        -> strength=1.00 x 0.4 = 0.40
```

这里的 strength 是用于公平比较的 **诊断 proxy**，不是实测 FLOPs、latency 或 GPU speedup。

### 6.2 预算约束

给定全局目标预算 \(B\)，router 在每个 reasoning segment 最多选择一个 action，并满足：

\[
\frac{1}{T}
\sum_{t=1}^{T}
\text{strength}(a_t)
\le B
\]

当前评估预算：

```text
0.10 / 0.20 / 0.30
```

### 6.3 选择逻辑

离线 router 根据 OOF 预测风险优先选择低风险 action，在预算范围内为 reasoning segment 分配剪枝动作。没有被分配剪枝动作的 segment 保持 dense。

Router 采用渐进升级策略：

1. 每个 segment 从 dense 状态开始；
2. 优先考虑预测风险较低的候选 action；
3. 如果同一 segment 已经选择轻量 action，预算充足时允许升级为更强 action；
4. 每次升级只计算新增的 strength proxy；
5. 总体 strength proxy 不超过目标预算。

这是一种可解释的 training-free greedy router，不是全局最优求解器。后续 RASP-Train 可以再用可学习 router 与 budget-aware objective 替代它。

---

## 7. v2 比较哪些策略？

| Policy | 作用 |
|---|---|
| `dense_lrm` | 完全不剪枝，flip rate 必须为 0 |
| `static_best` | 在相近强度下选择全局最安全的固定 action |
| `hidden_step_mlp_block` | v1 中表现最好的简单 hidden-state 路由 |
| `hard_stage_cap_ablation` | v1 手写 stage cap，保留为消融 |
| `action_conditioned` | state + action 风险预测，多模块路由 |
| `rasp_zero_v2_soft_stage` | state + action + stage 风险预测，多模块路由 |
| `safe_step_oracle` | 直接使用真实 flip label 的理论安全上界 |

最关键的比较是：

```text
RASP-Zero v2 vs. hidden_step_mlp_block
RASP-Zero v2 vs. static_best
RASP-Zero v2 vs. safe_step_oracle
```

---

## 8. 如何运行？

在远程服务器执行：

```bash
cd /home/cike/jjy/rasp-lrm
mkdir -p logs

export PYTHON=/home/cike/jjy/envs/rasp_qwen3/bin/python
export CUDA_VISIBLE_DEVICES=2
export TOKENIZERS_PARALLELISM=false

nohup bash scripts/21_rasp_zero_offline_v2.sh \
  > logs/rasp_zero_offline_v2.log 2>&1 &
```

查看进度：

```bash
tail -f logs/rasp_zero_offline_v2.log
```

---

## 9. 输出文件

```text
runs/rasp_zero_offline_v2/
├── hidden_step_oof_scores.jsonl
├── hidden_step_oof_summary.json
├── action_conditioned_oof_scores.jsonl
├── action_conditioned_oof_summary.json
├── action_stage_conditioned_oof_scores.jsonl
├── action_stage_conditioned_oof_summary.json
├── rasp_zero_offline_v2_summary.json
├── rasp_zero_offline_v2_summary.csv
├── rasp_zero_offline_v2_selected_actions.jsonl
└── figures/
    ├── rasp_zero_v2_policy_frontier.pdf
    └── rasp_zero_v2_policy_frontier.png
```

---

## 10. 如何判断实验是否成功？

### 10.1 Probe 层面

首先查看：

```text
action_conditioned_oof_summary.json
action_stage_conditioned_oof_summary.json
```

希望看到：

```text
action-conditioned ROC-AUC > hidden-only ROC-AUC
```

如果加入 soft stage 后继续提升，说明自动 stage 特征具有额外价值。

### 10.2 Policy 层面

查看：

```text
rasp_zero_offline_v2_summary.csv
figures/rasp_zero_v2_policy_frontier.pdf
```

在相近 `average_pruning_strength_proxy` 下，希望：

```text
RASP-Zero v2 flip rate < hidden-step MLP flip rate
RASP-Zero v2 flip rate < static best flip rate
RASP-Zero v2 接近 safe step oracle
```

同时需要查看：

| 字段 | 含义 |
|---|---|
| `budget_utilization` | 实际使用的 strength proxy 与目标预算之比 |
| `module_distribution` | Router 最终选择了哪些模块 |
| `choice_distribution` | Router 最终选择了哪些具体 action |
| `average_selected_predicted_risk` | 被选动作的平均预测风险 |

如果某个策略 flip rate 很低，但 `budget_utilization` 也很低，则不能直接宣称其更优秀：它可能只是没有充分使用剪枝预算。

### 10.3 应当如实报告的可能结果

如果 `action_conditioned` 优于 `rasp_zero_v2_soft_stage`，说明 rule-based stage assignment 噪声较大，最终方法可以保留 reasoning-state-conditioned router，但不显式使用 stage。

如果两者都没有优于 hidden-only，说明当前线性 probe 或 action bank 不够强，需要继续改进 estimator 或候选 mask，而不是直接进入运行时实现。

---

## 11. 与未来运行时 RASP 的关系

v2 仍然是离线策略实验。它的目标是验证：

> reasoning state 与结构 action 联合建模，是否比静态剪枝和粗粒度 step-risk 路由更安全。

只有 v2 成立后，才进入真实运行时实现。运行时第一版建议使用：

```text
MLP intermediate channel masks
ratio = 0 / 0.05 / 0.10 / 0.20
每 16 或 32 个生成 token 更新一次 router
高风险窗口恢复 dense
```

Attention、完整 layer skip、逐 token 路由和训练式 RASP-Train 暂时留到后续扩展。
