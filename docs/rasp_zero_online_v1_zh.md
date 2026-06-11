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

正式银行默认拆成 20 个 shard，每个 50 题。GPU 数量可以配置。八卡采集：

```bash
export PYTHON=/home/cike/jjy/envs/rasp_qwen3/bin/python
export RASP_BANK_GPU_COUNT=8
bash scripts/30_collect_runtime_bank_formal.sh
```

采集结果：

```text
runs/03_rasp_zero/02_runtime_banks/rasp_zero_runtime_bank_formal/
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

还有一类样本会在训练前自动过滤：如果 `ratio=0.00` 也导致答案翻转，说明这个 step 即使不剪枝，
仅从中间前缀续写也已经不稳定。这不是剪枝风险，而是 continuation protocol 的噪声。准备
router 数据时会整组删除这些 step，避免把“无剪枝也错”的样本误当成剪枝风险。

全部 shard 完成后，准备数据并训练 router：

```bash
export PYTHON=/home/cike/jjy/envs/rasp_qwen3/bin/python
bash scripts/27_train_runtime_router.sh
```

关键输出：

```text
runs/03_rasp_zero/03_runtime_router/rasp_zero_runtime_router/
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

## 11. 当前正式 Router Bank 与 Router 训练结果

本节记录当前已经跑完的一版正式 RASP-Zero Online v1 结果，来源为：

```text
runs/03_rasp_zero/02_runtime_banks/rasp_zero_runtime_bank_formal/
runs/03_rasp_zero/03_runtime_router/rasp_zero_runtime_router/
```

这部分结果不是最终在线 benchmark，而是在线 RASP-Zero 前置的训练银行与第一版风险 router
结果。它回答的问题是：

> 我们是否已经有足够稳定的数据，让一个轻量 router 学会预测“当前 reasoning state 下某个剪枝
> action 是否危险”？

### 11.1 Runtime Bank 规模

正式采集设定为：

```text
GSM8K train[:500]
rasbt/math_full_minus_math500 train[:500]
```

共计 `1000` 道训练来源题目。Dense Qwen3-1.7B 在这些题上答对 `801` 道，因此只有这 `801`
道题进入反事实训练银行。这个过滤很重要：如果 dense 本身已经错了，后续无法定义“剪枝导致答案
翻转”，因为 baseline answer 已经不是可靠参考。

最终统计为：

```text
problem_count = 801
problem_step_count = 3479
counterfactual_rows_with_dense_control = 24353
action_conditioned_risk_rows = 20874
```

解释如下：

- `problem_count`：进入训练银行的 dense-correct 题目数。
- `problem_step_count`：这些题自动切分出的 reasoning steps 数。
- `counterfactual_rows_with_dense_control`：包含 `ratio=0.00` 控制组的所有反事实 rows。
- `action_conditioned_risk_rows`：去掉 `ratio=0.00` 后真正用于训练 risk router 的 rows。

由于每个 step 有 6 个非零候选剪枝率：

```text
0.02, 0.05, 0.10, 0.20, 0.30, 0.40
```

所以：

```text
3479 steps x 6 actions = 20874 risk rows
```

### 11.2 ratio=0 不稳定 step 的过滤

采集过程中发现：

```text
unstable_ratio_zero_step_count = 58
rows_filtered_by_unstable_ratio_zero = 406
```

这里的 `ratio=0.00` 表示不施加剪枝，仅从当前 reasoning prefix 继续生成。如果 `ratio=0.00`
也导致最终答案翻转，说明问题不在剪枝，而在 continuation protocol 本身：模型从这个中间前缀
重新续写时已经不稳定。

因此这些 step 不能用来训练剪枝风险模型。否则 router 会把“无剪枝也错”的现象误学成“剪枝导致
错误”。当前流程会整组删除这些 step。过滤后，`ratio=0.00` 的翻转率被清零：

```text
ratio=0.00 flip rate = 0.0
```

这个处理让后续风险标签更干净。

### 11.3 剪枝率与答案翻转风险

过滤后，不同剪枝率对应的答案翻转率为：

```text
ratio=0.02  flip rate = 0.0325
ratio=0.05  flip rate = 0.0512
ratio=0.10  flip rate = 0.0960
ratio=0.20  flip rate = 0.2084
ratio=0.30  flip rate = 0.3352
ratio=0.40  flip rate = 0.4544
```

这是一个很重要的健康信号：随着剪枝率增大，答案翻转风险整体单调上升。它说明训练银行不是随机
噪声，而是包含明确的 action-risk 关系。

这也支持 RASP-Zero 的基本假设：

> 剪枝动作的风险不是固定的，但剪枝强度越高，整体风险越大；router 需要在具体 reasoning state
> 下选择尽可能大的安全剪枝率。

### 11.4 不同 reasoning stage 的风险差异

按自动 stage assignment 统计，当前 risk rows 的正例率为：

```text
understanding  flip rate = 0.3974
planning       flip rate = 0.3680
derivation     flip rate = 0.2248
verification   flip rate = 0.2056
final          flip rate = 0.0159
```

这个结果有两个含义。

第一，早期阶段非常敏感。`understanding` 和 `planning` 的翻转率最高，说明题意理解和解题规划
阶段一旦被扰动，后续整条推理链容易发生 drift。

第二，`final` 阶段风险最低。这可能是因为到 final step 时，答案已经在前文中基本确定，模型更多
是在输出格式化答案。因此在当前 setup 下，final segment 对 MLP channel mask 的敏感性反而较低。

需要注意：这里的 stage 是 rule-based heuristic，不是人工标注，也不是 LLM classifier。因此
它适合作为 motivation 和诊断信号，但论文中应明确说明“automatic stage assignment”，并最好
抽样人工检查 20-50 条。

### 11.5 Safe Oracle 说明了动态剪枝空间

Safe oracle 对每个 reasoning step 统计：

```text
max_safe_ratio = 在不翻转答案的前提下，该 step 可承受的最大剪枝率
```

当前 `max_safe_ratio` 分布为：

```text
0.00: 34
0.02: 55
0.05: 125
0.10: 356
0.20: 470
0.30: 541
0.40: 1898
```

这说明在 `3479` 个 step 中，有 `1898` 个 step 在离线观察中可以安全剪到 `40%`，但也有少数
step 几乎不能剪。这正是动态剪枝的空间：

> 固定剪枝率无法同时照顾“可大胆剪”的 step 和“必须保守”的 step。

更保守的 `monotonic_safe_ratio` 分布为：

```text
0.00: 113
0.02: 116
0.05: 193
0.10: 420
0.20: 489
0.30: 430
0.40: 1718
```

`monotonic_safe_ratio` 要求从 `0.00` 到该 ratio 的所有更小 ratio 都安全，因此比
`max_safe_ratio` 更严格。即便在这个严格口径下，仍有大量 step 能承受 `0.30` 或 `0.40`。

### 11.6 Budget-aware Safe Oracle

Budget-aware safe oracle 在给定平均剪枝预算 `B` 的前提下，为每个 step 选择安全动作，并尽量
用满预算。

当前结果显示，三个预算都几乎被精确利用：

```text
B=0.05  utilization ≈ 1.000
B=0.10  utilization ≈ 1.000
B=0.20  utilization ≈ 1.000
```

例如在 `max_safe_ratio` 口径下，`B=0.20` 时的选择分布为：

```text
0.00: 34
0.02: 55
0.05: 125
0.10: 356
0.20: 2199
0.30: 710
```

这说明，如果存在一个理想 router，它可以在平均 `20%` 剪枝预算下，把大量 step 分配到较高剪枝率，
同时对少数风险 step 保持低剪枝。这给在线 RASP-Zero 提供了上限参考。

### 11.7 第一版 Action-conditioned Risk Router

第一版 router 使用 problem-level split 训练。训练/验证划分为：

```text
train_problem_count = 601
val_problem_count = 200
train_rows = 15702
val_rows = 5172
```

验证集正例率为：

```text
positive_rate_val = 0.2071
```

Router 指标为：

```text
ROC-AUC = 0.8172
PR-AUC  = 0.5357
val_loss = 0.4092
```

解释：

- `ROC-AUC = 0.8172` 表示 router 能较好地区分高风险 action 和低风险 action。
- `PR-AUC = 0.5357` 明显高于验证集正例率 `0.2071`，说明它不是随机猜测。
- `problem-level split` 表明同一道题的所有 segment/action rows 不会同时出现在训练和验证中，
  因此验证指标比 row-level split 更可信。

这个结果说明第一版 router 已经学到了可用的风险信号，可以进入在线 smoke evaluation。

### 11.8 当前阶段结论

当前已经完成：

```text
RASP-Zero runtime bank 构造
ratio=0 不稳定 step 过滤
action-conditioned risk dataset 构造
problem-level risk router 训练
safe oracle / budget-aware oracle 分析
```

当前结果支持下一步：

```text
在线 RASP-Zero smoke evaluation
```

也就是让 router 在真实生成过程中每隔一段 token 观察当前 state，预测候选剪枝 action 的风险，
并动态选择当前可接受的最大剪枝率。

建议下一步先运行：

```bash
export CUDA_VISIBLE_DEVICES=0
bash scripts/28_eval_rasp_zero_runtime_router_smoke.sh
```

如果 GSM8K-2 和 MATH500-2 的 dense control 与 router 输出都正常，再扩大到：

```text
GSM8K-100 + MATH500-100
```

最后再考虑完整测试集。

### 11.9 Online policy calibration

第一次在线 smoke 使用：

```text
risk_threshold = 0.35
target_average_ratio = 0.20
ratios = [0.02, 0.05, 0.10, 0.20, 0.30, 0.40]
```

这证明在线闭环已经跑通，但在 GSM8K Janet 样例中出现了 dense-correct 转 router-wrong：

```text
dense:  16 - 3 - 4 = 9, 9 * 2 = 18
router: 16 - 3 = 13, 13 * 2 = 26
```

这说明当前策略对部分关键推理窗口偏激进。一个直接做法是降低最大 ratio，但这样会削弱方法的效率
空间。更合理的校准方式是：

```text
保留 0.40 候选动作
降低 risk_threshold 或 target_average_ratio
让 router 只在预测风险足够低、预算足够宽裕时才使用高 ratio
```

当前准备先做四组校准：

```text
A: threshold=0.25, target=0.10, ratios=[0.02,0.05,0.10,0.20,0.30,0.40]
B: threshold=0.30, target=0.10, ratios=[0.02,0.05,0.10,0.20,0.30,0.40]
C: threshold=0.30, target=0.15, ratios=[0.02,0.05,0.10,0.20,0.30,0.40]
D: threshold=0.35, target=0.15, ratios=[0.02,0.05,0.10,0.20,0.30,0.40]
```

运行：

```bash
export CUDA_VISIBLE_DEVICES=0
bash scripts/31_eval_rasp_zero_online_calibration.sh
```

该脚本会生成 `GSM8K-20 + MATH500-20` 的小型 calibration，并输出：

```text
runs/03_rasp_zero/04_online_eval/rasp_zero_online_calibration/summary.csv
```

主要观察：

- accuracy 是否接近 dense control；
- average pruning ratio 是否达到可见幅度；
- 是否仍出现明显 dense-correct 到 router-wrong 的逻辑遗漏；
- `0.40` 是否被少量、合理地使用，而不是频繁压到关键推理窗口。

### 11.10 Conservative RASP-Zero v1

当前选择继续路线 A：把 RASP-Zero Online v1 做成一个更安全的在线策略，而不是立刻跳到
RASP-Train。

核心思想是：

```text
保留 0.40 的动作空间
但在早期推理和不确定状态中限制最大 ratio
```

也就是说，`0.40` 仍然可以被使用，但不能在所有窗口中自由使用。当前新增三类 cap：

```text
early_tokens / early_max_ratio
    前若干 generated tokens 的最大剪枝率。

high_entropy_threshold / high_entropy_max_ratio
    下一 token 分布不确定时的最大剪枝率。

low_confidence_threshold / low_confidence_max_ratio
    最大 token 概率较低时的最大剪枝率。
```

一个候选 ratio 在线被选中，必须同时满足：

```text
ratio <= available_budget
ratio <= conservative_cap
predicted_risk <= risk_threshold
```

下一轮 conservative calibration 包含：

```text
e64cap010_thr025_tgt010
e96cap010_thr025_tgt010
e96cap020_thr025_tgt010
e96cap010_uncertain_thr025_tgt010
```

运行：

```bash
export CUDA_VISIBLE_DEVICES=0
bash scripts/33_eval_rasp_zero_online_conservative_calibration.sh
```

输出：

```text
runs/03_rasp_zero/04_online_eval/rasp_zero_online_conservative_calibration/summary.csv
```

如果这一轮能让 GSM8K-20 的 accuracy 明显接近 dense，同时保留约 `5%-10%` 的理论 MLP reduction，
则可以扩大到 `GSM8K-100 + MATH500-100`。如果仍然不稳，则 RASP-Zero 更适合定位为分析型
prototype，后续应转向 RASP-Train。
