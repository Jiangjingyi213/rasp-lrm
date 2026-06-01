# 在线 RASP-Zero Runtime v0：实现说明与验收流程

## 1. 当前阶段要解决什么问题？

离线 `RASP-Zero Offline v2` 已经验证：

> 在相同的平均剪枝强度下，根据当前 reasoning state 和候选剪枝 action 联合预测风险，比统一静态剪枝更加安全。

但是，离线实验中的 action bank 主要用于分析结构敏感性。它包含：

```text
layer
attention_block
attention_heads
mlp_block
mlp_channels
```

这些诊断动作并不都适合直接搬进在线推理系统。例如，Qwen3 使用 GQA attention，动态切换 attention heads 需要额外处理 query heads 与 KV heads 的映射；运行时频繁跳过完整 layer 也可能引入较高的工程风险。

因此，在线 Runtime v0 首先聚焦一个更明确、更容易公平评估的动作：

```text
mlp_intermediate_channels
```

它表示：在 Qwen3 FFN 内部，仅保留一部分 intermediate neurons。

---

## 2. 为什么新增 `mlp_intermediate_channels`？

旧 motivation 实验中的 `mlp_channels` hook 位于完整 MLP 的输出之后：

```text
MLP 完整计算
    ↓
输出 hidden states
    ↓
将部分输出维度置零
```

它适合做敏感性诊断，但无法减少 FFN 内部矩阵乘法的计算量。

真正可部署的 FFN channel pruning 应发生在：

```text
hidden states
    ↓
gate_proj / up_proj
    ↓
intermediate neurons
    ↓
down_proj
    ↓
MLP output
```

Qwen3-1.7B 每层 FFN 的 intermediate size 为 `6144`。如果剪掉 10% intermediate neurons，那么理论上可以同时减少：

```text
gate_proj 输出通道
up_proj 输出通道
down_proj 输入通道
```

对应的 FFN 矩阵乘法成本也近似减少 10%。

旧 `mlp_channels` 结果仍然保留，用于解释 motivation；新增 `mlp_intermediate_channels` 用于在线部署。两者不能混用。

---

## 3. Runtime v0 的工作流

Runtime v0 使用 prompt-conditioned、reasoning-process-aware 的动态 MLP mask：

```text
1. 输入完整问题 prompt
2. Prefill 阶段始终使用 Dense MLP
3. 根据 prefill 激活，为每层 FFN intermediate neurons 建立重要性排序
4. 开始逐 token greedy decode
5. 根据当前历史上下文选择下一窗口使用的 ratio
6. Decode 阶段对 intermediate neurons 应用嵌套 mask
7. 每生成固定数量 token 后重新调用 controller
```

默认动作空间：

```text
dense
5% intermediate-channel pruning
10% intermediate-channel pruning
20% intermediate-channel pruning
```

默认窗口：

```text
16 generated tokens
```

---

## 4. 激活排序如何计算？

设某一层 prefill 阶段的 FFN intermediate activation 为：

\[
H \in \mathbb{R}^{T \times D}
\]

其中：

- \(T\)：prefix token 数；
- \(D\)：FFN intermediate neurons 数量。

对每个 token 的 intermediate activation 做 L2 归一化：

\[
\widetilde{H}_{t,:}
=
\frac{H_{t,:}}{\lVert H_{t,:} \rVert_2 + \epsilon}
\]

然后沿 token 维度累计每个 neuron 的响应：

\[
s_j
=
\sqrt{
\sum_{t=1}^{T}
\widetilde{H}_{t,j}^2
}
\]

按照 \(s_j\) 从大到小排序，得到每层 neuron ranking。

这与本仓库已有 GRIFFIN-Qwen3 adapter 的序列级激活排序保持一致。

---

## 5. 为什么 mask 必须嵌套？

为了让剪枝强度变化稳定，轻量 mask 与重度 mask 使用同一条 neuron ranking：

```text
keep-80% ⊂ keep-90% ⊂ keep-95% ⊂ keep-100%
```

例如：

```text
5% pruning  -> 保留 top 95%
10% pruning -> 保留 top 90%
20% pruning -> 保留 top 80%
```

这样，当 controller 从 5% 升级到 10% 时，只会继续移除排名较低的神经元，不会完全更换一套不相关的 mask。

---

## 6. 当前 v0 为什么还不能宣称真实加速？

当前后端名称为：

```text
logical_mask_v0
```

它执行：

```python
intermediate = act_fn(gate_proj(x)) * up_proj(x)
intermediate = intermediate * mask
output = down_proj(intermediate)
```

虽然数学干预位置正确，但三个 Linear 层仍然使用原始矩阵尺寸。因此：

- 可以验证动态剪枝是否保持答案质量；
- 可以统计理论 activated-parameter reduction；
- 可以统计理论 FFN FLOPs reduction；
- **不能**将 wall-clock latency 变化解释为真实加速。

下一阶段 `Runtime v1` 会参考 `src/baselines/griffin_qwen3.py`，将保留通道抽取为：

```text
gate_proj_reduced
up_proj_reduced
down_proj_reduced
```

只有 v1 才能正式报告 tokens/s 与 latency speedup。

---

## 7. 在线 Controller 当前做到哪一步？

Runtime v0 已经实现统一 controller 接口：

```python
choose_ratio(observation) -> ratio
```

每次调用时，controller 只能读取已经存在的历史信息：

```text
generated token 数量
当前 next-token entropy
当前 next-token confidence
当前 hidden state
```

它不能读取：

```text
未来 token
完整 reasoning segment
counterfactual 标签
最终答案
```

当前提供两个 controller：

| Controller | 用途 |
|---|---|
| `fixed` | 验证 dense equivalence 和不同静态 ratio |
| `confidence_threshold` | 验证动态切换 plumbing，不作为最终方法 |

正式 action-conditioned risk router 尚未接入。必须先重新采集 deployment-aligned counterfactual bank，再训练对应 probe。

---

## 8. Deployment-aligned Counterfactual Bank

新增 action：

```text
mlp_intermediate_channels
```

在每个 reasoning prefix 上执行：

```text
1. Dense prefill 建立 neuron ranking
2. 使用 5% mask 继续生成答案
3. 使用 10% mask 继续生成答案
4. 使用 20% mask 继续生成答案
5. 分别记录是否导致答案翻转
```

每一条 probe row 将包含：

```text
dataset
problem id
segment id
segment type
hidden state
entropy
confidence
ratio
pruned layers
flipped
```

这批数据才适合训练 Runtime RASP-Zero 的 action-conditioned router。

---

## 9. 新增文件

### Runtime 核心

```text
src/rasp/activation_ranker.py
src/rasp/mlp_runtime.py
src/rasp/budget_controller.py
src/rasp/greedy_decode.py
src/rasp/metrics.py
```

### 入口

```text
src/main_eval_rasp_zero_runtime.py
```

### Counterfactual 接口

```text
src/pruning/mlp_pruner.py
src/pruning/contexts.py
src/main_counterfactual_prune.py
```

### 配置与脚本

```text
configs/exp_rasp_zero_runtime_smoke.yaml
configs/exp_rasp_zero_runtime_bank_gsm8k_smoke.yaml
scripts/23_collect_runtime_counterfactuals.sh
scripts/25_eval_rasp_zero_runtime_smoke.sh
```

---

## 10. 远程验收流程

### 10.1 静态检查与单元测试

```bash
cd /home/cike/jjy/rasp-lrm
export PYTHON=/home/cike/jjy/envs/rasp_qwen3/bin/python

$PYTHON -X pycache_prefix=/tmp/rasp_lrm_pycache -m compileall -q src tests
$PYTHON -m unittest discover -s tests -v
```

预期：

```text
全部测试通过
```

### 10.2 Dense-equivalence smoke

```bash
cd /home/cike/jjy/rasp-lrm
export CUDA_VISIBLE_DEVICES=2
export TOKENIZERS_PARALLELISM=false
export PYTHON=/home/cike/jjy/envs/rasp_qwen3/bin/python

$PYTHON -m src.main_eval_rasp_zero_runtime \
  --config configs/exp_rasp_zero_runtime_smoke.yaml
```

配置中：

```text
controller: fixed
fixed_ratio: 0.0
```

目标：

```text
Runtime v0 ratio=0 的输出与 dense Qwen3 保持一致。
```

结果：

```text
runs/rasp_zero_runtime_smoke_dense/00_runtime_summary.json
runs/rasp_zero_runtime_smoke_dense/01_trajectories.jsonl
```

### 10.3 Deployment-aligned bank smoke

```bash
cd /home/cike/jjy/rasp-lrm
export CUDA_VISIBLE_DEVICES=2
export TOKENIZERS_PARALLELISM=false
export PYTHON=/home/cike/jjy/envs/rasp_qwen3/bin/python

bash scripts/23_collect_runtime_counterfactuals.sh
```

结果：

```text
runs/rasp_zero_runtime_bank_gsm8k_smoke/01_trajectories.jsonl
runs/rasp_zero_runtime_bank_gsm8k_smoke/02_segments.jsonl
runs/rasp_zero_runtime_bank_gsm8k_smoke/03_counterfactuals.jsonl
runs/rasp_zero_runtime_bank_gsm8k_smoke/05_probe_dataset.jsonl
runs/rasp_zero_runtime_bank_gsm8k_smoke/05_probe_hidden_states.pt
```

重点检查：

```bash
head -n 3 runs/rasp_zero_runtime_bank_gsm8k_smoke/03_counterfactuals.jsonl
wc -l runs/rasp_zero_runtime_bank_gsm8k_smoke/*.jsonl
```

每条 counterfactual row 应满足：

```text
module = mlp_intermediate_channels
ratio in {0.05, 0.10, 0.20}
```

---

## 11. 下一阶段

通过 Runtime v0 smoke 后，继续实现：

```text
1. GSM8K + MATH500 deployment-aligned bank 配置
2. problem-level OOF action-conditioned router
3. Runtime Router 加载与在线推理
4. Runtime v1 reduced-weight 后端
5. Dense / Static / GRIFFIN / FLAP-MLP / RASP-Zero 公平评估
```

论文主结果必须区分：

| 结果类型 | 可以报告什么 |
|---|---|
| Runtime v0 logical mask | Accuracy、answer flip、理论 FFN FLOPs |
| Runtime v1 reduced weights | Accuracy、latency、tokens/s、真实 speedup |

这条区分需要始终保持。
