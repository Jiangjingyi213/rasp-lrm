# RASP-Train：从 Ratio Imitation v1 到 Action-Risk v2

## 1. Motivation 特征与在线判断的关系

Motivation 阶段比较过以下信号能否预测 counterfactual pruning risk：

- `entropy`：下一 token 分布熵；
- `confidence`：下一 token 最大概率；
- `activation summary`：层输出的 norm / mean / std / max；
- `hidden state`：当前 prefix 最后 token 的隐状态；
- `combined`：组合上述特征；
- `linear probe score`：线性 probe 输出的风险概率。

实验表明 entropy/confidence 只有较弱信号，activation 有部分信号，hidden state 明显更强，
combined 没有稳定超过 hidden state。因此后续方法没有把 entropy 当作唯一在线规则：

```text
Motivation:
  比较哪些 state features 包含 pruning-risk signal

RASP-Zero:
  hidden + entropy + confidence + action
  -> linear action-risk probe
  -> threshold + budget rule

RASP-Train v1:
  hidden + entropy + confidence + budget state
  -> oracle ratio classification

RASP-Train v2:
  hidden + entropy + confidence + budget state + candidate ratio
  -> nonlinear action-risk prediction
  -> calibrated threshold + causal budget controller
```

Entropy 和 confidence 在 v2 中仍是辅助特征，主要信息来源仍是 hidden state。Activation summary
暂未进入在线 v2，因为逐窗口采集多层 activation 会增加 hook 和同步开销，而 motivation 中 hidden
state 已更强。Activation 可作为后续消融，不能在没有开销收益验证时强行加入。

## 2. RASP-Train v1 结果

v1 已完成离线实验，结果保存在：

```text
runs/rasp_train_v1/
```

| 方法 | 平均 ratio | Flip rate |
|---|---:|---:|
| RASP-Train v1 B15 | 0.1241 | 0.1023 |
| RASP-Zero matched B15 | 0.1400 | 0.0621 |
| RASP-Train v1 B20 | 0.1761 | 0.1368 |
| RASP-Zero matched B20 | 0.1849 | 0.0851 |

v1 没有通过离线门槛，不进入在线 smoke。它保留为失败消融：

> 单一 oracle-ratio imitation 不如直接学习 action risk。

原因包括：

1. 同一个 step 可能有多个安全 ratio，单标签 CE 只承认一个 oracle action。
2. Oracle label 同时受安全上限与预算历史影响，分类器容易学习多数 ratio。
3. Batch-average budget loss 与真实 per-problem causal budget 不一致。
4. v1 没有直接利用七个候选 ratio 的完整 counterfactual labels。

## 3. RASP-Train v2

v2 对每个 candidate ratio 输出 unsafe probability：

```text
ratios = [0.00, 0.02, 0.05, 0.10, 0.20, 0.30, 0.40]
q(s_t, r) = P(unsafe | reasoning state s_t, candidate ratio r)
```

State features：

- last-token hidden state；
- entropy；
- confidence；
- reasoning position；
- dataset/domain；
- target budget；
- current available prefix budget。

Action features：

- ratio；
- ratio squared。

网络结构：

```text
state -> LayerNorm -> MLP state encoder
[state embedding; ratio; ratio^2] -> nonlinear action head -> unsafe logit
```

Qwen3 本体仍冻结，只训练轻量 action-risk policy。

## 4. Multi-Label Loss

每个 step 使用完整标签：

```text
candidate_unsafe = [safe, safe, safe, unsafe, unsafe, unsafe, unsafe]
```

`candidate_unsafe` 使用 monotonic-safe 定义。若较小 ratio 已翻转，即使某个更大 ratio 偶然未翻转，
更大 ratio 仍保守地视为 unsafe。原始 `candidate_flipped` 保留用于报告真实 flip rate。

```text
L = weighted BCE
  + lambda_monotonic * monotonic penalty
  + lambda_rank * safe/unsafe ranking loss
```

- weighted BCE：学习每个 ratio 是否 unsafe；
- monotonic penalty：约束高 ratio 的风险不低于低 ratio；
- ranking loss：要求 unsafe action 风险高于 safe action；
- 移除 batch-level budget loss。

预算由 controller 硬约束：

```text
available_t = B * (t + 1) - sum(previous_selected_ratios)
```

## 5. Problem Split 与 Threshold Calibration

v2 使用严格 problem-level 三段划分：

```text
70% train
15% calibration
15% test
```

- train：拟合 action-risk network；
- calibration：选择最佳 epoch，并校准 risk threshold；
- test：只用于最终离线报告。

阈值候选为 `0.01 ... 0.50`。B15 默认要求 calibration flip/unsafe rate 不超过 `0.06/0.08`，
B20 默认不超过 `0.08/0.10`。在满足条件的阈值中选择平均 ratio 最大者；若没有阈值满足目标，
则选择 calibration flip rate 最低的阈值。

最终 checkpoint 保存 calibrated threshold，在线 controller 默认直接加载。不能用 test problems
调阈值。`13_rasp_train_metrics.json` 还会记录 `calibrated_selection` 和
`calibration_constraints_satisfied`；若后者为 `false`，不得把该 checkpoint 视为通过安全门槛。

## 6. 离线与在线选择

每个决策窗口：

1. 根据历史动作计算 available prefix budget；
2. 预测全部 candidate ratio 的 unsafe probability；
3. 排除超过 budget/cap 的动作；
4. 排除风险高于 calibrated threshold 的非零动作；
5. 在剩余动作中选择最大 ratio；
6. 若没有安全非零动作，选择 ratio=0。

离线评估和在线 controller 使用同一选择逻辑。

## 7. 输出和命令

v1 结果保留在：

```text
runs/rasp_train_v1/
```

v2 默认写入：

```text
runs/rasp_train_v2/
```

脚本名称暂时沿用 `v1`，避免大范围入口变更：

```bash
export PYTHON=/home/cike/jjy/envs/rasp_qwen3/bin/python
mkdir -p logs
nohup env CUDA_VISIBLE_DEVICES=0 PYTHON="$PYTHON" bash -c '
set -e
bash scripts/35_prepare_rasp_train_v1_data.sh
bash scripts/36_train_rasp_train_v1.sh
bash scripts/37_eval_rasp_train_v1_offline.sh
' > logs/rasp_train_v2_offline.log 2>&1 &
echo $! > logs/rasp_train_v2_offline.pid
tail -f logs/rasp_train_v2_offline.log
```

重点检查：

```text
runs/rasp_train_v2/b15/13_rasp_train_metrics.json
runs/rasp_train_v2/b15/offline_eval/12_rasp_train_offline_summary.csv
runs/rasp_train_v2/b20/13_rasp_train_metrics.json
runs/rasp_train_v2/b20/offline_eval/12_rasp_train_offline_summary.csv
```

离线门槛：

- B15/B20 test flip rate 低于对应 RASP-Zero；
- conservative unsafe rate 同步下降；
- 不能通过接近 ratio=0 换取低 flip；
- calibration 与 test problems 严格隔离。
- `calibration_constraints_satisfied` 为 `true`。

离线通过后才运行：

```bash
bash scripts/38_eval_rasp_train_v1_online_smoke.sh
```

在线 smoke 会先运行 paired ratio-0 control，并输出 `14_paired_dense_comparison.json`。

## 8. 当前边界

- Bank state 来自 dense segment boundary，在线按 fixed token window 更新，仍有分布差异。
- 在线剪枝会改变后续 hidden state，离线 risk prediction 不能完全消除 reasoning drift。
- 当前只控制 MLP intermediate-channel ratio，尚未实现 attention/layer multi-module routing。
- Logical mask 仍执行 dense projection，只能报告 activated-channel proxy，不能宣称 wall-clock speedup。
