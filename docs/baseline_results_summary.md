# Baseline 结果整理：Dense Qwen3 / GRIFFIN / FLAP-MLP

本文档整理目前已经跑完、可以用于论文或周报讨论的 baseline 结果。重点区分：

- **正式全集结果**：可以进入正式 baseline 表格。
- **smoke / ablation / diagnostic 结果**：只用于调试和方法选择，不建议进入正式对比表。

当前主模型为 `Qwen/Qwen3-1.7B`，评测数据集为 GSM8K 与 MATH500。

## 1. Dense Qwen3 基线

Dense Qwen3 表示**不进行任何剪枝**的原始模型，是后续所有剪枝方法的主要对照。

正式结果位置：

- `runs/02_baselines/eval_dense_qwen3_gsm8k_budget`
- `runs/02_baselines/eval_dense_qwen3_math500_budget`

这两个目录的结果与 motivation 实验中分片跑出的 dense 结果一致：

- `runs/01_motivation/formal_qwen3_gsm8k_full_s0` + `runs/01_motivation/formal_qwen3_gsm8k_full_s1`
- `runs/01_motivation/formal_qwen3_math500_full_s0` + `runs/01_motivation/formal_qwen3_math500_full_s1`

| 方法 | 数据集 | 样本数 | 答对数 | 准确率 |
|---|---:|---:|---:|---:|
| Dense Qwen3 | GSM8K | 1319 | 1062 | 80.52% |
| Dense Qwen3 | MATH500 | 500 | 280 | 56.00% |
| Dense Qwen3 | 合并 | 1819 | 1342 | 73.78% |

**解释：**

Dense Qwen3 在 GSM8K 上有 80.52% 的准确率，在 MATH500 上有 56.00% 的准确率。这说明 Qwen3-1.7B 本身具备足够的数学推理能力，可以作为研究 LRM 剪枝风险的基础模型。后续所有剪枝方法的准确率下降，都应该主要和这个 dense baseline 对比。

## 2. GRIFFIN-style FFN Dynamic Pruning

GRIFFIN-style baseline 是一种**动态 FFN 剪枝**方法。它在生成过程中根据 FFN neuron / expert 的激活模式选择保留一部分 FFN 中间神经元。这里的实现是我们对 Qwen3 MLP 的适配版本，用来模拟 GRIFFIN 的 sequence-level FFN dynamic pruning 思路。

正式结果位置：

- `runs/02_baselines/eval_griffin_p20_qwen3_gsm8k_budget`
- `runs/02_baselines/eval_griffin_p20_qwen3_math500_budget`
- `runs/02_baselines/eval_griffin_p40_qwen3_gsm8k_budget`
- `runs/02_baselines/eval_griffin_p40_qwen3_math500_budget`
- `runs/02_baselines/eval_griffin_p60_qwen3_gsm8k_budget`
- `runs/02_baselines/eval_griffin_p60_qwen3_math500_budget`

其中：

- `p20` 表示约 20% FFN pruning。
- `p40` 表示约 40% FFN pruning。
- `p60` 表示约 60% FFN pruning。

| 方法 | 数据集 | 样本数 | 答对数 | 准确率 | 相对 Dense 下降 |
|---|---:|---:|---:|---:|---:|
| GRIFFIN p20 | GSM8K | 1319 | 520 | 39.42% | -41.09 |
| GRIFFIN p20 | MATH500 | 500 | 127 | 25.40% | -30.60 |
| GRIFFIN p20 | 合并 | 1819 | 647 | 35.57% | -38.20 |
| GRIFFIN p40 | GSM8K | 1319 | 89 | 6.75% | -73.77 |
| GRIFFIN p40 | MATH500 | 500 | 35 | 7.00% | -49.00 |
| GRIFFIN p40 | 合并 | 1819 | 124 | 6.82% | -66.96 |
| GRIFFIN p60 | GSM8K | 1319 | 28 | 2.12% | -78.39 |
| GRIFFIN p60 | MATH500 | 500 | 19 | 3.80% | -52.20 |
| GRIFFIN p60 | 合并 | 1819 | 47 | 2.58% | -71.19 |

Dense 原本答对、剪枝后答错的数量如下：

| 方法 | 数据集 | Dense 答对但剪枝后答错 | Dense 答错但剪枝后答对 |
|---|---:|---:|---:|
| GRIFFIN p20 | GSM8K | 570 | 28 |
| GRIFFIN p20 | MATH500 | 167 | 14 |
| GRIFFIN p40 | GSM8K | 979 | 6 |
| GRIFFIN p40 | MATH500 | 250 | 5 |
| GRIFFIN p60 | GSM8K | 1039 | 5 |
| GRIFFIN p60 | MATH500 | 271 | 10 |

**解释：**

GRIFFIN-style FFN dynamic pruning 虽然是动态剪枝，但它不是 reasoning-risk-aware。也就是说，它主要依据序列层面的 FFN 激活模式做 neuron selection，而不会判断当前 reasoning step 是否处在关键推理阶段。

结果显示，`p20` 已经从 Dense 合并准确率 73.78% 降到 35.57%。这说明单纯的 sequence-level FFN 动态剪枝对 reasoning tasks 仍然很危险，尤其是在推理链较长、局部步骤容易出错的情况下。这一结果可以支持我们的核心动机：LRM 剪枝不能只看 FFN 激活复用，还需要识别 reasoning process 内部哪些 step / stage 是高风险的。

## 3. FLAP-MLP Static Pruning

FLAP-MLP 是一种**静态结构化剪枝 baseline**。我们在 Qwen3 上实现的是 MLP-only 版本，目标是剪掉 MLP intermediate channels。

正式结果位置：

- `runs/02_baselines/flap_mlp_formal/eval_flap_mlp_wifn_ulum_wikitext2_nobias_p05_qwen3_gsm8k_budget`
- `runs/02_baselines/flap_mlp_formal/eval_flap_mlp_wifn_ulum_wikitext2_nobias_p05_qwen3_math500_budget`
- `runs/02_baselines/flap_mlp_formal/eval_flap_mlp_wifn_ulum_wikitext2_nobias_p20_qwen3_gsm8k_budget`
- `runs/02_baselines/flap_mlp_formal/eval_flap_mlp_wifn_ulum_wikitext2_nobias_p20_qwen3_math500_budget`
- `runs/02_baselines/flap_mlp_formal/eval_flap_mlp_wifn_ulum_wikitext2_nobias_p40_qwen3_gsm8k_budget`
- `runs/02_baselines/flap_mlp_formal/eval_flap_mlp_wifn_ulum_wikitext2_nobias_p40_qwen3_math500_budget`
- `runs/02_baselines/flap_mlp_formal/eval_flap_mlp_wifn_ulum_wikitext2_nobias_p60_qwen3_gsm8k_budget`
- `runs/02_baselines/flap_mlp_formal/eval_flap_mlp_wifn_ulum_wikitext2_nobias_p60_qwen3_math500_budget`

正式配置：

- pruning metric: `WIFN`
- pruning structure: `UL-UM`
- calibration data: `wikitext2`
- bias compensation: `false`
- target module: Qwen3 MLP intermediate channels

| 方法 | 数据集 | 样本数 | 答对数 | 准确率 | 相对 Dense 下降 |
|---|---:|---:|---:|---:|---:|
| FLAP-MLP p05 | GSM8K | 1319 | 926 | 70.20% | -10.31 |
| FLAP-MLP p05 | MATH500 | 500 | 113 | 22.60% | -33.40 |
| FLAP-MLP p05 | 合并 | 1819 | 1039 | 57.12% | -16.66 |
| FLAP-MLP p20 | GSM8K | 1319 | 29 | 2.20% | -78.32 |
| FLAP-MLP p20 | MATH500 | 500 | 7 | 1.40% | -54.60 |
| FLAP-MLP p20 | 合并 | 1819 | 36 | 1.98% | -71.80 |
| FLAP-MLP p40 | GSM8K | 1319 | 0 | 0.00% | -80.52 |
| FLAP-MLP p40 | MATH500 | 500 | 0 | 0.00% | -56.00 |
| FLAP-MLP p40 | 合并 | 1819 | 0 | 0.00% | -73.78 |
| FLAP-MLP p60 | GSM8K | 1319 | 0 | 0.00% | -80.52 |
| FLAP-MLP p60 | MATH500 | 500 | 2 | 0.40% | -55.60 |
| FLAP-MLP p60 | 合并 | 1819 | 2 | 0.11% | -73.67 |

Dense 原本答对、剪枝后答错的数量如下：

| 方法 | 数据集 | Dense 答对但剪枝后答错 | Dense 答错但剪枝后答对 |
|---|---:|---:|---:|
| FLAP-MLP p05 | GSM8K | 218 | 82 |
| FLAP-MLP p05 | MATH500 | 181 | 14 |
| FLAP-MLP p20 | GSM8K | 1035 | 2 |
| FLAP-MLP p20 | MATH500 | 275 | 2 |
| FLAP-MLP p40 | GSM8K | 1062 | 0 |
| FLAP-MLP p40 | MATH500 | 280 | 0 |
| FLAP-MLP p60 | GSM8K | 1062 | 0 |
| FLAP-MLP p60 | MATH500 | 279 | 1 |

**解释：**

FLAP-MLP 的 `p05` 是当前唯一没有完全崩溃的 FLAP 点。在 GSM8K 上，`p05` 从 80.52% 降到 70.20%，仍然保留了一部分能力；但在 MATH500 上，`p05` 从 56.00% 降到 22.60%，下降非常明显。

这说明复杂数学推理任务对静态 MLP width pruning 更敏感。也就是说，即使只剪 5%，对于 MATH500 这种更复杂、更依赖长链推理的任务，也可能严重破坏模型的 reasoning trajectory。

`p20` 及以上基本坍塌，说明当前 Qwen3 + FLAP-MLP 设定下，静态 MLP 结构剪枝在 reasoning tasks 上风险很高，不适合作为强性能 baseline，但可以作为“静态剪枝容易造成 reasoning collapse”的对照证据。

## 4. 不建议进入正式 baseline 表格的结果

下面这些结果只用于调试、smoke test 或 ablation，不建议写入正式 baseline 表格：

- `runs/02_baselines/eval_griffin_*_l20`
  - 20 样本 GRIFFIN density sweep / smoke test。
- `runs/02_baselines/eval_griffin_qwen3_*_smoke*`
  - 早期 GRIFFIN smoke test。
- `runs/02_baselines/flap_mlp_light_ablation`
  - 20 样本 FLAP metric / structure / bias compensation 选择实验。
- `runs/02_baselines/eval_flap_mlp_p20_*`
  - 早期 FLAP p20 测试，不是最终正式配置。
- `runs/02_baselines/eval_flap_mlp_wifn_ulum_*_p20_*`
  - 早期 FLAP WIFN+UL-UM 测试，不在 formal 文件夹下。
- `runs/02_baselines/llm_pruner_mlp_formal`
  - naive LLM-Pruner-style Qwen3 MLP port，`p05` 就出现重复 token 崩溃，已经排除出正式 baseline。

## 5. 当前 baseline 结论

目前可以进入正式讨论的 baseline 集合是：

1. **Dense Qwen3**
   - 原始模型，不剪枝。
   - 作为所有剪枝结果的准确率上界和对照。

2. **GRIFFIN-style FFN dynamic pruning**
   - 动态 FFN 剪枝。
   - 不是 reasoning-risk-aware。
   - `p20` 已经造成明显准确率下降，说明仅靠 sequence-level FFN 动态剪枝不足以保护推理过程。

3. **FLAP-MLP static pruning**
   - 静态 MLP 结构化剪枝。
   - `p05` 在 GSM8K 上还保留一定能力，但在 MATH500 上下降严重。
   - `p20` 及以上基本坍塌，说明静态结构化剪枝在 LRM reasoning tasks 上风险很高。

整体来看，这些 baseline 共同支持我们的 motivation：

> LRM 剪枝不能只依赖静态结构指标，也不能只依赖 sequence-level FFN 激活模式。推理模型的剪枝决策需要感知 reasoning process 内部的 step / stage 风险，并在 verification、final answer、关键 derivation 等高风险阶段采取更保守的剪枝策略。

这也是后续 RASP-Zero / RASP-Train 的核心出发点。
