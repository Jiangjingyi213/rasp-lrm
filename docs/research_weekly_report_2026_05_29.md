# 科研周报：Reasoning-Aware Dynamic Structured Pruning for LRMs

日期：2026-05-29  
项目：面向 Large Reasoning Models 的 reasoning-aware dynamic structured pruning  
当前主模型：`Qwen/Qwen3-1.7B`  
当前主数据集：GSM8K + MATH500

## 1. 本周工作概览

本周的核心目标是把前期零散的 smoke test 推进为一套相对完整、可解释、可用于论文 motivation 的实验闭环。围绕这个目标，我主要完成了三件事：

1. **完成 Qwen3 motivation 实验的正式统计与图表整理。**
   - 在 GSM8K 与 MATH500 上完成 dense generation、reasoning segmentation、counterfactual pruning、oracle analysis、risk probe 与 heatmap summary。
   - 对当前所有 counterfactual 结果进行了系统聚合，形成 `runs/01_motivation/motivation_analysis/motivation_report.md`。
   - 生成了 paper-ready motivation figures，包括 stage × module heatmap、stage × ratio heatmap、module × ratio heatmap、oracle gap、entropy/probe comparison 等。

2. **完成 Dense Qwen3 / GRIFFIN-style / FLAP-MLP baseline 的正式结果整理。**
   - Dense Qwen3 作为未剪枝基线。
   - GRIFFIN-style FFN dynamic pruning 作为 sequence-level dynamic pruning baseline。
   - FLAP-MLP static pruning 作为静态结构化剪枝 baseline。
   - 结果已整理在 `docs/baseline_results_summary.md`。

3. **尝试并否定 naive LLM-Pruner-style Qwen3 MLP port。**
   - 初步实现了一个 Qwen3 MLP-only 的 LLM-Pruner-style static width pruning baseline。
   - 但实验发现即使只剪 5% MLP intermediate channels，也会出现明显的 repeated-token generation collapse。
   - 因此目前不将其作为正式 LLM-Pruner baseline，而是将其标记为 diagnostic-only，后续若要使用 LLM-Pruner，应转向更严格的官方 dependency graph / Taylor importance / LoRA recovery 路线，不能用当前 naive port 代替。

总体上，本周的主要进展不是单纯跑出更多数字，而是逐步明确了本文工作的核心论证链条：

> LRM 的剪枝风险并不是一个只由 pruning ratio 或静态结构重要性决定的问题，而是与数据集难度、推理阶段、模块类型、剪枝粒度、当前 hidden state 共同相关。因此，LRM pruning 需要 reasoning-aware、stage-aware、risk-aware 的动态策略。

## 2. 当前实验链路与仓库状态

当前 motivation 实验链路已经形成比较清晰的 6 个阶段。

### 2.1 Dense Generation

首先运行未剪枝的 Qwen3，生成 dense reasoning trajectories。

输出文件：

- `01_trajectories.jsonl`

每条样本包含：

- 问题 `question`
- 标准答案 `gold`
- 模型生成的推理过程 `completion`
- 抽取出的最终答案 `prediction`
- 是否答对 `correct`

这一阶段的作用是建立原始模型能力基线。后续 counterfactual pruning 只在 dense 原本答对的样本上进行，这样 answer flip 才有明确含义：原本正确的 reasoning trajectory 在剪枝扰动下是否变成错误。

### 2.2 Reasoning Segmentation

对 dense-correct 样本进行 rule-based reasoning segmentation。

输出文件：

- `02_segments.jsonl`

当前 segment type 包括：

- `understanding`
- `planning`
- `derivation`
- `verification`
- `final`

需要强调的是，当前 stage assignment 是 **rule-based heuristic**，不是人工标注，也不是 LLM classifier。这一点后续论文中必须写清楚。它目前适合做 motivation signal，即证明不同 reasoning stage 的 pruning sensitivity 存在差异，但正式论文中最好抽样人工检查 20-50 条，验证 stage assignment 是否大体合理。

### 2.3 Counterfactual Structured Pruning

对每个 `(problem, segment, pruning action)` 进行 counterfactual continuation。

输出文件：

- `03_counterfactuals.jsonl`

当前 action space 包括：

- module：
  - `layer`
  - `attention_block`
  - `attention_heads`
  - `mlp_block`
  - `mlp_channels`
- pruning ratio：
  - `0.2`
  - `0.4`
  - `0.6`
- layer groups：
  - 不同层集合组合

如果 counterfactual continuation 后的答案与 dense baseline answer 不一致，则记为 `flipped=true`。这个 flip rate 是当前 motivation 实验最核心的风险指标。

注意：这里的 counterfactual pruning 是一种**在线 ablation / continuation 风险评估**，并不等价于真实导出 pruned model 后的硬件加速结果。因此它主要用于理解“哪些 reasoning state 对哪些结构扰动敏感”，不是直接声称获得真实 latency speedup。

### 2.4 Oracle Analysis

从同一张 counterfactual table 中计算三种 oracle：

- **static oracle**：全数据集固定选择一个最优 action。
- **prompt oracle**：每个 problem 选择一个最优 action。
- **step oracle**：每个 problem-step 选择一个最优 action。

这三者的关系可以回答一个关键问题：

> 剪枝策略是否应该从 static policy 推进到 prompt-level policy，再进一步推进到 step-level reasoning-aware policy？

当前结果显示 step oracle 显著强于 static oracle 和 prompt oracle，这是我们提出 RASP 类方法的重要 motivation。

### 2.5 Risk Probe

训练 problem-level split 的 probe，用于预测当前 segment/action 是否高风险。

比较的 feature set 包括：

- entropy
- confidence
- activation
- hidden state
- combined

这里已经修正为 problem-level split，避免同一题不同 layer/action 泄漏到 train 和 validation 两边。这一点对后续写论文非常重要，因为 row-level split 会高估 probe 能力。

### 2.6 Heatmap Summary and Paper Figures

最后将 counterfactual 结果聚合为：

- dataset-level flip rate
- module-level flip rate
- ratio-level flip rate
- stage-level flip rate
- module × ratio heatmap
- stage × module heatmap
- stage × ratio heatmap
- oracle gap figure
- entropy/probe comparison figure

当前图像文件位于：

- `runs/01_motivation/motivation_analysis/figures/`
- `runs/01_motivation/motivation_analysis/paper_figures/`

其中 paper-ready 图像包括：

- `fig1_reasoning_stage_sensitivity_heatmaps`
- `fig1b_module_ratio_heatmap`
- `fig2_oracle_gap`
- `fig3_entropy_not_enough`
- `fig5_counterfactual_proxy_pareto`

## 3. Motivation 实验核心结果

### 3.1 Dense Baseline 与 Counterfactual 规模

| 数据集 | Dense Acc | Dense Correct | Counterfactual 数量 | Flip Rate |
|---|---:|---:|---:|---:|
| GSM8K | 80.5% | 1062 / 1319 | 82650 | 41.8% |
| MATH500 | 56.0% | 280 / 500 | 21630 | 61.1% |

合并后：

- counterfactual actions：104280
- overall flip rate：45.8%

这个结果说明两点：

第一，Qwen3-1.7B 的 dense accuracy 足够高。GSM8K 80.5%、MATH500 56.0%，说明它不是一个过弱的模型，因此后续观察到的剪枝风险不是因为模型本身完全不会推理。

第二，dense-correct 样本在结构扰动下会大量 answer flip。也就是说，即使模型已经走出一条正确 reasoning trajectory，在中间某个 reasoning segment 施加结构扰动，也很容易导致最终答案翻转。这说明 LRM 的 reasoning process 对结构剪枝非常敏感。

更重要的是，MATH500 的 flip rate 达到 61.1%，明显高于 GSM8K 的 41.8%。这说明数据集难度会显著影响剪枝风险。复杂数学推理不是简单地“多生成一些 token”，而是对中间 reasoning state、模块完整性、局部计算链条更敏感。

### 3.2 Module Sensitivity

| Module | Flip Rate | n |
|---|---:|---:|
| attention_heads | 36.6% | 20856 |
| attention_block | 44.1% | 20856 |
| mlp_channels | 38.0% | 20856 |
| mlp_block | 53.1% | 20856 |
| layer | 57.4% | 20856 |

模块级结果非常关键。完整 layer 和 MLP block 的风险最高，attention heads 和 MLP channels 相对温和。

这说明 pruning granularity 本身就是风险来源之一。不能简单说“剪 20% 就安全，剪 60% 就危险”，而要看剪的是哪一种结构：

- 如果剪完整 layer，模型的 residual computation path 会被大幅改变，风险最高。
- 如果剪完整 MLP block，模型在该层的非线性变换能力会被强烈破坏，也非常危险。
- 如果只剪 attention heads 或 MLP channels，扰动更细，平均风险更低。

因此，后续 RASP 的 router 不应该只输出 pruning ratio，还应该输出 module choice。也就是说，策略空间应该至少是：

```text
当前 reasoning state -> 选择 module + ratio + layer group
```

而不是只判断“剪多少”。

### 3.3 Pruning Ratio Sensitivity

| Ratio | Flip Rate | n |
|---|---:|---:|
| 0.2 | 43.2% | 34760 |
| 0.4 | 41.6% | 34760 |
| 0.6 | 52.7% | 34760 |

`r=0.60` 明显更危险，这符合直觉。但更值得注意的是，`r=0.20` 和 `r=0.40` 并不是严格单调。`r=0.40` 的平均 flip rate 甚至略低于 `r=0.20`。

这并不是说 40% 剪枝比 20% 更安全，而是说明 ratio 不是唯一决定因素。由于不同 ratio 对应不同 module/layer/action 组合，平均值会受到 action composition 影响。因此论文中不应把 pruning risk 简化成一个单调 ratio curve，而应该强调：

> pruning ratio 只是风险因素之一，module、layer group、stage、dataset difficulty 和 hidden state 都会改变实际风险。

### 3.4 Reasoning Stage Sensitivity

| Stage | Flip Rate | n |
|---|---:|---:|
| understanding | 44.8% | 6420 |
| planning | 50.8% | 1800 |
| derivation | 48.8% | 84735 |
| verification | 60.4% | 555 |
| final | 21.2% | 10770 |

Stage 结果是当前 motivation 中非常直观的一部分。不同 reasoning stage 的 pruning sensitivity 存在明显差异：

- `planning`、`derivation`、`verification` 更敏感。
- `final` 相对不敏感。
- `verification` 最高，但样本数较少，需要谨慎解释。

我的理解是：

1. **planning 阶段**决定后续推理路线。如果这里被扰动，模型可能选错解法或错误分解问题，因此后续即使生成流畅，也可能沿着错误方向走。

2. **derivation 阶段**是大多数数学计算和逻辑推导发生的位置。这里的 token 数最多，因此样本量最大，也是 RASP 最需要建模的主体阶段。

3. **verification 阶段**虽然样本少，但 flip rate 很高。可能原因是 verification 往往接近最终答案修正和一致性检查，如果此时结构被扰动，模型可能无法发现前面的错误，或者把正确答案改错。

4. **final 阶段** flip rate 较低，可能是因为最终答案已经基本形成，后续再扰动不一定改变已生成的主要 reasoning path。不过这不能简单理解为 final 永远安全，因为当前 segmentation 是 rule-based，并且 final 段可能包含已经完成的答案表达。

这一结果直接支持论文中的核心句子：

> LRM pruning should be conditioned on the reasoning process, not only on the prompt or model structure.

## 4. Paper-ready 图像解释

后续你会把图贴进周报或论文草稿中，因此这里先写每张图应该如何解释。

### 4.1 Fig. 1：Reasoning Stage Sensitivity Heatmaps

对应文件：

- `runs/01_motivation/motivation_analysis/paper_figures/fig1_reasoning_stage_sensitivity_heatmaps.pdf`
- `.png`
- `.svg`

这张图主要展示：

- stage × module 的 flip rate
- stage × ratio 的 flip rate

图像想表达的问题是：

> 不同 reasoning stage 对不同剪枝结构和剪枝强度的敏感性是否一致？

当前答案是否定的。比如 layer 和 mlp_block 在多个 stage 中都更危险，而 final stage 整体风险更低。verification 虽然样本少，但在 layer / mlp_block 上风险很高。

这张图是 motivation 中最重要的图之一，因为它把“reasoning process 内部存在 pruning sensitivity 差异”可视化了。它可以支撑 RASP 中 stage-aware policy 的设计，即不同阶段应该采用不同剪枝上限：

- verification / final answer：保守剪枝
- derivation：中等剪枝，依赖 risk score
- low-risk explanation / non-critical segment：可以更激进

需要注意的是，图注中应该明确写：

> Stages are automatically assigned by rule-based heuristics.

否则读者可能误以为 stage 是人工标注或 LLM classifier 标注。

### 4.2 Fig. 1b：Module × Ratio Heatmap

对应文件：

- `runs/01_motivation/motivation_analysis/paper_figures/fig1b_module_ratio_heatmap.pdf`

这张图展示不同 module 与 ratio 组合下的 flip rate。

它最重要的观察是：

- layer 的 `r=0.60` 风险最高，flip rate 约 69.1%。
- mlp_block 的 `r=0.60` 也很高，约 61.2%。
- attention_heads 和 mlp_channels 整体更温和。

这张图说明：

> 剪枝风险不是 ratio 单独决定的，而是 module × ratio 的共同结果。

这对后续方法设计很重要。RASP-Zero 或 RASP-Train 不应该只预测一个 ratio，而应该做 action routing，例如：

```text
低风险 state -> 可以考虑 mlp_channels / attention_heads + 较高 ratio
高风险 state -> 避免 layer / mlp_block，或降低 ratio
```

### 4.3 Fig. 2：Oracle Gap

对应文件：

- `runs/01_motivation/motivation_analysis/paper_figures/fig2_oracle_gap.pdf`

当前 oracle 结果：

| Policy | Flip Rate |
|---|---:|
| static oracle | 69.1% |
| prompt oracle | 77.4% |
| step oracle | 83.0% |
| macro prompt oracle | 73.3% |
| macro step oracle | 83.0% |

这张图是当前最有说服力的 motivation 图之一。它展示了：

- 如果只能固定一个全局 action，static oracle 已经能达到 69.1%。
- 如果每个 prompt / problem 可以选择 action，prompt oracle 提升到 77.4%。
- 如果每个 reasoning step 可以选择 action，step oracle 达到 83.0%。

这个 gap 的含义是：

> 同一个问题内部，不同 reasoning step 对剪枝的敏感性不同。因此只根据 prompt 做一次剪枝决策仍然不够，step-level dynamic pruning 有额外价值。

这正是我们区别于 prompt-router pruning 或 static pruning 的地方。

### 4.4 Fig. 3：Entropy Is Not Enough

对应文件：

- `runs/01_motivation/motivation_analysis/paper_figures/fig3_entropy_not_enough.pdf`

这张图包含两个信息：

1. entropy 与 flip risk 的关系。
2. 不同 probe feature 的 ROC-AUC 对比。

当前 probe 结果：

| Dataset | Feature | ROC-AUC | PR-AUC |
|---|---|---:|---:|
| GSM8K | entropy | 0.566 | 0.430 |
| GSM8K | confidence | 0.565 | 0.429 |
| GSM8K | activation | 0.641 | 0.555 |
| GSM8K | hidden | 0.831 | 0.777 |
| GSM8K | combined | 0.828 | 0.774 |
| MATH500 | entropy | 0.624 | 0.683 |
| MATH500 | confidence | 0.515 | 0.585 |
| MATH500 | activation | 0.579 | 0.687 |
| MATH500 | hidden | 0.850 | 0.871 |
| MATH500 | combined | 0.847 | 0.874 |

这张图的核心结论是：

> next-token entropy / confidence 只能弱预测剪枝风险，而 hidden-state probe 明显更强。

这对 RASP 的 risk estimator 设计非常关键。如果只用 entropy-based pruning，就会漏掉很多“模型看起来很自信但内部 reasoning state 很脆弱”的情况。Hidden state 则更能反映当前 reasoning segment 是否依赖关键结构。

这也解释了为什么我们后续不应该只做 entropy router，而应该做 hidden-state / activation-aware risk estimator。

### 4.5 Fig. 5 Proxy：Counterfactual Proxy Pareto

对应文件：

- `runs/01_motivation/motivation_analysis/paper_figures/fig5_counterfactual_proxy_pareto.pdf`

这张图是 proxy Pareto，不是真实 latency/FLOPs Pareto。它基于 counterfactual action 的结构激活 proxy 来展示不同 action 在风险和剪枝强度之间的关系。

写作时需要谨慎说明：

> This figure uses activated-structure proxy rather than measured runtime speedup.

它适合用于 motivation：说明 action space 中存在风险-预算 tradeoff，不同 action 并不等价。但真正论文中的最终 efficiency table 仍然需要后续报告实际：

- latency
- tokens/s
- peak GPU memory
- activated parameters / FLOPs proxy

## 5. Baseline 结果总结

本周同时整理了三个可以进入正式讨论的 baseline。

### 5.1 Dense Qwen3

| 方法 | 数据集 | 样本数 | 答对数 | 准确率 |
|---|---:|---:|---:|---:|
| Dense Qwen3 | GSM8K | 1319 | 1062 | 80.52% |
| Dense Qwen3 | MATH500 | 500 | 280 | 56.00% |
| Dense Qwen3 | 合并 | 1819 | 1342 | 73.78% |

Dense Qwen3 是后续所有 pruning baseline 的准确率上界和主要对照。

### 5.2 GRIFFIN-style FFN Dynamic Pruning

| 方法 | GSM8K Acc | MATH500 Acc | Combined Acc |
|---|---:|---:|---:|
| GRIFFIN p20 | 39.42% | 25.40% | 35.57% |
| GRIFFIN p40 | 6.75% | 7.00% | 6.82% |
| GRIFFIN p60 | 2.12% | 3.80% | 2.58% |

GRIFFIN-style baseline 的意义在于：它代表一种动态 FFN neuron selection，但它并没有显式建模 reasoning stage 或 step-level risk。

结果显示，p20 就已经造成明显准确率下降。这说明“动态”本身并不够，动态策略还必须知道什么时候不能剪、剪哪里更安全。

### 5.3 FLAP-MLP Static Pruning

| 方法 | GSM8K Acc | MATH500 Acc | Combined Acc |
|---|---:|---:|---:|
| FLAP-MLP p05 | 70.20% | 22.60% | 57.12% |
| FLAP-MLP p20 | 2.20% | 1.40% | 1.98% |
| FLAP-MLP p40 | 0.00% | 0.00% | 0.00% |
| FLAP-MLP p60 | 0.00% | 0.40% | 0.11% |

FLAP-MLP 的结果说明静态结构化剪枝在 LRM 上风险很高。p05 在 GSM8K 上还能保留一定能力，但在 MATH500 上下降严重；p20 及以上基本崩溃。

我目前对这组结果的理解是：

- 静态结构指标可以找到一部分相对不重要的 MLP channels。
- 但 reasoning task 中的“重要性”不是固定的，某些 channel 可能只在特定推理阶段或复杂问题中变得关键。
- 因此静态剪枝在简单任务上可能看起来还能工作，但一到 MATH500 这种复杂推理任务就会暴露风险。

### 5.4 LLM-Pruner-style 诊断结果

我们还尝试了 naive LLM-Pruner-style Qwen3 MLP port，但该路线目前不进入正式 baseline。

原因是：

- p05 就出现 repeated-token collapse。
- 生成结果出现大量重复数字、重复短语、语义漂移。
- 这说明当前 naive port 不等价于官方 LLM-Pruner。

因此目前只保留为失败诊断，不用于正式结果展示。后续如果需要 LLM-Pruner baseline，应参考官方实现中的：

- dependency graph
- Taylor importance
- block-wise pruning range
- LoRA recovery
- Llama/Qwen 架构差异适配

不能用当前 naive MLP-only port 冒充官方 LLM-Pruner baseline。

## 6. 个人阶段性思考

本周结果让我对这个课题的理解更清楚了。最开始我们的问题比较像：“能不能把已有剪枝方法迁移到 LRM 上？”但现在更准确的问题应该是：

> 为什么已有 LLM 剪枝方法迁移到 LRM reasoning tasks 时会不稳定？这种不稳定是否来自 reasoning process 内部的阶段性结构依赖？

从当前结果看，答案越来越倾向于“是”。

首先，Dense Qwen3 的能力足够，不是模型太弱。其次，counterfactual flip rate 很高，说明正确推理轨迹对结构扰动敏感。第三，stage/module/ratio 的热力图显示风险不是均匀分布的。第四，oracle gap 显示 step-level policy 明显优于 static/prompt-level policy。第五，hidden-state probe 明显强于 entropy/confidence，说明风险确实藏在模型内部表征里，而不是简单地体现在输出分布不确定性上。

这些证据组合起来，形成了比较完整的 motivation：

1. LRM 推理过程很长，内部包含不同功能阶段。
2. 不同阶段对结构剪枝的容忍度不同。
3. 不同结构模块的扰动风险不同。
4. 静态剪枝和 sequence-level dynamic pruning 都缺少对 reasoning-critical state 的判断。
5. 因此需要一个 reasoning-aware risk estimator 和 dynamic router。

我认为这是目前实验最有价值的地方：它不是单纯证明“某个 baseline 很差”，而是在解释“为什么 reasoning-aware pruning 是必要的”。

## 7. 当前不足与需要谨慎表述的地方

### 7.1 Stage assignment 仍是 heuristic

当前 reasoning stage 是 rule-based assignment。它适合 motivation，但不能过度声称为精确语义标注。论文中应写：

> We use an automatic rule-based stage assignment and conduct manual spot checks.

后续建议抽样检查 20-50 条，记录 heuristic 的合理性。

### 7.2 Counterfactual pruning 不等于真实加速

Counterfactual 结果反映的是结构扰动风险，不是部署速度。当前 paper figures 中的 proxy Pareto 也应明确是 activated-structure proxy，不是真实 latency/FLOPs。

后续若进入正式 method evaluation，需要单独测：

- latency
- tokens/s
- peak memory
- activated parameter proxy
- possibly FLOPs proxy

### 7.3 FLAP / GRIFFIN 是适配版 baseline

当前 GRIFFIN-style 和 FLAP-MLP 都是面向 Qwen3 的适配版本，不应过度声称完全复现原论文所有设置。写作时建议使用：

- `GRIFFIN-style FFN dynamic pruning`
- `FLAP-MLP static pruning`

而不是直接说完整 GRIFFIN / 完整 FLAP。

### 7.4 LLM-Pruner 暂时不能作为正式 baseline

当前 naive LLM-Pruner-style MLP port 已经显示不稳定。后续如果要做正式 LLM-Pruner baseline，需要更高成本的官方适配，不应把失败的 naive port 写进主表。

## 8. 下一步计划

### 8.1 完成 RASP-Zero

下一步最自然的是建立 RASP-Zero。当前结果已经支持以下规则：

```text
if risk_probe_score > high_threshold:
    no pruning or very light pruning
elif risk_probe_score > mid_threshold:
    conservative MLP/channel pruning
else:
    stronger pruning allowed

if stage in {verification, final answer}:
    cap pruning ratio at low level
if stage in {planning, key derivation}:
    prefer conservative modules
if stage is low-risk explanation:
    allow stronger module/channel pruning
```

RASP-Zero 不需要训练复杂模型，可以先利用当前 probe score、stage type 和 action risk table 做规则策略。

### 8.2 做离线 policy simulation

利用已有 `03_counterfactuals.jsonl`，先离线比较：

- static policy
- prompt oracle
- step oracle
- entropy-based policy
- confidence-based policy
- hidden-probe policy
- RASP-Zero rule policy

这一步不需要重新生成模型，只需要在已有 counterfactual table 上做 policy selection。因此成本低，但能直接回答：

> RASP-Zero 是否比 entropy-only、static pruning 更接近 step oracle？

### 8.3 补充真实 efficiency 指标

如果 RASP-Zero 离线效果明显，再进入真实 generation-time pruning 实现。届时需要报告：

- accuracy
- answer flip rate
- average pruning ratio
- activated modules / parameters
- latency
- tokens/s
- peak memory

### 8.4 完善论文图像

当前可以放入周报/论文草稿的图：

- Fig. 1 stage sensitivity heatmap
- Fig. 1b module-ratio heatmap
- Fig. 2 oracle gap
- Fig. 3 entropy/probe comparison
- Fig. 5 proxy Pareto

后续暂时还不能严谨画 FFN flocking 图，因为当前没有记录 per-stage FFN top-neuron sets / Jaccard similarity。若后续想做该图，需要新增 hook，记录每个 stage 的 top-k FFN neurons。

## 9. 本周结论

本周已经基本完成 motivation 实验的闭环。当前证据表明：

1. Qwen3-1.7B 在 GSM8K / MATH500 上具备足够 dense reasoning 能力。
2. 正确 reasoning trajectory 对结构扰动高度敏感。
3. MATH500 比 GSM8K 更容易受到剪枝破坏。
4. Layer / MLP block 的风险显著高于 heads / channels。
5. Reasoning stage 之间存在明显 pruning sensitivity 差异。
6. Step-level oracle 明显优于 static 和 prompt-level oracle。
7. Hidden-state probe 明显优于 entropy/confidence。
8. GRIFFIN-style 和 FLAP-MLP baseline 都显示出非 reasoning-aware 剪枝在 LRM 上的局限。

因此，下一阶段推进 RASP-Zero 是合理的：它可以把当前 motivation 中观察到的 stage sensitivity、module sensitivity、risk probe 和 oracle gap 连接起来，形成一个真正 reasoning-aware 的 pruning policy。

从论文叙事角度看，目前已经可以比较清楚地写出核心问题：

> Existing pruning methods treat pruning as a static or sequence-level efficiency decision, while reasoning-centric models require pruning decisions to be conditioned on the evolving reasoning state.

这也是后续方法设计最应该抓住的主线。
