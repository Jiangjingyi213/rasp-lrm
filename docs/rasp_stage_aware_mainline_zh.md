# RASP Stage-Aware Hidden Controller 主线

## 1. 当前方法裁决

项目停止继续扩大“hidden/uncertainty 直接预测短窗口 final-answer flip”的连续在线 router。该路线
失败不等于 hidden 无效：Motivation 的 problem-level OOF 结果显示 hidden、action-hidden、
action-hidden-stage ROC-AUC 分别为 `0.825/0.839/0.846`，说明 hidden 能表示长期结构扰动下的
reasoning fragility，soft stage 也提供额外信息。

新的主线将 hidden 用于：

```text
reasoning stage recognition + fragility estimation
```

而不是直接要求 hidden 精确预测一次短窗口动作的最终答案 flip。

## 2. Phase S1：Hidden Stage Probe

S1 已实现完整代码链路。首轮五类规则标签人工审计失败后，当前默认使用可操作的四类 taxonomy：

```text
setup / reasoning / verification / final
```

`planning` 与 `derivation` 在模型生成的 segment 中经常同时发生，人工也无法稳定切开，因此合并为
`reasoning`；这不是降低 gate，而是删除不可可靠观测的标签边界。

```text
src/rasp/stage_probe.py
src/main_prepare_rasp_stage_probe_data.py
src/main_train_rasp_stage_probe.py
src/main_eval_rasp_stage_probe.py
scripts/57_prepare_rasp_stage_probe_data.sh
scripts/58_train_rasp_stage_probe.sh
scripts/59_eval_rasp_stage_probe.sh
scripts/60_summarize_rasp_stage_probe.py
```

数据准备从四个正式 Motivation run 中按 `(dataset, problem, segment)` 去重。每个 reasoning segment
只保留一份 boundary hidden、entropy、confidence、position 和规则伪标签，不把同一 segment 的
多个剪枝 action 当作独立 stage 样本。数据准备同时生成 100 条按 dataset/stage 分层的人工审计表。
人工审计需要在 CSV 的 `audited_stage` 列填写四类 operational 标签；汇总器会检查规则伪标签与
人工标签的一致率。
伪标签由当前 `rule_segmenter` 重新生成，数据摘要会记录它与 Motivation 旧缓存标签的不一致数。

比较五个 feature variant：

```text
position_only
uncertainty_only
hidden_pca_linear
hidden_pca_nonlinear
hidden_uncertainty
```

所有 variant 使用 problem-level train/validation/test split。标准化和 hidden PCA 只在 train
rows 拟合；validation macro-F1 选择 checkpoint；test 只做最终评估。类别不平衡由 train-only
class weights 处理。

S1 准入条件固定为：

```text
最佳 hidden variant test macro-F1 比最佳简单 baseline 至少高 0.05
setup / reasoning / verification / final 四类在每个 seed 的 recall 均 >= 0.70
五个 variant 均覆盖三个训练 seed，且最佳 hidden variant macro-F1 std <= 0.05
至少完成 100 条人工审计，规则伪标签与人工标签一致率 >= 0.80
```

## 3. 执行流程

服务器执行：

```bash
bash scripts/57_prepare_rasp_stage_probe_data.sh
```

先检查并人工填写：

```text
runs/07_stage_aware/02_s1_operational_stage_probe/data/00_stage_data_summary.json
runs/07_stage_aware/02_s1_operational_stage_probe/data/02_stage_manual_audit.csv
```

三张 GPU 并行训练：

> 当前人工审计未通过，以下训练命令暂停执行。保留命令仅用于阶段标签修订并重新审计通过后运行。

四类标签填写标准：

```text
setup         题意理解、条件提取、变量/目标建模，尚未进行主体求解
reasoning     规划、公式选择、计算、代数推导与中间结论
verification  对已有候选解、等式或结果进行显式复核
final         明确输出最终答案
```

```bash
mkdir -p logs/07_stage_aware

nohup env CUDA_VISIBLE_DEVICES=0 STAGE_PROBE_SEEDS="1" \
  bash scripts/58_train_rasp_stage_probe.sh > logs/07_stage_aware/s1_seed1.log 2>&1 &

nohup env CUDA_VISIBLE_DEVICES=1 STAGE_PROBE_SEEDS="2" \
  bash scripts/58_train_rasp_stage_probe.sh > logs/07_stage_aware/s1_seed2.log 2>&1 &

nohup env CUDA_VISIBLE_DEVICES=2 STAGE_PROBE_SEEDS="3" \
  bash scripts/58_train_rasp_stage_probe.sh > logs/07_stage_aware/s1_seed3.log 2>&1 &
```

训练完成后统一评估：

```bash
bash scripts/59_eval_rasp_stage_probe.sh
```

正式验收文件：

```text
runs/07_stage_aware/02_s1_operational_stage_probe/comparison_summary.csv
runs/07_stage_aware/02_s1_operational_stage_probe/s1_gate.json
```

只有 `s1_gate.json` 中 `s1_passed=true` 才实施并运行 S2 runtime stage sensitivity bank。S1 未通过
时不提前采集 S2、不实现 S3 controller，也不通过调整 gate 掩盖结果。

## 4. S1 人工审计结果

100 条分层样本已完成人工审计，结果位于：

```text
runs/07_stage_aware/01_s1_stage_probe/data/02_stage_manual_audit.csv
runs/07_stage_aware/01_s1_stage_probe/data/03_stage_manual_audit_summary.json
```

规则伪标签与人工标签总体一致率仅为 `61%`，低于 `80%` 准入线。各规则类别一致率为：

```text
understanding  75%
planning       20%
derivation     80%
verification   30%
final         100%
```

主要失真是 `planning` 和 `verification` 关键词误触发：`First/Second/Step 2` 常出现在普通计算，
`therefore/check` 也常属于推导过程而非独立验证阶段。当前人工标签中 `46%` 为 derivation，
planning 与 verification 分别只有 `5%/6%`，说明五类阶段本身存在明显类别不平衡。

**当前裁决：旧五类标签永久停止用于 S1。** 已实现四类 operational taxonomy，并将输出隔离到
`02_s1_operational_stage_probe/`。旧 100 条仅作为规则开发诊断；新目录会生成一批独立审计样本。
`scripts/58_train_rasp_stage_probe.sh` 在审计未达到 100 条且一致率未达到 `80%` 时会直接退出，
因此下一步是生成并审核新 CSV，而不是启动 GPU 训练。

## 5. 四类 Operational Stage 独立审计结果

第二批 100 条独立样本已完成审核，结果位于：

```text
runs/07_stage_aware/02_s1_operational_stage_probe/data/02_stage_manual_audit.csv
runs/07_stage_aware/02_s1_operational_stage_probe/data/03_stage_manual_audit_summary.json
configs/stage_audits/s1_operational_v2_labels.csv
```

由于 `runs/` 被 Git 忽略，审核标签另存为仓库内可同步的轻量清单。训练脚本会先运行
`scripts/62_apply_rasp_stage_audit_labels.py`，将人工标签按 `(dataset,id,segment_id)` 合并到服务器
生成的审计 CSV，再执行质量 gate；规则 stage 或样本集合发生变化时会拒绝合并。

总体规则一致率为 `86%`，通过预设 `80%` 标签质量 gate：

```text
setup         96.4%
reasoning     62.5%
verification  90.9%
final        100.0%
```

四类 taxonomy 明显优于旧五类标签，尤其 verification/final 已稳定。剩余 14 条错误中有 12 条为
`reasoning -> setup`：短片段仅列出条件或定义变量，却未被规则识别为 setup。该偏差不会阻止 S1
probe 训练，但必须作为安全性风险单独检查：最终 classifier 的 held-out confusion matrix 中，
真实 setup 被预测为 reasoning 的比例必须足够低，否则不能进入 S2/controller。

**当前裁决：允许启动四类 S1 三 seed probe 训练；仍不进入 S2。**

## 6. 首轮四类 S1 训练审查

首轮四类 checkpoint 已生成，但结果无效，不能执行 S1 准入裁决：

```text
全量 6952 segments:
reasoning=5616, final=719, setup=606, verification=11
```

旧 split 实现把稀有 stage-rich problem 几乎全部分配到 train，导致三个 seed 均出现：

```text
validation: final=0, verification=3
test:       verification=0, final=7, setup=8, reasoning=1164
```

因此 validation macro-F1、checkpoint 选择和后续 test 指标均不具备四类可比性。训练期间 hidden
variant 的 validation macro-F1 约为 `0.26–0.28`，高于 position/uncertainty baseline，但该信号
只能视为诊断，不能证明 hidden stage probe 通过。

代码已修复 problem split：使用 normalized stage deficit 分配，并强制每个 split 覆盖全部可学习
stage；数据准备也新增每类至少 100 行的硬 gate。当前 verification 只有 11 行，会被该 gate 正确
阻止。

**下一步方法裁决：verification 不作为 learned hidden stage。** 自然轨迹中的显式 verification
极少，但独立人工审计显示显式规则精度较高。后续 controller 应将显式 verification 作为保守
`dense override`；hidden probe 只学习样本充足的 `setup/reasoning/final`。完成三类数据与独立审计
后再重跑 S1。现有 `02_s1_operational_stage_probe/seed_*` checkpoint 全部作废。

## 7. 当前执行版本：S1 v3 三类 Learned Stage

S1 v3 已实现：

```text
learned hidden stages: setup / reasoning / final
explicit verification rule: dense override
output: runs/07_stage_aware/03_s1_three_stage_probe/
```

数据准备会将显式 verification 从 learned dataset 排除，并单独写入：

```text
data/01_verification_dense_overrides.jsonl
```

三类数据预计约为 `setup=606 / reasoning=5616 / final=719`，均满足最低 100 行要求。修复后的
problem split 强制 train/validation/test 均覆盖三类。新一批审计 CSV 必须重新审核；旧 v2 标签
不会自动套用到新样本。

S1 v3 除原有 macro-F1、三类 recall 与三 seed 稳定性 gate 外，新增：

```text
每个 seed 的 setup -> reasoning 错误率 <= 10%
```

这是 controller 的安全 gate：setup 被误判为 reasoning 可能导致过早剪枝。执行顺序：

```bash
bash scripts/57_prepare_rasp_stage_probe_data.sh
# 审核 runs/07_stage_aware/03_s1_three_stage_probe/data/02_stage_manual_audit.csv
# 审核标签同步后才运行 scripts/58_train_rasp_stage_probe.sh
```

v3 新一批 100 条独立审计已完成，总体一致率 `85%`，通过 `80%` gate：

```text
final      100.0%
reasoning   77.8%
setup       78.1%
```

人工混淆中 `reasoning -> setup` 有 8 条，表示规则可能把真实 setup 当作可学习 reasoning；
因此后续必须执行 `setup -> reasoning <= 10%` 的模型安全 gate。审核标签已同步到
`configs/stage_audits/s1_three_stage_v3_labels.csv`。

脚本使用专用环境变量 `STAGE_PROBE_ROOT`，不再读取通用 `OUTPUT_ROOT`，避免服务器残留变量将
产物错误写入 `runs/07_stage_aware/data/`。

## 8. S1 v3 三 Seed 训练结果

三 seed、五个 feature variant 已完成训练，但尚未运行 held-out test，因此当前只报告 validation
诊断，不作正式 S1 准入裁决。

```text
variant                macro-F1 mean±std   setup→reasoning mean/max
hidden_pca_nonlinear   0.7551 ± 0.0181     0.2997 / 0.4022
hidden_uncertainty     0.7497 ± 0.0116     0.3145 / 0.3804
hidden_pca_linear      0.6693 ± 0.0061     0.2409 / 0.3696
uncertainty_only       0.3630 ± 0.0061     0.5997 / 0.7609
position_only          0.3391 ± 0.0961     0.2863 / 0.8478
```

最佳 hidden nonlinear 相对最佳简单 baseline 的 validation macro-F1 提高约 `0.392`，且跨 seed
稳定，明确说明 hidden 包含强 reasoning-stage 信息。其平均 recall 为
`setup=0.700 / reasoning=0.807 / final=0.982`。

当前阻断是 argmax 决策下的 `setup -> reasoning` 安全错误率远高于 `10%`。这不等于 hidden
stage probe 失败：S3 controller 原设计要求“stage 置信度不足则 dense”，但当前评估尚未测量
confidence-gated selective classification。下一步先运行 held-out eval，然后仅在 held-out/
calibration 上选择 reasoning confidence threshold，报告：

```text
reasoning coverage
setup -> accepted-reasoning rate
accepted-reasoning precision
```

只有安全错误率不超过 `10%` 且 reasoning coverage 非零，才允许进入 S2。

## 9. S1 v3 Held-Out 正式裁决

三 seed held-out 汇总已完成，`s1_gate.json` 给出 `s1_passed=false`。最佳模型为
`hidden_pca_nonlinear`：

```text
macro-F1                     0.7562 ± 0.0034
相对最佳简单 baseline 增益   +0.3952
setup recall                 0.6857，最差 seed 0.6809
reasoning recall             0.8143，最差 seed 0.8083
final recall                 0.9786，最差 seed 0.9722
setup -> reasoning           0.3143，最差 seed 0.3191
```

已通过：

```text
hidden 相对简单 baseline 增益
三 seed 稳定性
100 条独立人工审计
reasoning/final recall
```

未通过：

```text
setup recall 每 seed >= 0.70
setup -> reasoning <= 0.10
```

失败模式跨三个 seed 几乎完全一致，说明不是随机性或 split 异常，而是 argmax classifier 对
setup/reasoning 边界存在系统性混淆。`hidden_pca_linear` 更保守，setup→reasoning 降为
`0.161–0.223`，但 reasoning recall 仅 `0.607–0.671`，同样不能通过。

**正式裁决：S1 argmax stage classifier 未通过，不进入 S2。** Hidden 的 stage 信息价值已经得到
强力验证，但 controller 必须采用 selective acceptance：只有 reasoning probability 高于
calibration threshold 时才视为可剪枝候选，其余全部 dense。下一实验只评估 threshold 下的
setup false-accept rate、reasoning coverage 和 accepted-reasoning precision；不重新设计 taxonomy，
不无目的重训模型。

## 10. S1.5 与全阶段 S2 实现

S1.5 已实现为 validation-only 阈值校准。对每个 hidden variant 和 seed：

1. 只在 validation 上选择 reasoning probability threshold；
2. 约束 validation `setup -> accepted reasoning <= 10%`；
3. variant 也只按 validation reasoning coverage 选择，避免 test leakage；
4. 固定 variant 与阈值后在 test 上报告 setup false-accept、reasoning coverage 和 accepted precision；
5. 三个 seed 的 test false-accept 均不超过 `10%`，且 reasoning coverage 均至少 `10%` 才通过。

```bash
bash scripts/63_eval_rasp_stage_selective.sh
cat runs/07_stage_aware/03_s1_three_stage_probe/s1_5_gate.json
```

S2 runtime sensitivity smoke 也已实现。需要区分两个 gate：

```text
s2_diagnostic_allowed=true  允许采集全阶段单窗口 sensitivity bank
s3_controller_allowed=true  才允许把 selective classifier 用于在线 controller
```

S1.5 controller gate 未通过时，S2 仍可作为 diagnostic measurement 运行，因为它不会根据 stage
决定是否剪枝，而是对全部阶段公平施加动作；但此时严禁进入 S3 controller。S2 不再预设
setup/final/verification 永远不能剪枝，而是在所有 operational stage 上公平执行
`ratio=0/0.05/0.10/0.20` 的单个 16-token MLP-channel 窗口，随后立即恢复 dense：

```text
learned hidden stage: setup / reasoning / final
explicit text rule:   verification
```

每个 boundary 的 stage 在动作前由 dense observation 固定一次；同一 boundary 的所有 ratio 共用
该 stage 标注。`reasoning_accepted` 仅用于未来 controller 准入，不会过滤 S2 的其他阶段。

服务器 smoke 命令：

```bash
RASP_S2_GPU_COUNT=4 bash scripts/66_collect_rasp_s2_stage_sensitivity.sh
python scripts/67_summarize_rasp_s2_stage_sensitivity.py
```

主要产物：

```text
runs/07_stage_aware/04_s2_stage_sensitivity_smoke/
  */03_stage_window_counterfactuals.jsonl
  */07_stage_window_bank_validation.json
  s2_stage_sensitivity_summary.json
  s2_stage_sensitivity_summary.csv
```

summary 会输出每个 `dataset × operational_stage × ratio` 的 paired flip rate、Wilson 95% CI 和
window divergence。smoke 的 point estimate 只用于决定是否扩大采样，不能直接作为 S2 正式准入。

## 11. S2-v1 Smoke 结果与位置对齐修复

首轮 `04_s2_stage_sensitivity_smoke` 的 10/10 shard 均通过 runtime validator：

```text
boundaries                       449
dense paired flip               0
dense replay flip               0
action                           single 16-token window then dense
```

粗粒度结果显示 reasoning 在 ratio `0.05` 下为 `5/375 = 1.33%` flip；但该轮不能作为正式
stage-safety 裁决，因为审查发现 stage probe 的 position 定义发生了错位：

```text
S1 train: segment_index / (num_segments - 1)
S2-v1:    generated_tokens / max_new_tokens
```

S2-v1 又最多只采前 12 个 boundary，即前 192 token；相对 `max_new_tokens=768` 的 position 全部
位于 `0–0.23`，导致 stage 分布严重偏向 reasoning：

```text
setup=50, reasoning=375, final=21, verification=3
```

paired counterfactual 本身有效，但其 stage 分组不可信。代码已修复为：

```text
stage_position = generated_tokens / (dense_trajectory_tokens - 1)
```

且 S2-v2 默认覆盖完整 dense trajectory，不再截断前 12 个 boundary。新结果隔离写入
`runs/07_stage_aware/05_s2_stage_sensitivity_v2/`，避免与 v1 混用。

## 12. S2-v2 后的方法审查：Stage 降级为辅助特征

S2-v2 的 runtime 链路正确，但进一步对照 Motivation 后，不能继续把“hard stage 决定剪枝率”
作为主线。

### 12.1 Ratio 的正确目标

`ratio=0.05` 只是当前已测动作中最保守、boundary flip 最低的动作，不是“性能最好”的动作。
项目目标应定义为：

```text
在 accuracy / flip-risk 约束下，最大化累计结构剪枝暴露与理论计算节省
```

因此后续必须报告 risk-efficiency Pareto，而不是只寻找最低 flip ratio。`0.20` 虽然结构比例仍
不算激进，但在单个 16-token window 下 reasoning flip 已约为 `5%`；它仍应保留为风险曲线上的
动作点，不能因为不满足严格安全线就从分析中删除。更高 ratio 也可以用于刻画曲线，但不能在
没有 risk selector 的情况下直接用于连续在线剪枝。

### 12.2 Motivation final 与 Runtime final 不可直接比较

正式 Motivation 使用：

```text
prefix_boundary=end
```

对 final segment 来说，正确答案通常已经完整包含在条件前缀中，再施加长期 continuation 扰动，
所以 `final` 的低 flip rate 很大程度测量的是“答案提交后是否还能被改坏”。S2 runtime 则在动作
前的 token boundary 做单窗口剪枝。

此外，S2-v2 的 41 个 `final` boundary 全部来自 hidden 三分类器，而不是显式 final 文本规则；
出现 flip 的部分 GSM8K final boundary 位于轨迹约 `23%–27%`，stage confidence 仅约
`0.56–0.68`，并不是真正语义 final。故旧 Motivation 的 `final 最安全` 不能用于解释 S2。

### 12.3 当前 hard taxonomy 的限制

S1 数据的旧标签到新标签映射显示：

```text
understanding -> reasoning  220
planning      -> reasoning  110
derivation    -> reasoning 5261
verification  -> reasoning   25
derivation    -> setup      387
```

S2-v2 中 `661/774 = 85.4%` boundary 被归为 reasoning。这个三分类器能识别宽泛阶段，但 hard
taxonomy 的信息量不足以独立承担 action routing；`reasoning` 已混合规划、推导、解释和部分验证。

**方法修正：**

```text
主信号：action-conditioned fragility / risk
辅助信号：hidden representation + relative progress + semantic event flags
事件标志：explicit verification / final-answer-start / answer-committed
输出目标：每个 action 的风险与 risk-efficiency Pareto
```

后续不再要求“某个 hard stage 固定对应某个 ratio”。Stage-aware 的含义改为：推理进度与语义
事件帮助风险模型判断当前 action 是否值得执行，而不是让 stage label 直接控制剪枝。

## 13. 八卡 Action-Risk Pilot

当前主线进一步落实为 action-conditioned risk-efficiency pilot。它不复活旧 Phase-B2 router，
也不把 `ratio=0.05` 当作最终方案，而是先测量 `0.05–0.50` 的风险剂量响应。

离线 bank 默认使用 GPU `0,1,2,3`，从 GSM8K train 与排除 MATH500 的 `math_train` 采集。目标
不是“读取 100 道题”，而是每个来源至少保留 100 个 dense-correct problem；默认先生成约
140/200 道输入题。每题最多选择 12 个均匀覆盖完整 dense trajectory 的边界，并执行：

```text
ratio = 0 / 0.05 / 0.10 / 0.20 / 0.30 / 0.40 / 0.50
window = 16 tokens, then dense
```

```bash
GPU_IDS=0,1,2,3 bash scripts/69_collect_action_risk_pilot.sh

# 四个 worker 退出后：
PYTHON=/home/cike/jjy/envs/rasp_qwen3/bin/python bash scripts/70_prepare_action_risk_pilot_data.sh
PYTHON=/home/cike/jjy/envs/rasp_qwen3/bin/python bash scripts/71_train_action_risk_pilot.sh
```

若某来源不足 100 个 dense-correct problem，数据准备会硬失败。提高
`ACTION_RISK_GSM8K_TRAIN_INPUTS` 或 `ACTION_RISK_MATH_TRAIN_INPUTS` 后重跑采集即可；已验证
shard 会自动跳过。OOF 分析使用 problem-level 5-fold，PCA/standardization 只在 fold train
上拟合。

当前 S1 stage probability 依赖完整轨迹相对位置，因此不是合法在线因果特征，本 pilot 不将其
输入风险模型。Stage 保留为后续可因果化的辅助信号，不能通过偷看 dense trajectory 引入。

在线固定动作诊断默认使用 GPU `4,5,6,7`，在 GSM8K test 与 MATH500 各 100 题上运行 dense 以及
`boundary=32/96/160 × ratio=0.10/0.20/0.30/0.40/0.50`。每题最多执行一个窗口：

```bash
GPU_IDS=4,5,6,7 bash scripts/73_eval_online_fixed_window_pilot.sh

# 四个 worker 退出后：
PYTHON=/home/cike/jjy/envs/rasp_qwen3/bin/python python scripts/74_summarize_online_fixed_window_pilot.py
```

主要产物：

```text
runs/07_stage_aware/06_action_risk_pilot/data/01_action_risk_data_summary.json
runs/07_stage_aware/06_action_risk_pilot/analysis/02_action_risk_pilot_summary.json
runs/07_stage_aware/07_online_fixed_window_pilot/online_fixed_window_summary.json
```

100 个 dense-correct problem 足够做路线 go/no-go，但不能证明 accuracy loss `<=1%`。Pilot
通过后才扩展到每来源至少 200 个 dense-correct problem，并训练新的 learned single-window
controller。

## 14. 在线 Fixed-Single-Window Pilot 结果

GPU 4–7 的在线固定单窗口诊断已完成。两个数据集各包含 100 道配对题，32/32 个运行产物完整；
未到达指定 boundary 的题目与 dense completion 完全一致，runtime 元数据与 16-token 单窗口
配置匹配，未发现 controller 接线错误。

最保守且最接近安全线的 GSM8K cell 是：

```text
boundary=96, ratio=0.10
paired accuracy delta          0.00
executed dense-correct flip    1/80 = 1.25%
flip 95% CI                    [0.00%, 3.95%]
theoretical pruning exposure   0.57%
```

它仍不能通过最终 `<=1%` 安全声明。GSM8K 的较高 ratio 在 boundary 96/160 呈明显风险上升；
boundary 32 相对更稳，但 `ratio=0.40` 的零净 accuracy delta 来自 4 个 flip 与 4 个 improvement
相互抵消，不能视为安全。

MATH500 对固定动作明显更脆弱，最安全 point estimate 仍为：

```text
boundary=160, ratio=0.10
paired accuracy delta          -3.00%
executed dense-correct flip    4/57 = 7.02%
flip 95% CI                    [1.61%, 14.55%]
theoretical pruning exposure   0.32%
```

因此不存在可跨 GSM8K/MATH500 直接部署的固定 `boundary × ratio`。但结果并不否定动态
Action-Risk 主线：事后 oracle 在可执行状态上选择“仍保持正确的最大 ratio”，两个数据集的平均
最大安全 ratio 都约为 `0.47`。该 oracle 使用结果泄漏，不能作为可实现性能；它只说明风险集中
于特定状态，存在学习 selector 的潜在空间。脆弱性也具有问题级集中性：MATH500 的全部 flip
event 中约 `42%` 来自最脆弱的 5 道题。

正式裁决：

```text
固定单窗口策略：不通过安全准入
连续固定剪枝：不运行
离线 Action-Risk bank：继续完成
下一判断：检查 OOF hidden/context/action 风险模型能否稳定识别脆弱状态
```

汇总脚本现同时报告总体 dense-correct flip 与仅在实际执行窗口题目上的条件 flip，避免较晚
boundary 因部分题未执行动作而被错误解释为更安全；manifest 未同步时也可直接从 runs 重建任务表。

## 15. 离线 Action-Risk Pilot 结果

GPU 0–3 离线 pilot 已完成。34/34 shard 全部通过 validator，所有 dense replay/control flip
均为 0；共保留 GSM8K `118` 与隔离 math_train `160` 个 full-window eligible dense-correct
problem、`2965` 个 boundary 和 `17790` 个非零 action row。

标签具有清晰的整体剂量响应：

```text
ratio       0.05   0.10   0.20   0.30   0.40   0.50
flip rate   2.23%  2.90%  5.33%  8.40% 10.69% 12.75%
```

但约 `10.0%` GSM8K boundary 与 `8.8%` math_train boundary 的逐状态标签对 ratio 非单调。这是
生成轨迹受局部扰动后的离散/混沌响应，不是 action-grid 缺失；controller 不能假设每个状态上
较小 ratio 必然更安全。

5-fold problem-level OOF 结果：

```text
variant                    ROC-AUC   PR-AUC
action-only                0.6660    0.1099
causal context + action    0.6941    0.1308
hidden + context + action  0.6983    0.1536
hidden PCA nonlinear       0.6892    0.1503
```

`causal context + action` 相对 action-only 的 ROC/PR 在 `4/5` folds 提升，证明因果状态特征具有
可复现价值。Hidden 的主要增益在稀有 flip 的 PR-AUC：两个 hidden variant 均在 `4/5` folds
提高 PR；但 ROC 增益不稳定，且 hidden-context 在 math_train ROC 低于 context baseline。因此
原 strict gate `pilot_passed=false`，不能宣称 hidden 已稳定成为通用风险 router。

风险分桶本身有效：context 模型最低到最高五分位真实 flip 约从 `2.1%` 单调升到 `15.2%`。
使用 OOF context 分数做诊断性最大-ratio选择时，可以得到约 `avg ratio=0.061 / flip=2.0%`，
略优于固定 `ratio=0.05 / flip=2.23%`；信号存在，但优势仍小，尚不支持扩大正式 bank。

正式裁决：

```text
Action-Risk 可学习性：通过低成本在线 pilot 准入
Hidden 稳定主增益假设：未通过
下一 controller：causal context/action 主模型 + hidden 风险 veto 消融
在线暴露：每题最多一个窗口，不进入连续剪枝
正式扩样：等待 learned controller 在线优于 fixed baseline 后再决定
```

当前 OOF 分数只用于离线诊断，不能直接部署；下一步需训练最终 checkpoint，并只用 problem-level
OOF 策略模拟选择阈值。在线验收必须分别报告 context-only 与 context+hidden-veto，避免把 hidden 的
不稳定增益隐藏在组合模型中。

## 16. Learned Action-Risk Single-Window Controller

第一版 learned controller 已实现，使用 causal context/action 作为主风险模型，并将 hidden
限制为单向风险 veto。训练与 runtime 共用同一组特征：

```text
entropy
confidence
generated_tokens / max_new_tokens
log1p(generated_tokens)
candidate ratio
candidate ratio squared
```

部署 checkpoint 会硬校验 feature schema、`max_new_tokens=768`、ratio grid、合法边界
`32/96/160` 与 `window_tokens=16`。风险预测对 ratio 使用单调包络，避免局部预测反转让更大的
ratio 被错误视为更安全。Controller 每题只执行首次被接受的非零 action，之后永久 dense。

三档 operating point 使用 OOF 策略模拟自动校准，目标每题平均单窗口 action ratio 约为
`0.10/0.20/0.30`，且两个训练来源的 OOF problem flip 均不得超过 5%。Hidden-veto 每档独立
准入；不满足 exposure 保留与不恶化条件的档位不会进入在线任务。

```bash
# CPU：训练最终模型并生成 checkpoint/calibration
PYTHON=/home/cike/jjy/envs/rasp_qwen3/bin/python \
bash scripts/75_train_action_risk_controller.sh

# 八卡：dense + context-only 三档 + 合格 hidden-veto 档位
mkdir -p logs/07_stage_aware/08_action_risk_learned_single_window_pilot
nohup env GPU_IDS=0,1,2,3,4,5,6,7 \
PYTHON=/home/cike/jjy/envs/rasp_qwen3/bin/python \
bash scripts/77_eval_action_risk_learned_pilot.sh \
> logs/07_stage_aware/08_action_risk_learned_single_window_pilot/launcher.log 2>&1 &

# 所有 worker 完成后
PYTHON=/home/cike/jjy/envs/rasp_qwen3/bin/python \
python scripts/78_summarize_action_risk_learned_pilot.py
```

主要产物：

```text
runs/07_stage_aware/08_action_risk_learned_single_window_pilot/checkpoints/action_risk_single_window.joblib
runs/07_stage_aware/08_action_risk_learned_single_window_pilot/checkpoints/calibration_summary.json
runs/07_stage_aware/08_action_risk_learned_single_window_pilot/online_learned_summary.json
```

本轮通过条件要求至少一个 context-only 档位在 GSM8K 与 MATH500 同时满足：accuracy delta
`>= -5%`、dense-correct flip `<=10%`、平均 action ratio `>=0.08`、执行窗口非零，并严格改善
相近理论 exposure 的 fixed baseline 至少一个风险/准确率指标且另一指标不恶化。

## 17. Stage-Action-Risk v2：先修复时机监督

Learned single-window pilot 的 14/14 个任务完整，但正式 gate 未通过。更关键的是，context-only
与 hidden-veto 在所有题上都于 token `32` 立即执行，六组 hidden-veto 实验没有产生一次实际
veto。该结果不能解释为“32 是最佳 reasoning stage”，因为旧 uniform bank 对精确边界覆盖不完整：

```text
source      problems   b32   b96   b160
gsm8k       118        83    70    70
math_train  160        93    83    75
```

因此旧校准选择晚边界时，常常只是该题未采到早期边界。下一步不直接训练 stage controller，而是
建立严格可比较的精确边界 bank。Collector/validator 新增显式 `boundary_positions` 支持；v2 固定：

```text
boundary_positions = [32, 96, 160]
ratios = [0, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50]
stage annotation = action-preceding hidden stage probability + causal recent-text verification rule
```

数据准备只保留三个边界均完整、每个边界均有完整 ratio grid、且动作窗口满 16 token 的问题。
默认要求 GSM8K train 与隔离 math_train 各至少 100 个 complete problem。

```bash
mkdir -p logs/07_stage_aware/09_stage_action_risk_v2
nohup env GPU_IDS=0,1,2,3,4,5,6,7 \
PYTHON=/home/cike/jjy/envs/rasp_qwen3/bin/python \
bash scripts/80_collect_stage_action_risk_v2.sh \
> logs/07_stage_aware/09_stage_action_risk_v2/launcher.log 2>&1 &

# 八个 worker 完成后
PYTHON=/home/cike/jjy/envs/rasp_qwen3/bin/python \
bash scripts/81_prepare_stage_action_risk_v2_data.sh

PYTHON=/home/cike/jjy/envs/rasp_qwen3/bin/python \
bash scripts/82_analyze_stage_action_risk_v2.sh
```

分析在同一 problem-level 5-fold split 上比较：

```text
action-only
causal context + action
stage + causal context + action
hidden + stage + causal context + action
```

Stage controller 的硬准入要求 `stage+context+action` 相对 `context+action` 在至少 4/5 folds 上
同时改善 ROC-AUC 与 PR-AUC，并在两个数据来源上均改善。策略模拟还必须保持至少 80% exposure，
且两个来源的 flip 均不恶化、至少一个严格改善。

主要产物：

```text
runs/07_stage_aware/09_stage_action_risk_v2/data/01_stage_action_risk_data_summary.json
runs/07_stage_aware/09_stage_action_risk_v2/analysis/02_stage_action_risk_analysis.json
```

## 18. Stage-Action-Risk v2 结果：数据修复成功，Stage Gate 未通过

Stage-Action-Risk v2 已完成。54 个 shard validator 全部为 `ok`，最终精确边界 bank 包含：

```text
source       observed dense-correct   complete exact-boundary
gsm8k        188                      158
math_train   256                      236

complete problems   394
boundaries          1182
nonzero action rows 7092
positive flips      512
```

每个保留问题均完整包含 `32/96/160` 和全 ratio grid，因此上一轮候选边界缺失造成的伪等待问题
已经排除。

OOF 风险预测结果：

```text
variant                         ROC-AUC   PR-AUC
action-only                     0.6384    0.1033
causal context + action         0.6680    0.1273
stage + context + action        0.6658    0.1236
hidden + stage + context        0.6703    0.1383
```

Stage 相对 causal context 仅在 `1/5` folds 同时改善 ROC 与 PR，且没有同时改善 GSM8K 与
math_train，因此：

```text
stage_controller_training_allowed = false
```

当前 hard operational stage 分布也高度不平衡：

```text
reasoning      1045 / 1182
setup            86 / 1182
final            50 / 1182
verification      1 / 1182
```

`setup` 与 `reasoning` 的 ratio-risk 曲线接近，说明当前三分类 stage 在这些固定边界上没有提供
足够的动作风险区分度。Hidden+stage 的 PR-AUC 有提升，但跨 fold、跨数据集仍不足以直接部署。

### 18.1 First-Accepted 仍未学习等待

精确边界虽然完整，但当前策略模拟的规则是：按 `32 → 96 → 160` 检查，只要当前边界存在任一
低于风险阈值的非零 ratio，就立即执行其中最大的 ratio。

实际选择分布：

```text
policy                    token 32   token 96   token 160   dense
causal context + action   394        0          0           0
stage + context + action  391        3          0           0
hidden + stage + context  313        41         17          23
```

Hidden variant 确实产生了部分等待，但未降低两个数据来源的 problem flip。由此可见，当前瓶颈
不只是 stage representation；`first-accepted` 决策规则本身也没有比较“现在执行”和“等待未来
边界”的相对价值。只要低 ratio 在 token `32` 被预测为安全，策略就没有理由等待。

### 18.2 下一步：显式 Timing-Value 诊断

暂不训练 stage-gated online controller。下一步使用当前 exact-boundary paired bank 建立显式
时机任务：

```text
当前状态 + 当前候选动作
  → immediate action risk / utility

当前状态
  → future safe-action value

decision
  → act now 或 wait
```

未来边界的 counterfactual 结果只用于离线训练标签；在线推理仍只能读取当前已生成的因果信息。
首先进行 problem-level OOF 诊断，判断 causal hidden/stage/context 能否预测“等待是否比当前
动作更有价值”。只有 timing-value 模型稳定优于固定 token `32` 策略并在两个数据来源上改善，
才实现新的 online waiting controller。

## 19. Full-Trajectory Multi-Window 集成工作流

### 19.1 为什么改为 full trajectory

精确边界 v2 证明 `32/96/160` 数据完整性已经修复，但三个位置仍无法覆盖不同长度题目的完整推理
过程，也无法观察连续动作造成的状态分布变化。新工作流因此使用因果网格：

```text
decision_start  = 32
decision_stride = 32
window_tokens   = 16
tail_anchor     = diagnostic only
```

自然边界只在剩余 token 足以形成完整 affected-token 窗口时进入训练；临近 EOS 的 tail-anchor
即使窗口不完整也只保留为诊断。模型输入只允许 `generated_tokens`、entropy/confidence、causal
hidden、causal soft-stage probability 和 candidate ratio；完整轨迹长度、相对位置、hard stage
与 tail 标记全部隔离在 `diagnostic_only`。

### 19.2 Stage 与多窗口运行语义

可信 hard stage 语义为：

```text
accepted_reasoning / confident_setup / confident_final /
explicit_verification / unknown
```

低置信 argmax 必须为 `unknown`。Stage checkpoint 若实际使用完整轨迹位置，runtime 会硬失败；
风险分析只使用 soft-stage probability，不使用 hard stage gate。

`FixedMultiWindowController` 每次只剪一个 16-token 窗口，至少经过 16 token dense cooldown 后
才能再次动作，达到 `max_windows` 后永久 dense。每个 runtime event 记录动作索引、完整动作历史、
累计理论 exposure、hidden drift、boundary top-k logits 与完整 logits hash。

### 19.3 自动 Gate

总脚本依次运行：

```text
00_preflight
01_dense_bank_smoke       # 过采样后固定 4 dense-correct/source
02_dense_bank_pilot       # 过采样后固定 20 dense-correct/source + 5-fold grouped OOF
03_fixed_multi_window_dev # 选择最高通过 exposure 的 behavior policy
04_on_policy_smoke        # >=4 exact-replay valid problem/source
```

在上述阶段之前，总控还会运行 `python -m unittest discover -s tests`；失败时不会启动 GPU 工作。
Pilot 的 grouped OOF 会分别比较 `action-only / causal-context / soft-stage / hidden /
hidden+soft-stage`，因此能客观区分 hidden 本身的增益与 stage probability 的附加价值。

Fixed dev 仅使用与最终测试集隔离的 GSM8K train 和 `math_full_minus_math500`，方向门槛为：

```text
average theoretical exposure >= 0.02
paired accuracy delta >= -10%
dense-correct flip <= 15%
```

On-policy collector 只在此前动作已经影响状态且 dense cooldown 完成后分支候选动作。重放必须复现
完整 action schedule、token prefix、boundary 完整 logits hash/top-k、entropy/confidence 与 hidden；
候选动作因 EOS 合法提前终止时仍保留为风险标签，避免系统性漏掉可能由剪枝触发的终止；只有既未
完成窗口、也非 EOS 终止的动作才视为无效，且不会与 replay integrity failure 混为一谈。

### 19.4 执行与产物

```bash
mkdir -p logs/07_stage_aware/10_full_trajectory_multi_window
nohup env GPU_IDS=0,1,2,3,4,5,6,7 \
PYTHON=/home/cike/jjy/envs/rasp_qwen3/bin/python \
bash scripts/90_run_full_trajectory_multi_window_workflow.sh \
> logs/07_stage_aware/10_full_trajectory_multi_window/launcher.log 2>&1 &

tail -f logs/07_stage_aware/10_full_trajectory_multi_window/launcher.log
```

预计八卡全部 gate 通过时约 `5–14` 小时；任一 gate 失败会提前停止。最终首先查看：

```text
runs/07_stage_aware/10_full_trajectory_multi_window/workflow_gate.json
runs/07_stage_aware/10_full_trajectory_multi_window/final_workflow_summary.json
runs/07_stage_aware/10_full_trajectory_multi_window/final_workflow_report_zh.md
```

当前只报告 logical MLP mask 的理论 exposure，不宣称真实速度提升。本轮无论结果如何，
`learned_multi_window_allowed=false`；后续必须扩大 on-policy bank 并通过 grouped OOF。
