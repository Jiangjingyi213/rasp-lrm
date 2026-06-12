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
