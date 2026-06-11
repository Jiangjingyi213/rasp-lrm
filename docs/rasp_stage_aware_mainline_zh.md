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

S1 已实现完整代码链路：

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
人工审计需要在 CSV 的 `audited_stage` 列填写五类标签；汇总器会检查规则伪标签与人工标签的一致率。
伪标签由当前 `rule_segmenter` 重新生成，数据摘要会记录它与 Motivation 旧缓存标签的不一致数。

比较五个 variant：

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
planning / derivation / final 三类在每个 seed 的 recall 均 >= 0.70
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
runs/07_stage_aware/01_s1_stage_probe/data/00_stage_data_summary.json
runs/07_stage_aware/01_s1_stage_probe/data/02_stage_manual_audit.csv
```

三张 GPU 并行训练：

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
runs/07_stage_aware/01_s1_stage_probe/comparison_summary.csv
runs/07_stage_aware/01_s1_stage_probe/s1_gate.json
```

只有 `s1_gate.json` 中 `s1_passed=true` 才实施并运行 S2 runtime stage sensitivity bank。S1 未通过
时不提前采集 S2、不实现 S3 controller，也不通过调整 gate 掩盖结果。
