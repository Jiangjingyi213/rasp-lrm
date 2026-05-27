from __future__ import annotations

import argparse
import csv
import html
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

DEFAULT_RUNS = [
    "runs/formal_qwen3_gsm8k_full_s0",
    "runs/formal_qwen3_gsm8k_full_s1",
    "runs/formal_qwen3_math500_full_s0",
    "runs/formal_qwen3_math500_full_s1",
]

FEATURE_ORDER = ["entropy", "confidence", "activation", "hidden", "combined"]
MODULE_ORDER = ["attention_heads", "attention_block", "mlp_channels", "mlp_block", "layer"]
RATIO_ORDER = [0.2, 0.4, 0.6]
STAGE_ORDER = ["understanding", "planning", "derivation", "verification", "final"]


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_json(path: str | Path, data: Any) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def fmt(x: Any, digits: int = 3) -> str:
    if x is None:
        return ""
    if isinstance(x, float):
        return f"{x:.{digits}f}"
    return str(x)


def pct(x: float | None, digits: int = 1) -> str:
    if x is None:
        return ""
    return f"{100 * x:.{digits}f}%"


def as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def markdown_table(rows: list[dict[str, Any]], fields: list[str], labels: dict[str, str] | None = None) -> str:
    labels = labels or {}
    if not rows:
        return ""
    def cell(value: Any) -> str:
        return str(value).replace("|", "\\|")

    header = [labels.get(field, field) for field in fields]
    out = ["| " + " | ".join(header) + " |", "| " + " | ".join(["---"] * len(fields)) + " |"]
    for row in rows:
        out.append("| " + " | ".join(cell(row.get(field, "")) for field in fields) + " |")
    return "\n".join(out)


def load_run(run_dir: Path) -> dict[str, Any]:
    trajectories = read_jsonl(run_dir / "01_trajectories.jsonl")
    dense_correct = sum(1 for row in trajectories if row.get("correct"))
    segments = read_jsonl(run_dir / "02_segments.jsonl")
    counterfactuals = read_jsonl(run_dir / "03_counterfactuals.jsonl")
    heatmap = read_json(run_dir / "06_heatmap_summary.json")
    entropy_auc = read_json(run_dir / "04_entropy_auc.json")
    probe = read_json(run_dir / "05_probe_metrics.json")
    oracle = read_json(run_dir / "03_counterfactuals.oracles.json")
    dataset = trajectories[0].get("dataset") if trajectories else run_dir.name.split("_")[2]
    split = "s1" if run_dir.name.endswith("_s1") else "s0" if run_dir.name.endswith("_s0") else ""
    return {
        "run_dir": str(run_dir),
        "dataset": dataset,
        "split": split,
        "trajectories": trajectories,
        "dense_total": len(trajectories),
        "dense_correct": dense_correct,
        "dense_accuracy": dense_correct / len(trajectories) if trajectories else None,
        "segments": segments,
        "segment_count": len(segments),
        "counterfactuals": counterfactuals,
        "counterfactual_count": len(counterfactuals),
        "heatmap": heatmap,
        "entropy_auc": entropy_auc,
        "probe": probe,
        "oracle": oracle,
    }


def weighted_rate(rows: list[dict[str, Any]]) -> float | None:
    n = sum(int(row.get("n", 0)) for row in rows)
    if n == 0:
        return None
    return sum(float(row["flip_rate"]) * int(row["n"]) for row in rows) / n


def aggregate_rates(rows: list[dict[str, Any]], keys: list[str]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[tuple(row.get(key) for key in keys)].append(row)
    out = []
    for values, group in groups.items():
        total_n = sum(int(item["n"]) for item in group)
        record = {key: value for key, value in zip(keys, values)}
        record["flip_rate"] = weighted_rate(group)
        record["n"] = total_n
        out.append(record)
    return sorted(out, key=lambda item: tuple(str(item.get(key)) for key in keys))


def collect_combined_tables(combined: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    summary = combined["combined_summary"]
    tables: dict[str, list[dict[str, Any]]] = {}
    for key in [
        "dataset_flip_rates",
        "module_flip_rates",
        "ratio_flip_rates",
        "segment_type_flip_rates",
        "dataset_module_flip_rates",
        "dataset_ratio_flip_rates",
        "module_ratio_flip_rates",
        "segment_type_module_ratio_flip_rates",
        "dataset_segment_type_module_ratio_flip_rates",
    ]:
        tables[key] = list(summary.get(key, []))
    tables["stage_module_flip_rates"] = aggregate_rates(
        tables["segment_type_module_ratio_flip_rates"], ["segment_type", "module"]
    )
    tables["stage_ratio_flip_rates"] = aggregate_rates(
        tables["segment_type_module_ratio_flip_rates"], ["segment_type", "ratio"]
    )
    return tables


def normalize_order(value: Any, order: list[Any]) -> tuple[int, str]:
    if value in order:
        return (order.index(value), str(value))
    return (len(order), str(value))


def rect(svg: list[str], x: float, y: float, w: float, h: float, fill: str, stroke: str = "none") -> None:
    svg.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" fill="{fill}" stroke="{stroke}"/>')


def text(svg: list[str], x: float, y: float, value: str, size: int = 12, anchor: str = "start", weight: str = "400") -> None:
    escaped = html.escape(value)
    svg.append(
        f'<text x="{x:.1f}" y="{y:.1f}" font-family="Arial, sans-serif" font-size="{size}" '
        f'font-weight="{weight}" text-anchor="{anchor}" fill="#17202a">{escaped}</text>'
    )


def color(value: float, min_value: float = 0.0, max_value: float = 0.9) -> str:
    if max_value <= min_value:
        t = 0.0
    else:
        t = max(0.0, min(1.0, (value - min_value) / (max_value - min_value)))
    # Light blue to warm red, restrained for paper figures.
    r = int(236 * t + 232 * (1 - t))
    g = int(94 * t + 242 * (1 - t))
    b = int(85 * t + 248 * (1 - t))
    return f"#{r:02x}{g:02x}{b:02x}"


def write_bar_svg(path: Path, rows: list[dict[str, Any]], label_key: str, value_key: str, title: str) -> None:
    ensure_dir(path.parent)
    rows = [row for row in rows if row.get(value_key) is not None]
    width = 920
    left = 220
    right = 60
    top = 64
    row_h = 34
    height = top + row_h * len(rows) + 50
    max_value = max([float(row[value_key]) for row in rows] + [1e-9])
    max_axis = max(0.7, math.ceil(max_value * 10) / 10)
    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">']
    rect(svg, 0, 0, width, height, "#ffffff")
    text(svg, 24, 34, title, 18, weight="700")
    axis_w = width - left - right
    for i in range(6):
        tick = max_axis * i / 5
        x = left + axis_w * tick / max_axis
        svg.append(f'<line x1="{x:.1f}" y1="{top-18}" x2="{x:.1f}" y2="{height-42}" stroke="#e5e7eb" stroke-width="1"/>')
        text(svg, x, height - 18, f"{tick:.1f}", 11, anchor="middle")
    for idx, row in enumerate(rows):
        y = top + idx * row_h
        value = float(row[value_key])
        label = str(row[label_key])
        bar_w = axis_w * value / max_axis
        text(svg, left - 12, y + 20, label, 12, anchor="end")
        rect(svg, left, y + 5, bar_w, 20, "#4f7cac")
        text(svg, left + bar_w + 8, y + 20, f"{value:.3f}", 12)
    svg.append("</svg>\n")
    path.write_text("\n".join(svg), encoding="utf-8")


def write_heatmap_svg(
    path: Path,
    rows: list[dict[str, Any]],
    x_key: str,
    y_key: str,
    value_key: str,
    title: str,
    x_order: list[Any] | None = None,
    y_order: list[Any] | None = None,
) -> None:
    ensure_dir(path.parent)
    x_values = sorted({row[x_key] for row in rows}, key=lambda v: normalize_order(v, x_order or []))
    y_values = sorted({row[y_key] for row in rows}, key=lambda v: normalize_order(v, y_order or []))
    lookup = {(row[x_key], row[y_key]): row for row in rows}
    cell_w = 142
    cell_h = 40
    left = 180
    top = 82
    width = left + cell_w * len(x_values) + 40
    height = top + cell_h * len(y_values) + 70
    max_value = max([float(row[value_key]) for row in rows if row.get(value_key) is not None] + [0.9])
    max_value = max(0.7, max_value)
    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">']
    rect(svg, 0, 0, width, height, "#ffffff")
    text(svg, 24, 34, title, 18, weight="700")
    for i, x_value in enumerate(x_values):
        text(svg, left + i * cell_w + cell_w / 2, top - 18, str(x_value), 12, anchor="middle", weight="700")
    for j, y_value in enumerate(y_values):
        text(svg, left - 12, top + j * cell_h + 25, str(y_value), 12, anchor="end", weight="700")
        for i, x_value in enumerate(x_values):
            row = lookup.get((x_value, y_value))
            value = as_float(row.get(value_key)) if row else None
            fill = color(value or 0.0, 0.0, max_value)
            x = left + i * cell_w
            y = top + j * cell_h
            rect(svg, x, y, cell_w, cell_h, fill, "#ffffff")
            label = "" if value is None else f"{value:.3f}"
            text(svg, x + cell_w / 2, y + 25, label, 12, anchor="middle")
    text(svg, left, height - 22, "Cell value = answer flip rate. Darker red means higher pruning sensitivity.", 11)
    svg.append("</svg>\n")
    path.write_text("\n".join(svg), encoding="utf-8")


def dense_tables(runs: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    run_rows = []
    dataset_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for run in runs:
        row = {
            "dataset": run["dataset"],
            "split": run["split"],
            "run_dir": run["run_dir"],
            "dense_total": run["dense_total"],
            "dense_correct": run["dense_correct"],
            "dense_accuracy": run["dense_accuracy"],
            "dense_correct_trajectories": run["segment_count"],
            "counterfactuals": run["counterfactual_count"],
            "overall_flip_rate": run["heatmap"]["overall_flip_rate"],
        }
        run_rows.append(row)
        dataset_groups[run["dataset"]].append(row)
    dataset_rows = []
    for dataset, group in sorted(dataset_groups.items()):
        total = sum(int(row["dense_total"]) for row in group)
        correct = sum(int(row["dense_correct"]) for row in group)
        cf = sum(int(row["counterfactuals"]) for row in group)
        dense_correct_trajectories = sum(int(row["dense_correct_trajectories"]) for row in group)
        flip = sum(float(row["overall_flip_rate"]) * int(row["counterfactuals"]) for row in group) / cf
        dataset_rows.append(
            {
                "dataset": dataset,
                "dense_total": total,
                "dense_correct": correct,
                "dense_accuracy": correct / total if total else None,
                "dense_correct_trajectories": dense_correct_trajectories,
                "counterfactuals": cf,
                "overall_flip_rate": flip,
            }
        )
    return run_rows, dataset_rows


def probe_tables(runs: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    per_run = []
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for run in runs:
        for feature in FEATURE_ORDER:
            metric = run["probe"].get(feature)
            if not metric:
                continue
            row = {
                "dataset": run["dataset"],
                "split": run["split"],
                "feature_set": feature,
                "roc_auc": metric.get("roc_auc"),
                "pr_auc": metric.get("pr_auc"),
                "val_loss": metric.get("val_loss"),
                "train_problem_count": metric.get("train_problem_count"),
                "val_problem_count": metric.get("val_problem_count"),
                "train_rows": metric.get("train_rows"),
                "val_rows": metric.get("val_rows"),
                "positive_rate_val": metric.get("positive_rate_val"),
            }
            per_run.append(row)
            grouped[(run["dataset"], feature)].append(row)
    summary = []
    for (dataset, feature), group in sorted(grouped.items(), key=lambda item: (item[0][0], normalize_order(item[0][1], FEATURE_ORDER))):
        summary.append(
            {
                "dataset": dataset,
                "feature_set": feature,
                "roc_auc_mean": mean(float(row["roc_auc"]) for row in group if row["roc_auc"] is not None),
                "pr_auc_mean": mean(float(row["pr_auc"]) for row in group if row["pr_auc"] is not None),
                "n_splits": len(group),
            }
        )
    return per_run, summary


def oracle_table(combined: dict[str, Any]) -> list[dict[str, Any]]:
    oracles = combined["combined_summary"]["oracles"]
    return [
        {"policy": "static oracle", "flip_rate": oracles.get("static_oracle_flip_rate")},
        {"policy": "prompt oracle", "flip_rate": oracles.get("prompt_oracle_flip_rate")},
        {"policy": "step oracle", "flip_rate": oracles.get("step_oracle_flip_rate")},
        {"policy": "macro prompt oracle", "flip_rate": oracles.get("macro_prompt_oracle_flip_rate")},
        {"policy": "macro step oracle", "flip_rate": oracles.get("macro_step_oracle_flip_rate")},
    ]


def top_bottom_actions(combined: dict[str, Any], k: int = 10) -> list[dict[str, Any]]:
    choice_rates = combined["combined_summary"]["oracles"].get("choice_flip_rates", {})
    items = sorted(choice_rates.items(), key=lambda item: item[1], reverse=True)
    rows = []
    for rank, (choice, flip_rate) in enumerate(items[:k], start=1):
        rows.append({"group": "highest_risk", "rank": rank, "choice": choice, "flip_rate": flip_rate})
    for rank, (choice, flip_rate) in enumerate(reversed(items[-k:]), start=1):
        rows.append({"group": "lowest_risk", "rank": rank, "choice": choice, "flip_rate": flip_rate})
    return rows


def write_report(
    output: Path,
    combined: dict[str, Any],
    run_dense_rows: list[dict[str, Any]],
    dataset_dense_rows: list[dict[str, Any]],
    tables: dict[str, list[dict[str, Any]]],
    oracle_rows: list[dict[str, Any]],
    probe_summary: list[dict[str, Any]],
    top_bottom: list[dict[str, Any]],
) -> None:
    ensure_dir(output.parent)
    summary = combined["combined_summary"]
    report_dense = [
        {
            "dataset": row["dataset"],
            "dense acc": pct(row["dense_accuracy"]),
            "dense correct": f'{row["dense_correct"]}/{row["dense_total"]}',
            "counterfactuals": row["counterfactuals"],
            "flip rate": pct(row["overall_flip_rate"]),
        }
        for row in dataset_dense_rows
    ]
    report_module = [
        {"module": row["module"], "flip rate": pct(row["flip_rate"]), "n": row["n"]}
        for row in sorted(tables["module_flip_rates"], key=lambda row: normalize_order(row["module"], MODULE_ORDER))
    ]
    report_ratio = [
        {"ratio": row["ratio"], "flip rate": pct(row["flip_rate"]), "n": row["n"]}
        for row in sorted(tables["ratio_flip_rates"], key=lambda row: normalize_order(row["ratio"], RATIO_ORDER))
    ]
    report_stage = [
        {"stage": row["segment_type"], "flip rate": pct(row["flip_rate"]), "n": row["n"]}
        for row in sorted(tables["segment_type_flip_rates"], key=lambda row: normalize_order(row["segment_type"], STAGE_ORDER))
    ]
    report_oracle = [
        {"policy": row["policy"], "flip rate": pct(row["flip_rate"])}
        for row in oracle_rows
    ]
    report_probe = [
        {
            "dataset": row["dataset"],
            "feature": row["feature_set"],
            "ROC-AUC": fmt(row["roc_auc_mean"]),
            "PR-AUC": fmt(row["pr_auc_mean"]),
        }
        for row in probe_summary
    ]
    high = [row for row in top_bottom if row["group"] == "highest_risk"][:5]
    low = [row for row in top_bottom if row["group"] == "lowest_risk"][:5]

    text_out = f"""# Qwen3 Motivation Experiment Report

## 1. Experiment Purpose

本实验的目标是验证一个核心 motivation：LRM 的结构剪枝风险不是一个静态、全局、只由剪枝率决定的问题，而是与数据集难度、推理阶段、模块类型、剪枝粒度和当前 reasoning state 共同相关。因此，LRM pruning 需要 reasoning-aware 的动态策略，而不是单一静态剪枝策略。

本轮正式实验使用 `Qwen/Qwen3-1.7B`，在 GSM8K 与 MATH500 上生成 dense reasoning trajectories。随后只保留 dense 原本答对的样本，对每个 reasoning segment 施加多种 counterfactual structured pruning action，并观察最终答案是否从正确翻转为错误。

## 2. Pipeline

1. **Dense generation**：运行未剪枝 Qwen3，得到 `01_trajectories.jsonl`。每行包含问题、gold answer、模型 completion、抽取答案和 `correct`。
2. **Dense-correct filtering and segmentation**：只对 dense 正确样本进入后续分析，生成 `02_segments.jsonl`。segment 是 rule-based automatic assignment，不是人工标注或 LLM classifier。
3. **Counterfactual pruning**：对每个 `(problem, segment, pruning action)` 重新 continuation，得到 `03_counterfactuals.jsonl`。如果 counterfactual answer 与 baseline answer 不一致，标记为 `flipped=true`。
4. **Oracle analysis**：从同一张 counterfactual table 计算 static / prompt / step oracle。它们分别表示固定全局 action、每题选择 action、每步选择 action 的上限。
5. **Uncertainty baselines**：计算 entropy/confidence 对 flip risk 的 ROC-AUC、PR-AUC，写入 `04_entropy_auc.json`。
6. **Risk probe**：训练 problem-level split 的线性 probe，比较 entropy、confidence、activation、hidden、combined feature set，写入 `05_probe_metrics.json`。
7. **Heatmap summary**：聚合 dataset、module、ratio、stage、module × ratio、stage × module 等 flip rate，写入 `06_heatmap_summary.json` 和本报告生成的表格/图。

## 3. Dense Baseline and Counterfactual Scale

{markdown_table(report_dense, ["dataset", "dense acc", "dense correct", "counterfactuals", "flip rate"])}

Combined counterfactual actions: **{summary["n"]}**. Combined overall flip rate: **{pct(summary["overall_flip_rate"])}**.

这个结果说明 Qwen3-1.7B 的 dense baseline 已经有足够推理能力，且在 dense-correct subset 上存在大规模 answer flip。MATH500 的 flip rate 明显高于 GSM8K，说明复杂数学推理对结构扰动更敏感。

## 4. Module Sensitivity

{markdown_table(report_module, ["module", "flip rate", "n"])}

模块级结果显示，完整 layer 与 MLP block 的风险最高，attention heads 与 MLP channels 相对更温和。这支持一个直接结论：不同结构单元不能用同一个风险假设处理，RASP 后续需要 module-aware routing。

## 5. Pruning Ratio Sensitivity

{markdown_table(report_ratio, ["ratio", "flip rate", "n"])}

`r=0.60` 明显更危险，但 `r=0.20` 与 `r=0.40` 不是严格单调。这说明风险不是只由 ratio 决定，还受到 module、layer group、stage 和数据集难度影响。

## 6. Reasoning Stage Sensitivity

{markdown_table(report_stage, ["stage", "flip rate", "n"])}

Stage 是自动 rule-based assignment，因此论文中应明确说明是 heuristic stage assignment。当前结果适合作为 motivation signal：不同推理阶段的剪枝敏感性确实不同。特别是 planning/derivation 与 verification 更敏感，而 final answer 段落 flip rate 较低，可能因为答案已经基本形成。

## 7. Oracle Gap

{markdown_table(report_oracle, ["policy", "flip rate"])}

Static oracle 已经能找到高风险 action，但 prompt oracle 更强，step oracle 最强。这个 gap 是本实验最关键的 motivation：LRM pruning 不应只看 prompt 或全局预算，而应看 reasoning step 的当前状态。

## 8. Probe Comparison

{markdown_table(report_probe, ["dataset", "feature", "ROC-AUC", "PR-AUC"])}

Hidden-state 与 combined probe 明显优于 entropy/confidence。这个结果说明剪枝风险不能只由 next-token entropy 或 confidence 判断；模型内部 hidden state 包含更强的 reasoning-criticality 信号。

## 9. Highest and Lowest Risk Actions

Highest-risk actions:

{markdown_table([{"rank": r["rank"], "choice": r["choice"], "flip rate": pct(r["flip_rate"])} for r in high], ["rank", "choice", "flip rate"])}

Lowest-risk actions:

{markdown_table([{"rank": r["rank"], "choice": r["choice"], "flip rate": pct(r["flip_rate"])} for r in low], ["rank", "choice", "flip rate"])}

这组结果说明 action space 内部有显著风险分层。后续 RASP-Zero 可以优先避免高风险 action，并在低风险 stage 或低风险 probe score 下选择更激进的剪枝。

## 10. Paper-ready Figures

本脚本生成的 figure 文件位于 `runs/motivation_analysis/figures/`：

- `dataset_flip_rates.svg`
- `module_flip_rates.svg`
- `ratio_flip_rates.svg`
- `stage_flip_rates.svg`
- `oracle_gap.svg`
- `probe_comparison.svg`
- `module_ratio_heatmap.svg`
- `stage_module_heatmap.svg`
- `dataset_module_heatmap.svg`
- `stage_ratio_heatmap.svg`

## 11. Main Takeaways

1. Qwen3-1.7B 在 GSM8K/MATH500 上的 dense accuracy 足以支持 LRM pruning 研究。
2. Dense-correct trajectories 在结构扰动下大量 answer flip，说明 reasoning 对 pruning 高敏感。
3. MATH500 比 GSM8K 更脆弱，说明数据集难度会改变剪枝风险。
4. Layer 和 MLP block 风险最高，head/channel 粒度更温和，说明 module granularity 很重要。
5. Stage sensitivity 存在，说明 pruning policy 应该跟随 reasoning process 变化。
6. Step oracle 明显强于 static/prompt oracle，直接支持 step-level dynamic pruning。
7. Hidden-state probe 明显强于 entropy/confidence，说明 RASP 的 risk estimator 应该使用内部表征。

## 12. Limitations

- Stage assignment 是 rule-based heuristic，应在论文中明确说明，并建议抽样人工检查 20-50 条。
- Counterfactual pruning 是在线 ablation/continuation 风险评估，不等价于真实导出后的加速模型。
- 当前结果用于 motivation 与 policy simulation；真正部署速度需要在 GRIFFIN/FLAP/RASP 实现中单独测量 latency、memory 和 tokens/s。
"""
    output.write_text(text_out, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--combined", default="runs/formal_qwen3_gsm8k_math500_combined.json")
    parser.add_argument("--run-dirs", nargs="*", default=DEFAULT_RUNS)
    parser.add_argument("--output-dir", default="runs/motivation_analysis")
    args = parser.parse_args()

    output_dir = ensure_dir(args.output_dir)
    table_dir = ensure_dir(output_dir / "tables")
    figure_dir = ensure_dir(output_dir / "figures")

    combined = read_json(args.combined)
    runs = [load_run(Path(path)) for path in args.run_dirs]
    run_dense_rows, dataset_dense_rows = dense_tables(runs)
    tables = collect_combined_tables(combined)
    probe_per_run, probe_summary = probe_tables(runs)
    oracle_rows = oracle_table(combined)
    top_bottom = top_bottom_actions(combined)

    # Raw and formatted tables.
    write_csv(table_dir / "dense_runs.csv", run_dense_rows)
    write_csv(table_dir / "dense_dataset_summary.csv", dataset_dense_rows)
    write_csv(table_dir / "dataset_flip_rates.csv", tables["dataset_flip_rates"])
    write_csv(table_dir / "module_flip_rates.csv", tables["module_flip_rates"])
    write_csv(table_dir / "ratio_flip_rates.csv", tables["ratio_flip_rates"])
    write_csv(table_dir / "stage_flip_rates.csv", tables["segment_type_flip_rates"])
    write_csv(table_dir / "dataset_module_flip_rates.csv", tables["dataset_module_flip_rates"])
    write_csv(table_dir / "dataset_ratio_flip_rates.csv", tables["dataset_ratio_flip_rates"])
    write_csv(table_dir / "module_ratio_heatmap.csv", tables["module_ratio_flip_rates"])
    write_csv(table_dir / "stage_module_heatmap.csv", tables["stage_module_flip_rates"])
    write_csv(table_dir / "stage_ratio_heatmap.csv", tables["stage_ratio_flip_rates"])
    write_csv(table_dir / "stage_module_ratio.csv", tables["segment_type_module_ratio_flip_rates"])
    write_csv(table_dir / "dataset_stage_module_ratio.csv", tables["dataset_segment_type_module_ratio_flip_rates"])
    write_csv(table_dir / "oracle_gap.csv", oracle_rows)
    write_csv(table_dir / "probe_per_run.csv", probe_per_run)
    write_csv(table_dir / "probe_comparison.csv", probe_summary)
    write_csv(table_dir / "top_bottom_actions.csv", top_bottom)

    # Figures.
    write_bar_svg(figure_dir / "dataset_flip_rates.svg", tables["dataset_flip_rates"], "dataset", "flip_rate", "Dataset-level pruning sensitivity")
    write_bar_svg(
        figure_dir / "module_flip_rates.svg",
        sorted(tables["module_flip_rates"], key=lambda row: normalize_order(row["module"], MODULE_ORDER)),
        "module",
        "flip_rate",
        "Module-level pruning sensitivity",
    )
    write_bar_svg(
        figure_dir / "ratio_flip_rates.svg",
        sorted(tables["ratio_flip_rates"], key=lambda row: normalize_order(row["ratio"], RATIO_ORDER)),
        "ratio",
        "flip_rate",
        "Pruning-ratio sensitivity",
    )
    write_bar_svg(
        figure_dir / "stage_flip_rates.svg",
        sorted(tables["segment_type_flip_rates"], key=lambda row: normalize_order(row["segment_type"], STAGE_ORDER)),
        "segment_type",
        "flip_rate",
        "Reasoning-stage pruning sensitivity",
    )
    write_bar_svg(figure_dir / "oracle_gap.svg", oracle_rows[:3], "policy", "flip_rate", "Static vs prompt vs step oracle")
    write_bar_svg(
        figure_dir / "probe_comparison.svg",
        [
            {"label": f'{row["dataset"]}:{row["feature_set"]}', "roc_auc_mean": row["roc_auc_mean"]}
            for row in probe_summary
        ],
        "label",
        "roc_auc_mean",
        "Risk-prediction probe comparison (ROC-AUC)",
    )
    write_heatmap_svg(
        figure_dir / "module_ratio_heatmap.svg",
        tables["module_ratio_flip_rates"],
        "ratio",
        "module",
        "flip_rate",
        "Module x pruning ratio flip-rate heatmap",
        x_order=RATIO_ORDER,
        y_order=MODULE_ORDER,
    )
    write_heatmap_svg(
        figure_dir / "stage_module_heatmap.svg",
        tables["stage_module_flip_rates"],
        "module",
        "segment_type",
        "flip_rate",
        "Reasoning stage x module flip-rate heatmap",
        x_order=MODULE_ORDER,
        y_order=STAGE_ORDER,
    )
    write_heatmap_svg(
        figure_dir / "dataset_module_heatmap.svg",
        tables["dataset_module_flip_rates"],
        "module",
        "dataset",
        "flip_rate",
        "Dataset x module flip-rate heatmap",
        x_order=MODULE_ORDER,
    )
    write_heatmap_svg(
        figure_dir / "stage_ratio_heatmap.svg",
        tables["stage_ratio_flip_rates"],
        "ratio",
        "segment_type",
        "flip_rate",
        "Reasoning stage x pruning ratio flip-rate heatmap",
        x_order=RATIO_ORDER,
        y_order=STAGE_ORDER,
    )

    report_path = output_dir / "motivation_report.md"
    write_report(report_path, combined, run_dense_rows, dataset_dense_rows, tables, oracle_rows, probe_summary, top_bottom)
    write_json(
        output_dir / "analysis_manifest.json",
        {
            "combined": args.combined,
            "run_dirs": args.run_dirs,
            "output_dir": str(output_dir),
            "tables": sorted(str(path.relative_to(output_dir)) for path in table_dir.glob("*.csv")),
            "figures": sorted(str(path.relative_to(output_dir)) for path in figure_dir.glob("*.svg")),
            "report": str(report_path.relative_to(output_dir)),
        },
    )
    print(f"Wrote motivation analysis to {output_dir}")


if __name__ == "__main__":
    main()
