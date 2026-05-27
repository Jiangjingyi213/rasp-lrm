#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable


def require_plotting_libs():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import pandas as pd
        import seaborn as sns
    except ImportError as exc:
        raise SystemExit(
            "Missing plotting dependencies. Install them with:\n"
            "  pip install pandas matplotlib seaborn\n"
            "or install this repo's updated requirements.txt on the server."
        ) from exc
    return plt, pd, sns


RUN_DIRS = [
    Path("runs/formal_qwen3_gsm8k_full_s0"),
    Path("runs/formal_qwen3_gsm8k_full_s1"),
    Path("runs/formal_qwen3_math500_full_s0"),
    Path("runs/formal_qwen3_math500_full_s1"),
]

STAGE_ORDER = ["understanding", "planning", "derivation", "verification", "final"]
MODULE_ORDER = ["attention_heads", "attention_block", "mlp_channels", "mlp_block", "layer"]
FEATURE_ORDER = ["entropy", "confidence", "activation", "hidden", "combined"]
RATIO_ORDER = [0.2, 0.4, 0.6]

PALETTE = {
    "gsm8k": "#4C78A8",
    "math500": "#E45756",
    "attention_heads": "#72B7B2",
    "attention_block": "#54A24B",
    "mlp_channels": "#F58518",
    "mlp_block": "#B279A2",
    "layer": "#E45756",
    "entropy": "#9D755D",
    "confidence": "#BAB0AC",
    "activation": "#F2CF5B",
    "hidden": "#4C78A8",
    "combined": "#E45756",
}


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def savefig(fig, output: Path, stem: str, formats: Iterable[str], dpi: int) -> None:
    for fmt in formats:
        path = output / f"{stem}.{fmt}"
        fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")


def style(sns, plt) -> None:
    sns.set_theme(
        context="paper",
        style="whitegrid",
        font="DejaVu Sans",
        rc={
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.titleweight": "bold",
            "axes.labelsize": 11,
            "axes.titlesize": 12,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "figure.titlesize": 13,
            "savefig.dpi": 300,
        },
    )
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42
    plt.rcParams["svg.fonttype"] = "none"


def load_tables(pd, analysis_dir: Path) -> dict[str, object]:
    table_dir = analysis_dir / "tables"
    tables = {}
    for path in table_dir.glob("*.csv"):
        tables[path.stem] = pd.read_csv(path)
    return tables


def load_counterfactuals(pd, run_dirs: list[Path]) -> object:
    frames = []
    for run_dir in run_dirs:
        path = run_dir / "03_counterfactuals.jsonl"
        if not path.exists():
            continue
        rows = read_jsonl(path)
        if rows:
            frames.append(pd.DataFrame(rows))
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def fig1_stage_sensitivity(plt, sns, tables: dict, output: Path, formats: list[str], dpi: int) -> None:
    stage_module = tables["stage_module_heatmap"].copy()
    stage_ratio = tables["stage_ratio_heatmap"].copy()
    stage_module["segment_type"] = stage_module["segment_type"].astype(str)
    stage_ratio["segment_type"] = stage_ratio["segment_type"].astype(str)

    module_pivot = stage_module.pivot(index="segment_type", columns="module", values="flip_rate").reindex(
        index=STAGE_ORDER, columns=MODULE_ORDER
    )
    ratio_pivot = stage_ratio.pivot(index="segment_type", columns="ratio", values="flip_rate").reindex(
        index=STAGE_ORDER, columns=RATIO_ORDER
    )

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.4), gridspec_kw={"width_ratios": [1.6, 1.0]})
    cmap = sns.color_palette("rocket_r", as_cmap=True)
    sns.heatmap(
        module_pivot,
        ax=axes[0],
        cmap=cmap,
        vmin=0.15,
        vmax=0.72,
        annot=True,
        fmt=".2f",
        linewidths=0.7,
        linecolor="white",
        cbar_kws={"label": "Answer flip rate"},
    )
    axes[0].set_title("Stage x pruning module")
    axes[0].set_xlabel("")
    axes[0].set_ylabel("Reasoning stage")
    sns.heatmap(
        ratio_pivot,
        ax=axes[1],
        cmap=cmap,
        vmin=0.15,
        vmax=0.72,
        annot=True,
        fmt=".2f",
        linewidths=0.7,
        linecolor="white",
        cbar=False,
    )
    axes[1].set_title("Stage x pruning ratio")
    axes[1].set_xlabel("")
    axes[1].set_ylabel("")
    fig.suptitle("LRM reasoning pruning sensitivity is stage-dependent", y=1.03, fontweight="bold")
    fig.text(
        0.02,
        -0.02,
        "Rows are automatically assigned reasoning stages. Values are answer flip rates on dense-correct trajectories.",
        fontsize=9,
        color="#4b5563",
    )
    fig.tight_layout()
    savefig(fig, output, "fig1_reasoning_stage_sensitivity_heatmaps", formats, dpi)
    plt.close(fig)


def fig1b_module_ratio(plt, sns, tables: dict, output: Path, formats: list[str], dpi: int) -> None:
    df = tables["module_ratio_heatmap"].copy()
    pivot = df.pivot(index="module", columns="ratio", values="flip_rate").reindex(index=MODULE_ORDER, columns=RATIO_ORDER)
    fig, ax = plt.subplots(figsize=(6.1, 3.9))
    sns.heatmap(
        pivot,
        ax=ax,
        cmap=sns.color_palette("mako_r", as_cmap=True),
        vmin=0.25,
        vmax=0.72,
        annot=True,
        fmt=".2f",
        linewidths=0.8,
        linecolor="white",
        cbar_kws={"label": "Answer flip rate"},
    )
    ax.set_title("Module x ratio counterfactual sensitivity")
    ax.set_xlabel("Pruning ratio")
    ax.set_ylabel("Pruning unit")
    fig.tight_layout()
    savefig(fig, output, "fig1b_module_ratio_heatmap", formats, dpi)
    plt.close(fig)


def fig2_oracle_gap(plt, sns, tables: dict, output: Path, formats: list[str], dpi: int) -> None:
    df = tables["oracle_gap"].copy()
    df = df[df["policy"].isin(["static oracle", "prompt oracle", "step oracle"])]
    order = ["static oracle", "prompt oracle", "step oracle"]
    df["policy"] = df["policy"].astype(str)
    fig, ax = plt.subplots(figsize=(5.5, 3.8))
    colors = ["#8E9AAF", "#4C78A8", "#E45756"]
    sns.barplot(data=df, x="policy", y="flip_rate", order=order, palette=colors, ax=ax)
    for patch, (_, row) in zip(ax.patches, df.set_index("policy").loc[order].reset_index().iterrows()):
        ax.text(
            patch.get_x() + patch.get_width() / 2,
            patch.get_height() + 0.015,
            f"{row['flip_rate']:.2f}",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )
    static = float(df[df["policy"] == "static oracle"]["flip_rate"].iloc[0])
    step = float(df[df["policy"] == "step oracle"]["flip_rate"].iloc[0])
    ax.annotate(
        f"+{step - static:.2f}",
        xy=(2, step),
        xytext=(0, static + 0.08),
        arrowprops=dict(arrowstyle="->", lw=1.5, color="#17202a"),
        ha="center",
        fontsize=10,
        color="#17202a",
    )
    ax.set_ylim(0, min(1.0, max(df["flip_rate"]) + 0.12))
    ax.set_xlabel("")
    ax.set_ylabel("Oracle answer flip rate")
    ax.set_title("Step-level oracle reveals dynamic pruning headroom")
    ax.set_xticklabels(["Static", "Prompt-level", "Step-level"])
    fig.tight_layout()
    savefig(fig, output, "fig2_oracle_gap", formats, dpi)
    plt.close(fig)


def fig3_entropy_not_enough(plt, sns, pd, tables: dict, cf: object, output: Path, formats: list[str], dpi: int) -> None:
    probe = tables["probe_comparison"].copy()
    probe = probe[probe["feature_set"].isin(FEATURE_ORDER)]
    probe["feature_set"] = pd.Categorical(probe["feature_set"], FEATURE_ORDER, ordered=True)
    probe = probe.sort_values(["dataset", "feature_set"])

    decile_rows = []
    if not cf.empty and "entropy" in cf.columns:
        tmp = cf[["dataset", "entropy", "flipped"]].dropna().copy()
        for dataset, group in tmp.groupby("dataset"):
            # qcut can collapse duplicate bins for very peaked entropy distributions.
            group = group.copy()
            group["entropy_decile"] = pd.qcut(group["entropy"], q=10, labels=False, duplicates="drop")
            by = group.groupby("entropy_decile", observed=True).agg(
                entropy_mean=("entropy", "mean"),
                flip_rate=("flipped", "mean"),
                n=("flipped", "size"),
            )
            by["dataset"] = dataset
            decile_rows.append(by.reset_index())
    deciles = pd.concat(decile_rows, ignore_index=True) if decile_rows else pd.DataFrame()

    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.2), gridspec_kw={"width_ratios": [1.1, 1.4]})
    if not deciles.empty:
        sns.lineplot(
            data=deciles,
            x="entropy_mean",
            y="flip_rate",
            hue="dataset",
            marker="o",
            palette={k: PALETTE[k] for k in ["gsm8k", "math500"] if k in set(deciles["dataset"])},
            ax=axes[0],
        )
    axes[0].set_title("Entropy vs pruning risk")
    axes[0].set_xlabel("Mean prefix entropy (decile)")
    axes[0].set_ylabel("Answer flip rate")
    axes[0].legend(title="Dataset", frameon=False)

    sns.barplot(
        data=probe,
        x="feature_set",
        y="roc_auc_mean",
        hue="dataset",
        palette={k: PALETTE[k] for k in ["gsm8k", "math500"] if k in set(probe["dataset"])},
        ax=axes[1],
    )
    axes[1].axhline(0.5, color="#6b7280", lw=1.0, ls="--")
    axes[1].set_title("Risk predictor comparison")
    axes[1].set_xlabel("")
    axes[1].set_ylabel("ROC-AUC")
    axes[1].tick_params(axis="x", rotation=25)
    axes[1].legend(title="Dataset", frameon=False, loc="lower right")
    fig.suptitle("Entropy alone is not enough for pruning-risk estimation", y=1.03, fontweight="bold")
    fig.tight_layout()
    savefig(fig, output, "fig3_entropy_not_enough", formats, dpi)
    plt.close(fig)


def fig5_proxy_pareto(plt, sns, tables: dict, output: Path, formats: list[str], dpi: int) -> None:
    df = tables["module_ratio_heatmap"].copy()
    df["answer_retention"] = 1.0 - df["flip_rate"]
    df["activated_proxy"] = 1.0 - df["ratio"]
    df["module"] = pd_categorical(df["module"], MODULE_ORDER)
    fig, ax = plt.subplots(figsize=(6.6, 4.6))
    sns.scatterplot(
        data=df,
        x="activated_proxy",
        y="answer_retention",
        hue="module",
        style="module",
        s=95,
        palette={module: PALETTE[module] for module in MODULE_ORDER},
        ax=ax,
    )
    for _, row in df.iterrows():
        ax.text(row["activated_proxy"] + 0.008, row["answer_retention"], f"r={row['ratio']:.1f}", fontsize=7)
    ax.scatter([1.0], [1.0], marker="*", s=160, color="#111827", label="dense")
    ax.text(1.0, 0.985, "Dense", ha="right", va="top", fontsize=9, fontweight="bold")
    ax.set_xlim(0.34, 1.04)
    ax.set_ylim(max(0.25, df["answer_retention"].min() - 0.08), 1.03)
    ax.set_xlabel("Activated structure proxy (1 - pruning ratio)")
    ax.set_ylabel("Answer retention on dense-correct subset (1 - flip rate)")
    ax.set_title("Counterfactual action frontier (proxy, not measured latency)")
    ax.legend(frameon=False, bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()
    savefig(fig, output, "fig5_counterfactual_proxy_pareto", formats, dpi)
    plt.close(fig)


def pd_categorical(series, order):
    # Kept tiny helper so this script does not need pandas at module import time.
    import pandas as pd

    return pd.Categorical(series, order, ordered=True)


def write_manifest(output: Path, generated: list[str]) -> None:
    text = """# Paper Figure Manifest

Generated from `runs/formal_qwen3_gsm8k_math500_combined.json` and the four formal Qwen3 motivation runs.

## Figures generated from current data

- **Fig. 1** `fig1_reasoning_stage_sensitivity_heatmaps`: stage x module and stage x ratio pruning sensitivity.
- **Fig. 1b** `fig1b_module_ratio_heatmap`: module x ratio sensitivity.
- **Fig. 2** `fig2_oracle_gap`: static / prompt / step oracle gap.
- **Fig. 3** `fig3_entropy_not_enough`: entropy-risk curve and probe ROC-AUC comparison.
- **Fig. 5 proxy** `fig5_counterfactual_proxy_pareto`: counterfactual action frontier using activated-structure proxy, not true latency/FLOPs.

## Not generated yet

- **Fig. 4 FFN flocking across reasoning stages** is not generated because the current runs do not store per-stage FFN top-neuron sets or Jaccard similarity inputs. To make this figure rigorously, add hooks that record top-k FFN intermediate neuron indices per segment/stage, then compute within-stage and cross-stage Jaccard similarity.
- **Final latency/FLOPs Pareto** should be generated after Dense, Static, GRIFFIN, FLAP, and RASP runs all report real accuracy and measured latency/FLOPs/activated-parameter metrics.

## Files

"""
    for name in generated:
        text += f"- `{name}`\n"
    (output / "README.md").write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis-dir", default="runs/motivation_analysis")
    parser.add_argument("--combined", default="runs/formal_qwen3_gsm8k_math500_combined.json")
    parser.add_argument("--output-dir", default="runs/motivation_analysis/paper_figures")
    parser.add_argument("--formats", nargs="+", default=["png", "pdf", "svg"], choices=["png", "pdf", "svg"])
    parser.add_argument("--dpi", type=int, default=400)
    parser.add_argument("--skip-counterfactual-scan", action="store_true")
    args = parser.parse_args()

    plt, pd, sns = require_plotting_libs()
    style(sns, plt)

    analysis_dir = Path(args.analysis_dir)
    output = ensure_dir(Path(args.output_dir))
    tables = load_tables(pd, analysis_dir)
    run_dirs = RUN_DIRS
    cf = pd.DataFrame() if args.skip_counterfactual_scan else load_counterfactuals(pd, run_dirs)

    fig1_stage_sensitivity(plt, sns, tables, output, args.formats, args.dpi)
    fig1b_module_ratio(plt, sns, tables, output, args.formats, args.dpi)
    fig2_oracle_gap(plt, sns, tables, output, args.formats, args.dpi)
    fig3_entropy_not_enough(plt, sns, pd, tables, cf, output, args.formats, args.dpi)
    fig5_proxy_pareto(plt, sns, tables, output, args.formats, args.dpi)

    generated = sorted(path.name for path in output.iterdir() if path.suffix in {".png", ".pdf", ".svg"})
    write_manifest(output, generated)
    print(f"Wrote {len(generated)} figure files to {output}")


if __name__ == "__main__":
    main()
