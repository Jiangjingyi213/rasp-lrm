#!/usr/bin/env python3
"""Create the compact, publication-ready diagnostic oracle-gap figure."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter


POLICY_ORDER = ["static oracle", "prompt oracle", "step oracle"]
DISPLAY_LABELS = ["Global action\n(Static)", "Per problem\n(Prompt-level)", "Per problem-step\n(Step-level)"]
COLORS = ["#A7B0BE", "#5F82A6", "#D46A6A"]


def read_oracle_values(path: Path) -> list[float]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = {row["policy"]: float(row["flip_rate"]) for row in csv.DictReader(handle)}
    missing = [policy for policy in POLICY_ORDER if policy not in rows]
    if missing:
        raise ValueError(f"Missing oracle policies: {', '.join(missing)}")
    return [rows[policy] for policy in POLICY_ORDER]


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 7,
            "axes.titlesize": 11,
            "axes.labelsize": 8,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7,
            "axes.linewidth": 0.7,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.dpi": 600,
        }
    )


def draw(values: list[float], output_dir: Path) -> None:
    configure_style()
    output_dir.mkdir(parents=True, exist_ok=True)

    # 183 mm wide: suitable for a double-column AAAI figure while remaining compact.
    fig, ax = plt.subplots(figsize=(7.20, 2.55))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    positions = range(len(values))
    bars = ax.bar(
        positions,
        values,
        width=0.58,
        color=COLORS,
        edgecolor="white",
        linewidth=0.8,
        zorder=3,
    )

    ax.set_ylim(0.0, 0.92)
    ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8])
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
    ax.set_ylabel("Maximum answer-flip rate")
    ax.set_xticks(list(positions), DISPLAY_LABELS)
    ax.tick_params(axis="x", length=0, pad=5)
    ax.tick_params(axis="y", length=0, pad=4)
    ax.grid(axis="y", color="#D9DEE5", linewidth=0.65, zorder=0)
    ax.spines["left"].set_color("#AAB2BD")
    ax.spines["bottom"].set_color("#AAB2BD")

    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.023,
            f"{value * 100:.1f}%",
            ha="center",
            va="bottom",
            fontsize=8.5,
            fontweight="bold",
            color="#20252B",
        )

    gap = values[-1] - values[0]
    bracket_y = 0.885
    ax.plot([0, 0, 2, 2], [bracket_y - 0.012, bracket_y, bracket_y, bracket_y - 0.012],
            color="#24313F", linewidth=0.9, clip_on=False)
    ax.text(
        1,
        bracket_y + 0.003,
        f"+{gap * 100:.1f} pp",
        ha="center",
        va="bottom",
        fontsize=7.5,
        fontweight="bold",
        color="#24313F",
    )

    fig.suptitle(
        "Finer-grained action selection exposes step-dependent pruning risk",
        x=0.095,
        y=0.985,
        ha="left",
        fontsize=11,
        fontweight="bold",
        color="#17212B",
    )
    ax.set_title(
        "Diagnostic oracle computed from the same counterfactual action table",
        loc="left",
        pad=9,
        fontsize=7.5,
        color="#5C6670",
        fontweight="normal",
    )

    fig.subplots_adjust(left=0.095, right=0.985, bottom=0.24, top=0.78)

    stem = output_dir / "fig2_oracle_gap_diagnostic"
    fig.savefig(stem.with_suffix(".png"), dpi=600, facecolor="white")
    fig.savefig(stem.with_suffix(".tiff"), dpi=600, facecolor="white")
    fig.savefig(stem.with_suffix(".pdf"), facecolor="white")
    fig.savefig(stem.with_suffix(".svg"), facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    draw(read_oracle_values(args.input), args.output_dir)


if __name__ == "__main__":
    main()
