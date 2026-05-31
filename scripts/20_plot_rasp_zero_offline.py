from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


POLICY_LABELS = {
    "static_mlp_block": "Static MLP block",
    "entropy_only": "Entropy-only",
    "confidence_only": "Confidence-only",
    "hidden_probe": "Hidden-state probe",
    "rasp_zero_stage": "RASP-Zero",
}

COLORS = {
    "Static MLP block": "#6B7280",
    "Entropy-only": "#D97706",
    "Confidence-only": "#2563EB",
    "Hidden-state probe": "#059669",
    "RASP-Zero": "#B91C1C",
}


def policy_family(policy: str) -> str | None:
    for prefix, label in POLICY_LABELS.items():
        if policy.startswith(prefix):
            return label
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = pd.read_csv(args.input)
    rows["family"] = rows["policy"].map(policy_family)
    plot_rows = rows[rows["family"].notna()].copy()
    plot_rows["pruning_pct"] = 100 * plot_rows["average_pruning_ratio"]
    plot_rows["flip_pct"] = 100 * plot_rows["selected_action_flip_rate"]

    sns.set_theme(style="whitegrid", context="paper", font_scale=1.25)
    fig, ax = plt.subplots(figsize=(7.4, 4.8), constrained_layout=True)
    for family in POLICY_LABELS.values():
        family_rows = plot_rows[plot_rows["family"] == family].sort_values("pruning_pct")
        if family_rows.empty:
            continue
        ax.plot(
            family_rows["pruning_pct"],
            family_rows["flip_pct"],
            marker="o",
            linewidth=2.2,
            markersize=6,
            color=COLORS[family],
            label=family,
        )
    ax.set_xlabel("Realized average pruning ratio (%)")
    ax.set_ylabel("Answer flip rate (%)")
    ax.set_title("Offline policy safety under matched pruning budgets", pad=12, weight="bold")
    ax.legend(frameon=True, ncol=2)
    ax.grid(axis="both", color="#E5E7EB", linewidth=0.8)
    sns.despine(ax=ax)
    for suffix in ("png", "pdf"):
        fig.savefig(output_dir / f"rasp_zero_policy_frontier.{suffix}", dpi=320, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
