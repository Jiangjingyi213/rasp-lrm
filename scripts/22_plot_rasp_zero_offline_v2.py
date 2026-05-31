from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


POLICY_LABELS = {
    "static_best_strength": "Static best",
    "hidden_step_mlp_block": "Hidden-step MLP",
    "hard_stage_cap_ablation": "Hard stage cap",
    "action_conditioned": "Action-conditioned",
    "rasp_zero_v2_soft_stage": "RASP-Zero v2",
    "safe_step_oracle": "Safe step oracle",
}

COLORS = {
    "Static best": "#6B7280",
    "Hidden-step MLP": "#059669",
    "Hard stage cap": "#D97706",
    "Action-conditioned": "#2563EB",
    "RASP-Zero v2": "#B91C1C",
    "Safe step oracle": "#7C3AED",
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
    plot_rows["strength_pct"] = 100 * plot_rows["average_pruning_strength_proxy"]
    plot_rows["flip_pct"] = 100 * plot_rows["selected_action_flip_rate"]

    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
    fig, ax = plt.subplots(figsize=(7.8, 5.0), constrained_layout=True)
    for family in POLICY_LABELS.values():
        family_rows = plot_rows[plot_rows["family"] == family].sort_values("strength_pct")
        if family_rows.empty:
            continue
        ax.plot(
            family_rows["strength_pct"],
            family_rows["flip_pct"],
            marker="o",
            linewidth=2.1,
            markersize=6,
            color=COLORS[family],
            label=family,
        )
    ax.set_xlabel("Realized average pruning-strength proxy (%)")
    ax.set_ylabel("Answer flip rate (%)")
    ax.set_title("RASP-Zero v2: multi-module offline policy frontier", pad=12, weight="bold")
    ax.legend(frameon=True, ncol=2)
    ax.grid(axis="both", color="#E5E7EB", linewidth=0.8)
    sns.despine(ax=ax)
    for suffix in ("png", "pdf"):
        fig.savefig(output_dir / f"rasp_zero_v2_policy_frontier.{suffix}", dpi=320, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
