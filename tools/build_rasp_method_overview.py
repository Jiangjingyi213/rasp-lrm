#!/usr/bin/env python3
"""Build the RASP-LRM method overview as publication-ready vector artwork.

The figure is intentionally implementation-faithful:
  * one shared dense backbone, not one model per instance;
  * offline calibration stores stage-conditioned priors and means, not final masks;
  * online masks combine the active-stage prior with causal recent activations;
  * pruning is applied to gated-MLP intermediate channels;
  * protected core, warmup, refresh/cache, and dense fallback are visible;
  * the current backend is logical masking and does not imply kernel speedup.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import (
    FancyArrowPatch,
    FancyBboxPatch,
    Polygon,
    Rectangle,
)


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "docs" / "figures" / "rasp_method_overview"
OUT_DIR.mkdir(parents=True, exist_ok=True)


mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "font.size": 7.0,
        "axes.linewidth": 0.8,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "text.usetex": False,
    }
)


COL = {
    "ink": "#22303C",
    "muted": "#65717D",
    "line": "#AEB8C2",
    "panel": "#F8FAFC",
    "white": "#FFFFFF",
    "blue": "#2F6B9A",
    "blue_light": "#DDECF6",
    "blue_mid": "#8CB7D5",
    "teal": "#3E8E8A",
    "teal_light": "#DDF0ED",
    "gold": "#C88A24",
    "gold_light": "#F7E8C6",
    "violet": "#7563A8",
    "violet_light": "#EAE5F5",
    "rose": "#B95F67",
    "rose_light": "#F4DEE1",
    "gray": "#DDE2E7",
    "gray_dark": "#8D99A5",
    "danger": "#B84B4B",
}

STAGE_COLORS = {
    "SETUP": "#5E88B2",
    "REASONING": "#4B9A91",
    "VERIFY": "#9277B5",
    "FINAL": "#C48652",
}


def rounded_box(
    ax,
    x,
    y,
    w,
    h,
    *,
    face=COL["white"],
    edge=COL["line"],
    lw=0.9,
    radius=0.018,
    z=1,
):
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle=f"round,pad=0.006,rounding_size={radius}",
        linewidth=lw,
        edgecolor=edge,
        facecolor=face,
        zorder=z,
    )
    ax.add_patch(patch)
    return patch


def add_text(
    ax,
    x,
    y,
    text,
    *,
    size=7,
    weight="normal",
    color=COL["ink"],
    ha="center",
    va="center",
    z=5,
    linespacing=1.15,
    style="normal",
):
    return ax.text(
        x,
        y,
        text,
        fontsize=size,
        fontweight=weight,
        color=color,
        ha=ha,
        va=va,
        zorder=z,
        linespacing=linespacing,
        fontstyle=style,
    )


def arrow(
    ax,
    start,
    end,
    *,
    color=COL["muted"],
    lw=1.1,
    mutation=8,
    style="-|>",
    connectionstyle="arc3",
    z=3,
    dashed=False,
):
    patch = FancyArrowPatch(
        start,
        end,
        arrowstyle=style,
        mutation_scale=mutation,
        linewidth=lw,
        color=color,
        connectionstyle=connectionstyle,
        linestyle="--" if dashed else "-",
        shrinkA=0,
        shrinkB=0,
        zorder=z,
    )
    ax.add_patch(patch)
    return patch


def panel_label(ax, x, y, label, title):
    add_text(ax, x, y, label, size=9.5, weight="bold", ha="left")
    add_text(ax, x + 0.026, y, title, size=8.4, weight="bold", ha="left")


def draw_document_stack(ax, x, y, w, h):
    for dx, dy in ((0.018, 0.014), (0.009, 0.007), (0, 0)):
        rounded_box(
            ax,
            x + dx,
            y + dy,
            w,
            h,
            face=COL["white"],
            edge=COL["blue_mid"],
            radius=0.008,
            lw=0.8,
            z=2,
        )
    for i in range(3):
        ax.plot(
            [x + 0.012, x + w - 0.012],
            [y + h - 0.018 - i * 0.014] * 2,
            color=COL["blue_mid"],
            lw=0.7,
            zorder=4,
        )


def draw_filter(ax, cx, cy, scale=1.0):
    pts = [
        (cx - 0.025 * scale, cy + 0.022 * scale),
        (cx + 0.025 * scale, cy + 0.022 * scale),
        (cx + 0.008 * scale, cy - 0.002 * scale),
        (cx + 0.008 * scale, cy - 0.026 * scale),
        (cx - 0.007 * scale, cy - 0.018 * scale),
        (cx - 0.007 * scale, cy - 0.002 * scale),
    ]
    ax.add_patch(
        Polygon(pts, closed=True, facecolor=COL["teal_light"], edgecolor=COL["teal"], lw=0.9, zorder=4)
    )


def draw_stage_strip(ax, x, y, w, h, *, with_markers=True, compact=False):
    labels = ["SETUP", "REASONING", "VERIFY", "FINAL"]
    widths = [0.19, 0.38, 0.26, 0.17]
    cursor = x
    for label, frac in zip(labels, widths):
        ww = w * frac
        ax.add_patch(
            Rectangle(
                (cursor, y),
                ww,
                h,
                facecolor=STAGE_COLORS[label],
                edgecolor=COL["white"],
                linewidth=0.7,
                zorder=3,
            )
        )
        display_label = label[0] if compact else label.title()
        add_text(
            ax,
            cursor + ww / 2,
            y + h / 2,
            display_label,
            size=5.4 if compact else 5.6,
            weight="bold",
            color=COL["white"],
        )
        if with_markers and cursor > x:
            ax.plot([cursor, cursor], [y - 0.006, y + h + 0.006], color=COL["ink"], lw=0.7, zorder=4)
        cursor += ww


def draw_database(ax, x, y, w, h):
    rounded_box(
        ax,
        x,
        y,
        w,
        h,
        face=COL["violet_light"],
        edge=COL["violet"],
        radius=0.015,
        lw=1.0,
    )
    add_text(ax, x + w / 2, y + h - 0.025, "Stage calibration bank", size=7.1, weight="bold")
    stage_names = ["S", "R", "V", "F"]
    ys = [y + h - 0.052 - i * 0.026 for i in range(4)]
    patterns = [
        [0.18, 0.48, 0.78, 0.30, 0.62, 0.88, 0.39, 0.70],
        [0.62, 0.25, 0.46, 0.86, 0.73, 0.31, 0.91, 0.54],
        [0.84, 0.65, 0.27, 0.92, 0.41, 0.76, 0.35, 0.58],
        [0.30, 0.74, 0.55, 0.37, 0.88, 0.48, 0.67, 0.81],
    ]
    violet_rgb = mpl.colors.to_rgb(COL["violet"])
    white_rgb = mpl.colors.to_rgb(COL["white"])
    for stage, yy, vals, stage_color in zip(stage_names, ys, patterns, STAGE_COLORS.values()):
        add_text(ax, x + 0.017, yy, stage, size=5.5, weight="bold", color=stage_color)
        bx = x + 0.030
        for j, val in enumerate(vals):
            alpha = 0.15 + 0.78 * val
            c = mpl.colors.to_hex(
                tuple(alpha * fg + (1.0 - alpha) * bg for fg, bg in zip(violet_rgb, white_rgb))
            )
            ax.add_patch(
                Rectangle(
                    (bx + j * 0.0102, yy - 0.006),
                    0.0082,
                    0.012,
                    facecolor=c,
                    edgecolor="none",
                    zorder=3,
                )
            )
    add_text(
        ax,
        x + w / 2,
        y + 0.018,
        r"$I_{s,l}$  importance prior     $\mu_{s,l}$  stage mean",
        size=5.5,
        color=COL["muted"],
    )


def draw_channel_vector(ax, x, y, states, *, cell_w=0.014, cell_h=0.019, edge=True):
    color_map = {
        "protected": COL["gold"],
        "kept": COL["blue"],
        "pruned": COL["gray"],
        "recent": COL["teal"],
    }
    for idx, state in enumerate(states):
        ax.add_patch(
            Rectangle(
                (x + idx * cell_w, y),
                cell_w * 0.78,
                cell_h,
                facecolor=color_map[state],
                edgecolor=COL["white"] if edge else "none",
                linewidth=0.45,
                zorder=5,
            )
        )


def draw_shield(ax, cx, cy, scale=1.0):
    pts = [
        (cx, cy + 0.027 * scale),
        (cx + 0.022 * scale, cy + 0.017 * scale),
        (cx + 0.017 * scale, cy - 0.012 * scale),
        (cx, cy - 0.028 * scale),
        (cx - 0.017 * scale, cy - 0.012 * scale),
        (cx - 0.022 * scale, cy + 0.017 * scale),
    ]
    ax.add_patch(
        Polygon(
            pts,
            closed=True,
            facecolor=COL["gold_light"],
            edgecolor=COL["gold"],
            linewidth=1.0,
            zorder=5,
        )
    )
    ax.plot(
        [cx - 0.010 * scale, cx - 0.002 * scale, cx + 0.012 * scale],
        [cy, cy - 0.009 * scale, cy + 0.010 * scale],
        color=COL["gold"],
        lw=1.2,
        zorder=6,
    )


def draw_clock(ax, cx, cy, scale=1.0):
    circ = plt.Circle((cx, cy), 0.020 * scale, fc=COL["blue_light"], ec=COL["blue"], lw=0.9, zorder=5)
    ax.add_patch(circ)
    ax.plot([cx, cx], [cy, cy + 0.011 * scale], color=COL["blue"], lw=0.9, zorder=6)
    ax.plot([cx, cx + 0.009 * scale], [cy, cy - 0.006 * scale], color=COL["blue"], lw=0.9, zorder=6)


def build_figure():
    fig = plt.figure(figsize=(7.25, 5.35), facecolor="white")
    ax = fig.add_axes([0.015, 0.02, 0.97, 0.96])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # Figure title and thesis.
    add_text(
        ax,
        0.5,
        0.976,
        "RASP-LRM: Reasoning-stage adaptive structured pruning of MLP channels",
        size=10.2,
        weight="bold",
    )
    add_text(
        ax,
        0.5,
        0.948,
        "Calibration defines a stage-conditioned protection prior; the current trajectory decides the actual mask online.",
        size=6.5,
        color=COL["muted"],
    )

    # Main panel backgrounds.
    rounded_box(ax, 0.012, 0.655, 0.976, 0.265, face=COL["panel"], edge="#D6DEE6", radius=0.012)
    rounded_box(ax, 0.012, 0.055, 0.976, 0.575, face=COL["panel"], edge="#D6DEE6", radius=0.012)
    panel_label(ax, 0.027, 0.898, "a", "Offline stage-conditioned calibration")
    panel_label(ax, 0.027, 0.607, "b", "Causal stage-adaptive decoding with a shared dense backbone")

    # ------------------------------------------------------------------
    # Panel a: offline calibration.
    # ------------------------------------------------------------------
    draw_document_stack(ax, 0.045, 0.736, 0.075, 0.090)
    add_text(ax, 0.091, 0.710, "Independent\nreasoning pool", size=6.2)
    arrow(ax, (0.135, 0.780), (0.172, 0.780), color=COL["blue"], lw=1.2)

    rounded_box(ax, 0.173, 0.721, 0.155, 0.121, face=COL["blue_light"], edge=COL["blue"], radius=0.012)
    add_text(ax, 0.2505, 0.818, "Shared dense LRM", size=7.0, weight="bold")
    draw_stage_strip(ax, 0.190, 0.763, 0.121, 0.028, compact=True)
    add_text(ax, 0.2505, 0.742, "self-generated explicit-stage trajectory", size=5.4, color=COL["muted"])
    arrow(ax, (0.334, 0.780), (0.369, 0.780), color=COL["teal"], lw=1.2)

    rounded_box(ax, 0.370, 0.721, 0.116, 0.121, face=COL["teal_light"], edge=COL["teal"], radius=0.012)
    draw_filter(ax, 0.428, 0.798, 0.75)
    add_text(ax, 0.428, 0.754, "Keep correct,\nvalid, non-truncated", size=5.8)
    arrow(ax, (0.492, 0.780), (0.527, 0.780), color=COL["violet"], lw=1.2)

    rounded_box(ax, 0.528, 0.700, 0.185, 0.159, face=COL["white"], edge=COL["violet"], radius=0.012)
    add_text(ax, 0.6205, 0.835, "Per stage × layer × channel", size=6.4, weight="bold")
    add_text(
        ax,
        0.6205,
        0.802,
        r"$h_{l,t}=\phi(W_l^g x_{l,t})\odot(W_l^u x_{l,t})$",
        size=6.5,
    )
    add_text(
        ax,
        0.6205,
        0.764,
        r"$I_{s,l,j}=\mathrm{Var}(h_{l,:,j})\Vert W^d_{l,:,j}\Vert_2^2$",
        size=6.3,
        color=COL["violet"],
    )
    add_text(
        ax,
        0.6205,
        0.728,
        r"store $I_{s,l}$ and $\mu_{s,l}$ — not a frozen final mask",
        size=5.5,
        color=COL["muted"],
    )
    arrow(ax, (0.721, 0.780), (0.752, 0.780), color=COL["violet"], lw=1.2)
    draw_database(ax, 0.754, 0.693, 0.198, 0.174)

    # ------------------------------------------------------------------
    # Panel b: online stage trajectory and controller.
    # ------------------------------------------------------------------
    # Stage trajectory / causal parser.
    rounded_box(ax, 0.038, 0.493, 0.563, 0.083, face=COL["white"], edge=COL["line"], radius=0.012)
    add_text(ax, 0.058, 0.551, "Dense prefill", size=6.3, weight="bold", ha="left")
    add_text(ax, 0.058, 0.523, "prompt + forced setup marker", size=5.3, color=COL["muted"], ha="left")
    arrow(ax, (0.178, 0.535), (0.208, 0.535), color=COL["ink"], lw=1.0)
    draw_stage_strip(ax, 0.211, 0.516, 0.352, 0.038)
    add_text(ax, 0.387, 0.500, "generated reasoning trajectory (past tokens only)", size=5.2, color=COL["muted"])

    rounded_box(ax, 0.620, 0.493, 0.150, 0.083, face=COL["blue_light"], edge=COL["blue"], radius=0.012)
    add_text(ax, 0.695, 0.551, "Causal stage parser", size=6.6, weight="bold")
    add_text(ax, 0.695, 0.523, "complete legal marker → $s_t$", size=5.5)
    arrow(ax, (0.564, 0.535), (0.614, 0.535), color=COL["blue"], lw=1.1)

    rounded_box(ax, 0.791, 0.493, 0.161, 0.083, face=COL["rose_light"], edge=COL["rose"], radius=0.012)
    add_text(ax, 0.8715, 0.551, "Protocol guard", size=6.6, weight="bold")
    add_text(ax, 0.8715, 0.523, "protocol failure → dense fallback", size=5.2)
    arrow(ax, (0.770, 0.535), (0.785, 0.535), color=COL["rose"], lw=1.0)

    # Stage-conditioned selector.
    rounded_box(ax, 0.038, 0.215, 0.494, 0.250, face=COL["white"], edge=COL["blue"], lw=1.0, radius=0.014)
    add_text(ax, 0.055, 0.440, "Stage-conditioned online mask controller", size=7.2, weight="bold", ha="left")

    # Inputs: prior/budget and recent signal.
    rounded_box(ax, 0.058, 0.337, 0.148, 0.075, face=COL["violet_light"], edge=COL["violet"], radius=0.010)
    add_text(ax, 0.132, 0.391, "Stable stage evidence", size=6.0, weight="bold")
    add_text(ax, 0.132, 0.364, r"$I_{s_t,l},\ \mu_{s_t,l}$", size=7.2, color=COL["violet"])
    add_text(ax, 0.132, 0.345, r"$r_s,\ \kappa_s,\ w_s,\ W_s,\ \Delta_s$", size=5.5, color=COL["muted"])

    rounded_box(ax, 0.058, 0.270, 0.148, 0.053, face=COL["teal_light"], edge=COL["teal"], radius=0.010)
    add_text(ax, 0.132, 0.306, "Current-instance evidence", size=5.9, weight="bold")
    add_text(ax, 0.132, 0.286, r"$R_{s_t,l}(t)$ from recent stage-local activations", size=5.1)

    arrow(ax, (0.209, 0.374), (0.247, 0.374), color=COL["violet"], lw=1.0)
    arrow(ax, (0.209, 0.297), (0.247, 0.331), color=COL["teal"], lw=1.0)

    # Protection and fusion.
    rounded_box(ax, 0.250, 0.336, 0.122, 0.079, face=COL["gold_light"], edge=COL["gold"], radius=0.010)
    draw_shield(ax, 0.275, 0.376, 0.72)
    add_text(ax, 0.325, 0.390, "Protected core", size=6.0, weight="bold")
    add_text(ax, 0.325, 0.365, r"top-$\kappa_s$ of $I_{s,l}$", size=5.5)
    add_text(ax, 0.325, 0.348, "never enters prune set", size=4.9, color=COL["muted"])

    rounded_box(ax, 0.250, 0.266, 0.246, 0.056, face=COL["blue_light"], edge=COL["blue"], radius=0.010)
    add_text(
        ax,
        0.373,
        0.302,
        "Prunability fusion outside the protected core",
        size=5.8,
        weight="bold",
    )
    add_text(
        ax,
        0.373,
        0.281,
        r"$P=\lambda_p z(-I)+\lambda_r z(-R)$  →  top-$\tilde{k}_{s,l}$ to prune",
        size=5.8,
        color=COL["blue"],
    )

    arrow(ax, (0.375, 0.336), (0.375, 0.324), color=COL["gold"], lw=0.9)
    arrow(ax, (0.498, 0.294), (0.515, 0.294), color=COL["blue"], lw=1.1)

    # Safety strip within controller.
    rounded_box(ax, 0.058, 0.226, 0.183, 0.029, face=COL["panel"], edge=COL["line"], radius=0.007)
    draw_clock(ax, 0.075, 0.2405, 0.52)
    add_text(ax, 0.091, 0.2405, "dense stage warmup", size=5.0, ha="left", color=COL["muted"])
    rounded_box(ax, 0.253, 0.226, 0.243, 0.029, face=COL["panel"], edge=COL["line"], radius=0.007)
    add_text(
        ax,
        0.3745,
        0.2405,
        r"piecewise-constant mask: cache and refresh every $\Delta_s$",
        size=4.9,
        color=COL["muted"],
    )

    # The stage parser feeds the controller; the offline bank is referenced by "from (a)".
    arrow(
        ax,
        (0.650, 0.493),
        (0.495, 0.456),
        color=COL["blue"],
        lw=1.0,
        connectionstyle="arc3,rad=0.05",
    )
    add_text(ax, 0.575, 0.483, "active stage $s_t$", size=4.7, color=COL["blue"])

    # Runtime mask output and stage-dependent mask examples.
    rounded_box(ax, 0.546, 0.252, 0.162, 0.213, face=COL["white"], edge=COL["blue"], lw=1.0, radius=0.014)
    add_text(ax, 0.627, 0.440, "Online mask $m_{s_t,l}(t)$", size=6.8, weight="bold")
    add_text(ax, 0.627, 0.419, "same layer; varies by stage and instance", size=4.8, color=COL["muted"])
    patterns = {
        "S": ["protected", "kept", "pruned", "kept", "protected", "pruned", "kept", "pruned"],
        "R": ["kept", "protected", "kept", "pruned", "protected", "kept", "pruned", "pruned"],
        "V": ["pruned", "protected", "kept", "protected", "kept", "pruned", "kept", "pruned"],
        "F": ["kept", "pruned", "protected", "kept", "protected", "pruned", "pruned", "kept"],
    }
    yy = 0.382
    for (stage, states), stage_color in zip(patterns.items(), STAGE_COLORS.values()):
        add_text(ax, 0.568, yy + 0.009, stage, size=5.5, weight="bold", color=stage_color)
        draw_channel_vector(ax, 0.583, yy, states, cell_w=0.014, cell_h=0.018)
        yy -= 0.032
    add_text(ax, 0.627, 0.264, "illustrative binary masks", size=4.8, color=COL["muted"], style="italic")
    arrow(ax, (0.709, 0.350), (0.734, 0.350), color=COL["blue"], lw=1.2)

    # ------------------------------------------------------------------
    # Panel c embedded at right: exact MLP execution.
    # ------------------------------------------------------------------
    rounded_box(ax, 0.735, 0.154, 0.217, 0.311, face=COL["white"], edge=COL["ink"], lw=1.0, radius=0.014)
    add_text(ax, 0.751, 0.440, "Per-layer gated MLP only", size=6.8, weight="bold", ha="left")
    add_text(ax, 0.939, 0.440, "× $L$", size=6.0, weight="bold", ha="right", color=COL["muted"])

    # x branches.
    rounded_box(ax, 0.754, 0.382, 0.043, 0.030, face=COL["panel"], edge=COL["line"], radius=0.007)
    add_text(ax, 0.7755, 0.397, "$x_{l,t}$", size=6.5)
    arrow(ax, (0.798, 0.397), (0.822, 0.410), color=COL["ink"], lw=0.9)
    arrow(ax, (0.798, 0.397), (0.822, 0.362), color=COL["ink"], lw=0.9)
    rounded_box(ax, 0.824, 0.394, 0.074, 0.035, face=COL["blue_light"], edge=COL["blue"], radius=0.007)
    add_text(ax, 0.861, 0.412, "$W_l^g$ + $\\phi$", size=5.8)
    rounded_box(ax, 0.824, 0.344, 0.074, 0.035, face=COL["teal_light"], edge=COL["teal"], radius=0.007)
    add_text(ax, 0.861, 0.362, "$W_l^u$", size=6.0)
    arrow(ax, (0.899, 0.412), (0.919, 0.388), color=COL["ink"], lw=0.9)
    arrow(ax, (0.899, 0.362), (0.919, 0.382), color=COL["ink"], lw=0.9)
    add_text(ax, 0.925, 0.385, "$\\odot$", size=10, weight="bold")

    # Intermediate activation and mask.
    add_text(ax, 0.755, 0.321, "intermediate channel vector $h_{l,t}$", size=5.5, ha="left")
    draw_channel_vector(
        ax,
        0.759,
        0.291,
        ["protected", "kept", "pruned", "kept", "protected", "pruned", "kept", "pruned"],
        cell_w=0.022,
        cell_h=0.023,
    )
    add_text(ax, 0.842, 0.278, "$\\odot\\ m_{s_t,l}(t)$", size=5.6)

    # Compensation and down projection.
    rounded_box(ax, 0.758, 0.224, 0.177, 0.041, face=COL["gold_light"], edge=COL["gold"], radius=0.007)
    add_text(
        ax,
        0.8465,
        0.245,
        r"$\bar h=m\odot h+(1-m)\odot\mu_{s_t,l}$",
        size=5.8,
    )
    arrow(ax, (0.846, 0.224), (0.846, 0.205), color=COL["ink"], lw=0.9)
    rounded_box(ax, 0.802, 0.171, 0.089, 0.034, face=COL["violet_light"], edge=COL["violet"], radius=0.007)
    add_text(ax, 0.8465, 0.188, "down projection $W_l^d$", size=5.5)
    arrow(ax, (0.892, 0.188), (0.928, 0.188), color=COL["ink"], lw=0.9)
    add_text(ax, 0.938, 0.188, "$o_{l,t}$", size=6.4)

    # Autoregressive loop, routed around the content rather than through it.
    arrow(
        ax,
        (0.940, 0.171),
        (0.940, 0.082),
        color=COL["ink"],
        lw=0.9,
        connectionstyle="arc3,rad=0.0",
    )
    arrow(
        ax,
        (0.940, 0.082),
        (0.026, 0.082),
        color=COL["ink"],
        lw=0.9,
        connectionstyle="arc3,rad=0.0",
    )
    arrow(
        ax,
        (0.026, 0.082),
        (0.026, 0.535),
        color=COL["ink"],
        lw=0.9,
        connectionstyle="arc3,rad=0.0",
    )
    arrow(ax, (0.026, 0.535), (0.038, 0.535), color=COL["ink"], lw=0.9)
    add_text(ax, 0.544, 0.067, "next token → update stage state and recent window → reuse or refresh mask", size=5.4)

    # Legend and implementation boundary.
    legend_y = 0.130
    draw_channel_vector(ax, 0.054, legend_y, ["protected"], cell_w=0.017, cell_h=0.015)
    add_text(ax, 0.073, legend_y + 0.0075, "protected", size=5.2, ha="left")
    draw_channel_vector(ax, 0.135, legend_y, ["kept"], cell_w=0.017, cell_h=0.015)
    add_text(ax, 0.154, legend_y + 0.0075, "kept", size=5.2, ha="left")
    draw_channel_vector(ax, 0.204, legend_y, ["pruned"], cell_w=0.017, cell_h=0.015)
    add_text(ax, 0.223, legend_y + 0.0075, "masked + mean compensated", size=5.2, ha="left")

    add_text(
        ax,
        0.5,
        0.035,
        "One dense model and one calibration bank are shared by all requests; each request maintains only its own stage state, recent window, and temporary mask cache.",
        size=5.7,
        color=COL["muted"],
    )

    return fig


def build_figure_v2():
    """Clear three-panel layout: calibration, mask decision, mask execution."""

    fig = plt.figure(figsize=(7.25, 5.60), facecolor="white")
    ax = fig.add_axes([0.015, 0.018, 0.97, 0.965])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    add_text(
        ax,
        0.5,
        0.982,
        "RASP-LRM: reasoning-stage adaptive structured pruning of MLP channels",
        size=10.0,
        weight="bold",
    )

    # A compact system-level spine makes the relations among panels explicit.
    rounded_box(ax, 0.150, 0.936, 0.176, 0.026, face=COL["violet_light"], edge=COL["violet"], radius=0.006)
    add_text(ax, 0.238, 0.949, "(a) stage prior bank", size=5.2, weight="bold", color=COL["violet"])
    arrow(ax, (0.330, 0.949), (0.363, 0.949), color=COL["violet"], lw=0.9)
    rounded_box(ax, 0.367, 0.936, 0.190, 0.026, face=COL["blue_light"], edge=COL["blue"], radius=0.006)
    add_text(ax, 0.462, 0.949, "(b) online mask decision", size=5.2, weight="bold", color=COL["blue"])
    arrow(ax, (0.561, 0.949), (0.594, 0.949), color=COL["blue"], lw=0.9)
    rounded_box(ax, 0.598, 0.936, 0.176, 0.026, face=COL["gold_light"], edge=COL["gold"], radius=0.006)
    add_text(ax, 0.686, 0.949, "(c) masked MLP execution", size=5.2, weight="bold", color=COL["ink"])
    arrow(ax, (0.778, 0.949), (0.810, 0.949), color=COL["teal"], lw=0.9)
    rounded_box(ax, 0.814, 0.936, 0.135, 0.026, face=COL["teal_light"], edge=COL["teal"], radius=0.006)
    add_text(ax, 0.8815, 0.949, "$h\\rightarrow R$ feedback to (b)", size=4.8, weight="bold", color=COL["teal"])

    # Three visually independent panels.
    rounded_box(ax, 0.012, 0.716, 0.976, 0.205, face=COL["panel"], edge="#D6DEE6", radius=0.012)
    rounded_box(ax, 0.012, 0.372, 0.976, 0.320, face=COL["panel"], edge="#D6DEE6", radius=0.012)
    rounded_box(ax, 0.012, 0.055, 0.976, 0.290, face=COL["panel"], edge="#D6DEE6", radius=0.012)
    panel_label(ax, 0.027, 0.900, "a", "Build a shared stage-conditioned calibration bank")
    panel_label(ax, 0.027, 0.671, "b", "Decide the mask online: stage prior + current-instance evidence")
    panel_label(ax, 0.027, 0.324, "c", "Execute the mask inside one Transformer layer")

    # ==================================================================
    # a. Offline calibration
    # ==================================================================
    draw_document_stack(ax, 0.046, 0.782, 0.055, 0.060)
    add_text(ax, 0.078, 0.752, "independent\nreasoning pool", size=5.4)
    arrow(ax, (0.108, 0.812), (0.139, 0.812), color=COL["blue"], lw=1.0)

    rounded_box(ax, 0.142, 0.763, 0.160, 0.100, face=COL["blue_light"], edge=COL["blue"], radius=0.010)
    add_text(ax, 0.222, 0.842, "Shared dense LRM", size=6.5, weight="bold")
    draw_stage_strip(ax, 0.162, 0.797, 0.120, 0.024, compact=True)
    add_text(ax, 0.222, 0.778, "explicit-stage trajectories", size=5.0, color=COL["muted"])
    arrow(ax, (0.307, 0.812), (0.334, 0.812), color=COL["teal"], lw=1.0)

    rounded_box(ax, 0.337, 0.763, 0.110, 0.100, face=COL["teal_light"], edge=COL["teal"], radius=0.010)
    draw_filter(ax, 0.392, 0.829, 0.55)
    add_text(ax, 0.392, 0.790, "correct + valid\n+ non-truncated", size=5.1)
    arrow(ax, (0.452, 0.812), (0.478, 0.812), color=COL["violet"], lw=1.0)

    rounded_box(ax, 0.481, 0.750, 0.230, 0.126, face=COL["white"], edge=COL["violet"], radius=0.010)
    add_text(ax, 0.596, 0.851, "Collect gated-MLP activations\nby stage × layer", size=5.35, weight="bold")
    add_text(
        ax,
        0.596,
        0.820,
        r"$h_{l,t}=\phi(W_l^g x_{l,t})\odot(W_l^u x_{l,t})$",
        size=5.8,
    )
    add_text(
        ax,
        0.596,
        0.789,
        r"$I_{s,l,j}=\mathrm{Var}(h_{l,:,j})\Vert W^d_{l,:,j}\Vert_2^2$",
        size=5.6,
        color=COL["violet"],
    )
    add_text(ax, 0.596, 0.765, r"retain stage mean $\mu_{s,l}$ for compensation", size=4.9, color=COL["muted"])
    arrow(ax, (0.716, 0.812), (0.740, 0.812), color=COL["violet"], lw=1.0)

    draw_database(ax, 0.744, 0.721, 0.216, 0.170)

    # ==================================================================
    # b. Online mask decision
    # ==================================================================
    # Top row: state and two evidence sources.
    rounded_box(ax, 0.041, 0.553, 0.287, 0.081, face=COL["white"], edge=COL["blue"], radius=0.010)
    add_text(ax, 0.055, 0.616, "1  Causally read the active reasoning stage", size=5.8, weight="bold", ha="left")
    draw_stage_strip(ax, 0.056, 0.575, 0.204, 0.024)
    ax.add_patch(
        Polygon(
            [(0.151, 0.568), (0.159, 0.568), (0.155, 0.556)],
            closed=True,
            facecolor=COL["ink"],
            edgecolor="none",
            zorder=6,
        )
    )
    rounded_box(ax, 0.269, 0.568, 0.045, 0.035, face=COL["blue_light"], edge=COL["blue"], radius=0.007)
    add_text(ax, 0.2915, 0.5855, "$s_t$", size=6.7, weight="bold", color=COL["blue"])
    add_text(ax, 0.184, 0.559, "completed marker only", size=4.6, color=COL["muted"])

    rounded_box(ax, 0.367, 0.548, 0.246, 0.090, face=COL["violet_light"], edge=COL["violet"], radius=0.010)
    add_text(ax, 0.490, 0.618, "2  Select stable evidence from (a) for stage $s_t$", size=5.55, weight="bold")
    add_text(ax, 0.490, 0.590, r"prior $I_{s_t,l}$     mean $\mu_{s_t,l}$", size=6.0, color=COL["violet"])
    add_text(
        ax,
        0.490,
        0.565,
        r"budget $r_s$   core $\kappa_s$   warmup $w_s$   window $W_s$   refresh $\Delta_s$",
        size=4.65,
        color=COL["muted"],
    )
    arrow(ax, (0.329, 0.593), (0.361, 0.593), color=COL["blue"], lw=1.0)

    rounded_box(ax, 0.652, 0.548, 0.306, 0.090, face=COL["teal_light"], edge=COL["teal"], radius=0.010)
    add_text(ax, 0.805, 0.618, "3  Update instance evidence using $h$ from (c)", size=5.55, weight="bold")
    add_text(ax, 0.681, 0.588, "recent stage-local tokens", size=4.9, color=COL["muted"], ha="left")
    token_x = 0.682
    token_heights = [0.013, 0.025, 0.018, 0.031, 0.016, 0.027]
    for idx, height in enumerate(token_heights):
        ax.add_patch(
            Rectangle(
                (token_x + idx * 0.018, 0.562),
                0.012,
                height,
                facecolor=COL["teal"],
                edgecolor="none",
                zorder=5,
            )
        )
    arrow(ax, (0.800, 0.575), (0.832, 0.575), color=COL["teal"], lw=0.9)
    add_text(ax, 0.889, 0.580, r"$R_{s_t,l}(t)$", size=7.0, weight="bold", color=COL["teal"])
    add_text(ax, 0.889, 0.558, "causal: current and past tokens", size=4.55, color=COL["muted"])

    # Bottom row: protection, fusion, and temporary mask.
    rounded_box(ax, 0.094, 0.431, 0.206, 0.083, face=COL["gold_light"], edge=COL["gold"], radius=0.010)
    draw_shield(ax, 0.125, 0.473, 0.65)
    add_text(ax, 0.211, 0.493, "4  Form the protected core", size=5.7, weight="bold")
    add_text(ax, 0.211, 0.467, r"$\Omega_{s,l}=\mathrm{TopK}(I_{s,l},\kappa_s C_l)$", size=5.4)
    add_text(ax, 0.211, 0.446, "excluded from all deletion candidates", size=4.7, color=COL["muted"])

    rounded_box(ax, 0.365, 0.421, 0.294, 0.103, face=COL["blue_light"], edge=COL["blue"], radius=0.010)
    add_text(ax, 0.512, 0.502, "5  Rank only the non-protected channels", size=5.8, weight="bold")
    add_text(
        ax,
        0.512,
        0.470,
        r"$P_{s,l,j}(t)=\lambda_p z(-I_{s,l,j})+\lambda_r z(-R_{s,l,j}(t))$",
        size=5.55,
        color=COL["blue"],
    )
    add_text(
        ax,
        0.512,
        0.442,
        r"top-$\tilde{k}_{s,l}$ prunability scores $\rightarrow$ deletion set $\mathcal{D}_{s,l}(t)$",
        size=5.0,
        color=COL["muted"],
    )

    rounded_box(ax, 0.721, 0.421, 0.225, 0.103, face=COL["white"], edge=COL["blue"], radius=0.010)
    add_text(ax, 0.8335, 0.502, "6  Cache a temporary mask for (c)", size=5.55, weight="bold")
    draw_channel_vector(
        ax,
        0.749,
        0.462,
        ["protected", "kept", "pruned", "kept", "protected", "pruned", "kept", "pruned"],
        cell_w=0.023,
        cell_h=0.022,
    )
    add_text(ax, 0.8335, 0.442, "stage- and instance-specific", size=4.8, color=COL["muted"])

    arrow(ax, (0.490, 0.548), (0.225, 0.518), color=COL["violet"], lw=0.9, connectionstyle="arc3,rad=-0.08")
    arrow(ax, (0.490, 0.548), (0.477, 0.526), color=COL["violet"], lw=0.9)
    arrow(ax, (0.805, 0.548), (0.558, 0.526), color=COL["teal"], lw=0.9, connectionstyle="arc3,rad=0.08")
    arrow(ax, (0.306, 0.472), (0.359, 0.472), color=COL["gold"], lw=1.0)
    arrow(ax, (0.665, 0.472), (0.715, 0.472), color=COL["blue"], lw=1.0)

    # Safety and update policy as a separate, quiet strip.
    rounded_box(ax, 0.099, 0.386, 0.218, 0.025, face=COL["white"], edge=COL["line"], radius=0.006)
    add_text(ax, 0.208, 0.3985, "warmup → observe densely before pruning", size=4.7, color=COL["muted"])
    rounded_box(ax, 0.389, 0.386, 0.235, 0.025, face=COL["white"], edge=COL["line"], radius=0.006)
    add_text(ax, 0.5065, 0.3985, r"cache → reuse until the next $\Delta_s$ refresh", size=4.7, color=COL["muted"])
    rounded_box(ax, 0.696, 0.386, 0.233, 0.025, face=COL["rose_light"], edge=COL["rose"], radius=0.006)
    add_text(ax, 0.8125, 0.3985, "protocol failure → dense fallback", size=4.7, color=COL["rose"])

    # ==================================================================
    # c. Per-layer execution
    # ==================================================================
    # Step 1: transformer context.
    rounded_box(ax, 0.044, 0.109, 0.185, 0.169, face=COL["white"], edge=COL["line"], radius=0.010)
    add_text(ax, 0.1365, 0.258, "1  Transformer layer", size=5.9, weight="bold")
    rounded_box(ax, 0.071, 0.211, 0.130, 0.030, face=COL["panel"], edge=COL["line"], radius=0.006)
    add_text(ax, 0.136, 0.226, "self-attention (unchanged)", size=5.0, color=COL["muted"])
    arrow(ax, (0.136, 0.209), (0.136, 0.193), color=COL["ink"], lw=0.8)
    rounded_box(ax, 0.071, 0.153, 0.130, 0.039, face=COL["blue_light"], edge=COL["blue"], radius=0.006)
    add_text(ax, 0.136, 0.1725, "gated MLP (adapted)", size=5.4, weight="bold", color=COL["blue"])
    arrow(ax, (0.136, 0.151), (0.136, 0.133), color=COL["ink"], lw=0.8)
    add_text(ax, 0.136, 0.121, "layer output (residuals omitted)", size=4.4, color=COL["muted"])
    arrow(ax, (0.235, 0.193), (0.267, 0.193), color=COL["blue"], lw=1.0)

    # Step 2: gated MLP produces the prunable intermediate channels.
    rounded_box(ax, 0.273, 0.109, 0.292, 0.169, face=COL["white"], edge=COL["blue"], radius=0.010)
    add_text(ax, 0.419, 0.258, "2  Compute the intermediate channel vector", size=5.9, weight="bold")
    rounded_box(ax, 0.292, 0.190, 0.047, 0.032, face=COL["panel"], edge=COL["line"], radius=0.006)
    add_text(ax, 0.3155, 0.206, "$x_{l,t}$", size=6.0)
    arrow(ax, (0.340, 0.206), (0.366, 0.222), color=COL["ink"], lw=0.8)
    arrow(ax, (0.340, 0.206), (0.366, 0.177), color=COL["ink"], lw=0.8)
    rounded_box(ax, 0.369, 0.209, 0.077, 0.030, face=COL["blue_light"], edge=COL["blue"], radius=0.006)
    add_text(ax, 0.4075, 0.224, "$W_l^g$ + $\\phi$", size=5.4)
    rounded_box(ax, 0.369, 0.164, 0.077, 0.030, face=COL["teal_light"], edge=COL["teal"], radius=0.006)
    add_text(ax, 0.4075, 0.179, "$W_l^u$", size=5.6)
    arrow(ax, (0.447, 0.224), (0.473, 0.204), color=COL["ink"], lw=0.8)
    arrow(ax, (0.447, 0.179), (0.473, 0.199), color=COL["ink"], lw=0.8)
    add_text(ax, 0.480, 0.201, "$\\odot$", size=9.0, weight="bold")
    arrow(ax, (0.491, 0.201), (0.512, 0.201), color=COL["ink"], lw=0.8)
    add_text(ax, 0.530, 0.201, "$h_{l,t}$", size=6.2, weight="bold")
    draw_channel_vector(
        ax,
        0.331,
        0.125,
        ["protected", "kept", "pruned", "kept", "protected", "pruned", "kept", "pruned"],
        cell_w=0.026,
        cell_h=0.022,
    )
    add_text(ax, 0.435, 0.113, "this vector also updates the recent score in (b)", size=4.55, color=COL["teal"])
    arrow(ax, (0.570, 0.193), (0.602, 0.193), color=COL["blue"], lw=1.0)

    # Step 3: exact mask placement and mean compensation.
    rounded_box(ax, 0.608, 0.090, 0.350, 0.188, face=COL["white"], edge=COL["ink"], radius=0.010)
    add_text(ax, 0.783, 0.258, "3  Apply the mask from (b) before the down projection", size=5.65, weight="bold")
    add_text(ax, 0.632, 0.226, "$h_{l,t}$", size=5.6, ha="left")
    draw_channel_vector(
        ax,
        0.674,
        0.214,
        ["protected", "kept", "pruned", "kept", "protected", "pruned", "kept", "pruned"],
        cell_w=0.026,
        cell_h=0.022,
    )
    add_text(ax, 0.632, 0.190, "$m_{s_t,l}(t)$", size=5.2, ha="left")
    mask_states = ["protected", "kept", "pruned", "kept", "protected", "pruned", "kept", "pruned"]
    draw_channel_vector(ax, 0.674, 0.178, mask_states, cell_w=0.026, cell_h=0.022)
    rounded_box(ax, 0.632, 0.126, 0.244, 0.036, face=COL["gold_light"], edge=COL["gold"], radius=0.006)
    add_text(
        ax,
        0.754,
        0.144,
        r"$\bar h=m\odot h+(1-m)\odot\mu_{s_t,l}$",
        size=5.6,
    )
    arrow(ax, (0.879, 0.144), (0.899, 0.144), color=COL["ink"], lw=0.8)
    rounded_box(ax, 0.901, 0.126, 0.041, 0.036, face=COL["violet_light"], edge=COL["violet"], radius=0.006)
    add_text(ax, 0.9215, 0.144, "$W_l^d$", size=5.3)
    add_text(ax, 0.783, 0.105, r"output $o_{l,t}=W_l^d\bar h_{l,t}$", size=5.2, color=COL["muted"])

    # Direct color legend and implementation boundary.
    legend_y = 0.072
    draw_channel_vector(ax, 0.075, legend_y, ["protected"], cell_w=0.017, cell_h=0.014)
    add_text(ax, 0.094, legend_y + 0.007, "protected", size=4.8, ha="left")
    draw_channel_vector(ax, 0.157, legend_y, ["kept"], cell_w=0.017, cell_h=0.014)
    add_text(ax, 0.176, legend_y + 0.007, "kept", size=4.8, ha="left")
    draw_channel_vector(ax, 0.222, legend_y, ["pruned"], cell_w=0.017, cell_h=0.014)
    add_text(ax, 0.241, legend_y + 0.007, "masked and mean-compensated", size=4.8, ha="left")

    return fig


def build_figure_v3():
    """Compact, visually connected overview with one closed-loop story."""

    fig = plt.figure(figsize=(7.25, 4.25), facecolor="white")
    ax = fig.add_axes([0.012, 0.018, 0.976, 0.965])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    add_text(
        ax,
        0.5,
        0.974,
        "RASP-LRM: stage knowledge guides online, instance-adaptive MLP pruning",
        size=10.0,
        weight="bold",
    )
    add_text(
        ax,
        0.5,
        0.940,
        "One shared model · a different temporary channel mask as reasoning evolves",
        size=6.3,
        color=COL["muted"],
    )

    # Online/offline regions make the execution boundary immediately visible.
    rounded_box(ax, 0.018, 0.205, 0.282, 0.680, face="#FAF8FD", edge="#D9D0E8", radius=0.016)
    rounded_box(ax, 0.326, 0.205, 0.656, 0.680, face="#F8FBFD", edge="#D5E2EA", radius=0.016)
    add_text(ax, 0.039, 0.864, "OFFLINE · built once", size=5.5, weight="bold", color=COL["violet"], ha="left")
    add_text(ax, 0.347, 0.864, "ONLINE · causal decoding for each request", size=5.5, weight="bold", color=COL["blue"], ha="left")

    # ------------------------------------------------------------------
    # a. Shared stage knowledge
    # ------------------------------------------------------------------
    panel_label(ax, 0.039, 0.824, "a", "Shared stage knowledge")

    rounded_box(ax, 0.051, 0.673, 0.216, 0.105, face=COL["blue_light"], edge=COL["blue"], radius=0.012)
    draw_document_stack(ax, 0.064, 0.696, 0.038, 0.046)
    add_text(ax, 0.124, 0.746, "Dense calibration trajectories", size=5.9, weight="bold", ha="left")
    draw_stage_strip(ax, 0.116, 0.704, 0.133, 0.024, compact=True)
    add_text(ax, 0.182, 0.688, "valid reasoning only", size=4.7, color=COL["muted"])

    arrow(ax, (0.159, 0.668), (0.159, 0.633), color=COL["violet"], lw=1.2)

    rounded_box(ax, 0.051, 0.515, 0.216, 0.112, face=COL["white"], edge=COL["violet"], radius=0.012)
    add_text(ax, 0.159, 0.602, "Observe MLP channels by stage", size=6.0, weight="bold")
    # Small stage-specific activation fingerprints replace detailed equations.
    fingerprint_y = [0.566, 0.545]
    fingerprint_states = [
        ["protected", "kept", "pruned", "kept", "protected", "pruned", "kept", "pruned"],
        ["kept", "pruned", "protected", "kept", "pruned", "protected", "pruned", "kept"],
    ]
    for yy, states in zip(fingerprint_y, fingerprint_states):
        draw_channel_vector(ax, 0.091, yy, states, cell_w=0.018, cell_h=0.014)
    add_text(ax, 0.159, 0.530, "the important channels change with stage", size=4.7, color=COL["muted"])

    arrow(ax, (0.159, 0.509), (0.159, 0.474), color=COL["violet"], lw=1.2)

    rounded_box(ax, 0.051, 0.316, 0.216, 0.152, face=COL["violet_light"], edge=COL["violet"], radius=0.012)
    add_text(ax, 0.159, 0.443, "Stage prior bank", size=6.5, weight="bold", color=COL["violet"])
    labels = ["Setup", "Reason", "Verify", "Final"]
    for i, (label, color) in enumerate(zip(labels, STAGE_COLORS.values())):
        yy = 0.407 - i * 0.024
        add_text(ax, 0.072, yy, label, size=4.8, weight="bold", color=color, ha="left")
        draw_channel_vector(
            ax,
            0.126,
            yy - 0.007,
            fingerprint_states[i % 2],
            cell_w=0.015,
            cell_h=0.013,
        )

    # ------------------------------------------------------------------
    # b. Hero: stage-aware online decision
    # ------------------------------------------------------------------
    rounded_box(ax, 0.347, 0.310, 0.338, 0.510, face=COL["white"], edge=COL["blue"], lw=1.25, radius=0.016)
    panel_label(ax, 0.367, 0.790, "b", "Stage-aware mask decision")

    add_text(ax, 0.378, 0.741, "Current reasoning trajectory", size=5.3, weight="bold", ha="left")
    draw_stage_strip(ax, 0.378, 0.697, 0.276, 0.032)
    # Marker points at the currently active stage.
    ax.add_patch(
        Polygon(
            [(0.522, 0.691), (0.532, 0.691), (0.527, 0.675)],
            closed=True,
            facecolor=COL["ink"],
            edgecolor="none",
            zorder=7,
        )
    )
    add_text(ax, 0.527, 0.657, "active stage", size=4.6, color=COL["muted"])

    # Two evidence cards visually converge on one selector.
    rounded_box(ax, 0.374, 0.546, 0.126, 0.079, face=COL["violet_light"], edge=COL["violet"], radius=0.010)
    add_text(ax, 0.437, 0.600, "Stage knowledge", size=5.6, weight="bold", color=COL["violet"])
    draw_shield(ax, 0.397, 0.567, 0.48)
    add_text(ax, 0.450, 0.568, "stable protection", size=4.7, color=COL["muted"])

    rounded_box(ax, 0.531, 0.546, 0.126, 0.079, face=COL["teal_light"], edge=COL["teal"], radius=0.010)
    add_text(ax, 0.594, 0.600, "Recent activity", size=5.6, weight="bold", color=COL["teal"])
    for i, hh in enumerate([0.012, 0.023, 0.016, 0.029, 0.019]):
        ax.add_patch(
            Rectangle(
                (0.550 + i * 0.014, 0.556),
                0.009,
                hh,
                facecolor=COL["teal"],
                edgecolor="none",
                zorder=5,
            )
        )
    add_text(ax, 0.626, 0.568, "instance signal", size=4.7, color=COL["muted"])

    arrow(ax, (0.437, 0.540), (0.486, 0.493), color=COL["violet"], lw=1.25)
    arrow(ax, (0.594, 0.540), (0.545, 0.493), color=COL["teal"], lw=1.25)

    rounded_box(ax, 0.448, 0.423, 0.136, 0.074, face=COL["blue_light"], edge=COL["blue"], lw=1.2, radius=0.012)
    add_text(ax, 0.516, 0.474, "Dynamic selector", size=6.3, weight="bold", color=COL["blue"])
    add_text(ax, 0.516, 0.445, "protect first · rank the rest", size=4.8, color=COL["muted"])

    arrow(ax, (0.516, 0.418), (0.516, 0.385), color=COL["blue"], lw=1.35)
    rounded_box(ax, 0.383, 0.329, 0.266, 0.052, face=COL["white"], edge=COL["blue"], radius=0.010)
    draw_channel_vector(
        ax,
        0.403,
        0.344,
        ["protected", "kept", "pruned", "kept", "protected", "pruned", "kept", "pruned"],
        cell_w=0.022,
        cell_h=0.021,
    )
    add_text(ax, 0.615, 0.354, "temporary mask", size=4.8, weight="bold", color=COL["blue"])

    # a -> b: the wide payload arrow crosses the offline/online boundary.
    arrow(ax, (0.270, 0.420), (0.365, 0.578), color=COL["violet"], lw=1.7, connectionstyle="arc3,rad=-0.08")
    add_text(ax, 0.299, 0.510, "select\nby stage", size=4.4, weight="bold", color=COL["violet"], z=8)

    # ------------------------------------------------------------------
    # c. The actual execution site inside the shared model
    # ------------------------------------------------------------------
    rounded_box(ax, 0.712, 0.310, 0.247, 0.510, face=COL["white"], edge=COL["ink"], lw=1.05, radius=0.016)
    panel_label(ax, 0.732, 0.790, "c", "Masked gated MLP")
    add_text(ax, 0.835, 0.750, "inside every Transformer layer", size=4.8, color=COL["muted"])

    rounded_box(ax, 0.735, 0.686, 0.052, 0.034, face=COL["panel"], edge=COL["line"], radius=0.006)
    add_text(ax, 0.761, 0.703, "input", size=5.0)
    arrow(ax, (0.788, 0.703), (0.815, 0.718), color=COL["ink"], lw=0.9)
    arrow(ax, (0.788, 0.703), (0.815, 0.669), color=COL["ink"], lw=0.9)
    rounded_box(ax, 0.817, 0.704, 0.095, 0.035, face=COL["blue_light"], edge=COL["blue"], radius=0.006)
    add_text(ax, 0.8645, 0.7215, "gate projection", size=5.0)
    rounded_box(ax, 0.817, 0.652, 0.095, 0.035, face=COL["teal_light"], edge=COL["teal"], radius=0.006)
    add_text(ax, 0.8645, 0.6695, "up projection", size=5.0)
    arrow(ax, (0.865, 0.648), (0.865, 0.617), color=COL["ink"], lw=0.9)
    arrow(ax, (0.865, 0.700), (0.865, 0.617), color=COL["ink"], lw=0.9)
    add_text(ax, 0.865, 0.606, "multiply", size=4.7, weight="bold")

    add_text(ax, 0.835, 0.566, "intermediate MLP channels", size=5.4, weight="bold")
    draw_channel_vector(
        ax,
        0.742,
        0.531,
        ["protected", "kept", "kept", "kept", "protected", "kept", "kept", "kept"],
        cell_w=0.027,
        cell_h=0.024,
    )
    arrow(ax, (0.835, 0.524), (0.835, 0.490), color=COL["blue"], lw=1.2)

    rounded_box(ax, 0.742, 0.423, 0.187, 0.065, face=COL["gold_light"], edge=COL["gold"], radius=0.009)
    add_text(ax, 0.8355, 0.469, "Apply the mask from (b)", size=5.7, weight="bold")
    draw_channel_vector(
        ax,
        0.758,
        0.435,
        ["protected", "kept", "pruned", "kept", "protected", "pruned", "kept", "pruned"],
        cell_w=0.023,
        cell_h=0.020,
    )
    arrow(ax, (0.835, 0.418), (0.835, 0.388), color=COL["ink"], lw=1.0)
    rounded_box(ax, 0.770, 0.344, 0.131, 0.041, face=COL["violet_light"], edge=COL["violet"], radius=0.007)
    add_text(ax, 0.8355, 0.3645, "down projection", size=5.4, weight="bold")
    arrow(ax, (0.902, 0.365), (0.935, 0.365), color=COL["ink"], lw=0.9)
    add_text(ax, 0.943, 0.365, "output", size=4.8, ha="right")

    # b -> c: the most important runtime interface.
    arrow(ax, (0.653, 0.355), (0.735, 0.455), color=COL["blue"], lw=1.8, connectionstyle="arc3,rad=-0.05")
    add_text(ax, 0.695, 0.410, "mask", size=5.0, weight="bold", color=COL["blue"])

    # c -> b: bold causal feedback loop, visually closing the system.
    arrow(
        ax,
        (0.744, 0.543),
        (0.657, 0.585),
        color=COL["teal"],
        lw=1.8,
        connectionstyle="arc3,rad=0.10",
        z=7,
    )
    add_text(ax, 0.700, 0.575, "activation\nfeedback", size=4.2, weight="bold", color=COL["teal"], z=9)

    # Compact operational safeguards; no equation-heavy implementation detail.
    rounded_box(ax, 0.347, 0.229, 0.612, 0.052, face=COL["white"], edge=COL["line"], radius=0.010)
    draw_clock(ax, 0.374, 0.255, 0.48)
    add_text(ax, 0.392, 0.255, "dense warmup", size=4.8, ha="left", color=COL["muted"])
    add_text(ax, 0.492, 0.255, "·", size=7.0, color=COL["line"])
    add_text(ax, 0.515, 0.255, "cache, then refresh", size=4.8, ha="left", color=COL["muted"])
    add_text(ax, 0.650, 0.255, "·", size=7.0, color=COL["line"])
    add_text(ax, 0.673, 0.255, "invalid stage → dense fallback", size=4.8, ha="left", color=COL["muted"])

    # Shared-model statement and direct color key.
    add_text(
        ax,
        0.5,
        0.151,
        "The backbone weights are shared; only the stage state, recent activation window, and temporary mask change during decoding.",
        size=5.5,
        color=COL["muted"],
    )
    legend_y = 0.097
    draw_channel_vector(ax, 0.335, legend_y, ["protected"], cell_w=0.018, cell_h=0.015)
    add_text(ax, 0.355, legend_y + 0.0075, "protected", size=4.8, ha="left")
    draw_channel_vector(ax, 0.445, legend_y, ["kept"], cell_w=0.018, cell_h=0.015)
    add_text(ax, 0.465, legend_y + 0.0075, "active", size=4.8, ha="left")
    draw_channel_vector(ax, 0.535, legend_y, ["pruned"], cell_w=0.018, cell_h=0.015)
    add_text(ax, 0.555, legend_y + 0.0075, "masked + compensated", size=4.8, ha="left")

    return fig


def build_figure_v4():
    """Innovation-focused overview: contrast fixed masks with RASP's dynamic loop."""

    fig = plt.figure(figsize=(7.25, 4.95), facecolor="white")
    ax = fig.add_axes([0.010, 0.018, 0.980, 0.965])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    add_text(
        ax,
        0.5,
        0.978,
        "RASP-LRM: stage-conditioned, instance-adaptive MLP channel pruning during reasoning",
        size=9.4,
        weight="bold",
    )
    add_text(
        ax,
        0.5,
        0.948,
        "Static pruning uses one fixed mask; RASP keeps one shared backbone but refreshes a temporary mask as the reasoning stage changes.",
        size=5.9,
        color=COL["muted"],
    )

    # ------------------------------------------------------------------
    # Top: make the innovation visible before the detailed workflow.
    # ------------------------------------------------------------------
    rounded_box(ax, 0.018, 0.735, 0.305, 0.180, face="#FCFCFD", edge=COL["line"], radius=0.014)
    add_text(ax, 0.036, 0.892, "Conventional assumption", size=6.6, weight="bold", ha="left")
    add_text(ax, 0.036, 0.868, "one pruning decision is reused across reasoning", size=4.9, color=COL["muted"], ha="left")
    draw_stage_strip(ax, 0.046, 0.820, 0.236, 0.028, compact=True)
    add_text(ax, 0.166, 0.800, "same mask for every stage", size=4.8, color=COL["muted"])
    for yy in [0.782, 0.763]:
        draw_channel_vector(
            ax,
            0.066,
            yy,
            ["kept", "pruned", "kept", "protected", "kept", "pruned", "kept", "pruned"],
            cell_w=0.022,
            cell_h=0.015,
        )
    ax.plot([0.046, 0.303], [0.812, 0.812], color=COL["line"], lw=0.8, linestyle="--")

    rounded_box(ax, 0.345, 0.735, 0.637, 0.180, face="#F8FBFD", edge=COL["blue"], lw=1.15, radius=0.014)
    add_text(ax, 0.363, 0.892, "RASP-LRM view", size=6.6, weight="bold", color=COL["blue"], ha="left")
    add_text(
        ax,
        0.363,
        0.868,
        "stage-specific prior + causal recent activity → a temporary mask for the current request",
        size=4.9,
        color=COL["muted"],
        ha="left",
    )
    draw_stage_strip(ax, 0.367, 0.820, 0.274, 0.028, compact=True)
    ax.add_patch(
        Polygon(
            [(0.493, 0.816), (0.503, 0.816), (0.498, 0.801)],
            closed=True,
            facecolor=COL["ink"],
            edgecolor="none",
            zorder=7,
        )
    )
    add_text(ax, 0.498, 0.804, "active", size=4.0, color=COL["ink"])
    arrow(ax, (0.653, 0.835), (0.700, 0.835), color=COL["blue"], lw=1.0)
    rounded_box(ax, 0.705, 0.810, 0.092, 0.050, face=COL["gold_light"], edge=COL["gold"], radius=0.009)
    add_text(ax, 0.751, 0.845, "protect", size=5.1, weight="bold", color=COL["gold"])
    add_text(ax, 0.751, 0.823, "stage core", size=4.6, color=COL["muted"])
    arrow(ax, (0.800, 0.835), (0.845, 0.835), color=COL["blue"], lw=1.0)
    rounded_box(ax, 0.850, 0.810, 0.103, 0.050, face=COL["blue_light"], edge=COL["blue"], radius=0.009)
    add_text(ax, 0.9015, 0.845, "rank rest", size=5.1, weight="bold", color=COL["blue"])
    add_text(ax, 0.9015, 0.823, "instance-aware", size=4.4, color=COL["muted"])
    for i, (label, states, color) in enumerate(
        [
            ("setup", ["protected", "kept", "pruned", "kept", "protected", "pruned", "kept", "pruned"], STAGE_COLORS["SETUP"]),
            ("reason", ["kept", "protected", "kept", "pruned", "protected", "kept", "pruned", "pruned"], STAGE_COLORS["REASONING"]),
            ("verify", ["pruned", "protected", "kept", "protected", "kept", "pruned", "kept", "pruned"], STAGE_COLORS["VERIFY"]),
        ]
    ):
        yy = 0.783 - i * 0.021
        add_text(ax, 0.376, yy + 0.006, label, size=4.2, weight="bold", color=color, ha="left")
        draw_channel_vector(ax, 0.426, yy, states, cell_w=0.017, cell_h=0.012)
    add_text(ax, 0.782, 0.770, "same dense backbone,\nstage-varying masks", size=4.2, weight="bold", color=COL["blue"])

    # ------------------------------------------------------------------
    # Main workflow panels.
    # ------------------------------------------------------------------
    rounded_box(ax, 0.018, 0.160, 0.255, 0.560, face="#FAF8FD", edge="#D9D0E8", radius=0.016)
    rounded_box(ax, 0.295, 0.160, 0.397, 0.560, face="#F8FBFD", edge="#D5E2EA", radius=0.016)
    rounded_box(ax, 0.714, 0.160, 0.268, 0.560, face="#FCFCFD", edge="#DCE1E6", radius=0.016)

    add_text(ax, 0.038, 0.700, "a", size=9.2, weight="bold", ha="left")
    add_text(ax, 0.063, 0.700, "Build shared stage knowledge", size=7.0, weight="bold", ha="left")
    add_text(ax, 0.315, 0.700, "b", size=9.2, weight="bold", ha="left", color=COL["blue"])
    add_text(ax, 0.340, 0.700, "Online causal mask controller", size=7.0, weight="bold", ha="left")
    add_text(ax, 0.734, 0.700, "c", size=9.2, weight="bold", ha="left")
    add_text(ax, 0.759, 0.700, "Execute inside gated MLP", size=7.0, weight="bold", ha="left")

    # Panel a: offline calibration.
    rounded_box(ax, 0.042, 0.604, 0.207, 0.064, face=COL["blue_light"], edge=COL["blue"], radius=0.010)
    draw_document_stack(ax, 0.054, 0.618, 0.032, 0.033)
    add_text(ax, 0.101, 0.646, "Dense calibration trajectories", size=5.3, weight="bold", ha="left")
    draw_stage_strip(ax, 0.100, 0.618, 0.130, 0.021, compact=True)

    arrow(ax, (0.145, 0.598), (0.145, 0.566), color=COL["violet"], lw=1.15)
    rounded_box(ax, 0.042, 0.484, 0.207, 0.075, face=COL["white"], edge=COL["violet"], radius=0.010)
    add_text(ax, 0.146, 0.541, "Observe MLP channel behavior", size=5.6, weight="bold")
    add_text(ax, 0.146, 0.516, "importance and mean by stage × layer", size=4.8, color=COL["muted"])
    draw_channel_vector(ax, 0.071, 0.493, ["protected", "kept", "pruned", "kept", "protected", "pruned", "kept", "kept"], cell_w=0.019, cell_h=0.015)

    arrow(ax, (0.145, 0.478), (0.145, 0.446), color=COL["violet"], lw=1.15)
    rounded_box(ax, 0.042, 0.270, 0.207, 0.168, face=COL["violet_light"], edge=COL["violet"], radius=0.011)
    add_text(ax, 0.146, 0.416, "Stage prior bank", size=6.3, weight="bold", color=COL["violet"])
    add_text(ax, 0.146, 0.393, r"stores $I_{s,l}$ and $\mu_{s,l}$, not final masks", size=4.7, color=COL["muted"])
    stage_masks = [
        ("Setup", ["protected", "kept", "pruned", "kept", "protected", "pruned", "kept", "pruned"]),
        ("Reason", ["kept", "protected", "kept", "pruned", "protected", "kept", "pruned", "pruned"]),
        ("Verify", ["pruned", "protected", "kept", "protected", "kept", "pruned", "kept", "pruned"]),
        ("Final", ["kept", "pruned", "protected", "kept", "protected", "pruned", "pruned", "kept"]),
    ]
    for i, (label, states) in enumerate(stage_masks):
        yy = 0.363 - i * 0.027
        add_text(ax, 0.061, yy + 0.007, label, size=4.4, weight="bold", color=list(STAGE_COLORS.values())[i], ha="left")
        draw_channel_vector(ax, 0.112, yy, states, cell_w=0.0165, cell_h=0.014)
    add_text(ax, 0.146, 0.238, "built once · shared by all requests", size=4.7, color=COL["muted"])

    # a -> b cross-boundary arrow.
    arrow(ax, (0.252, 0.356), (0.305, 0.565), color=COL["violet"], lw=1.6, connectionstyle="arc3,rad=-0.10")
    add_text(ax, 0.273, 0.473, "select prior\nby stage", size=4.6, weight="bold", color=COL["violet"])

    # Panel b: online controller.
    rounded_box(ax, 0.323, 0.612, 0.340, 0.060, face=COL["white"], edge=COL["blue"], radius=0.010)
    add_text(ax, 0.340, 0.653, "Current reasoning trajectory", size=5.3, weight="bold", ha="left")
    draw_stage_strip(ax, 0.340, 0.625, 0.211, 0.022, compact=True)
    ax.add_patch(Polygon([(0.454, 0.621), (0.463, 0.621), (0.4585, 0.608)], closed=True, facecolor=COL["ink"], edgecolor="none", zorder=7))
    rounded_box(ax, 0.575, 0.622, 0.063, 0.029, face=COL["blue_light"], edge=COL["blue"], radius=0.007)
    add_text(ax, 0.6065, 0.6365, "stage $s_t$", size=5.5, weight="bold", color=COL["blue"])

    rounded_box(ax, 0.323, 0.508, 0.150, 0.074, face=COL["violet_light"], edge=COL["violet"], radius=0.010)
    draw_shield(ax, 0.348, 0.545, 0.55)
    add_text(ax, 0.411, 0.562, "Stage prior", size=5.6, weight="bold", color=COL["violet"])
    add_text(ax, 0.411, 0.538, r"$I_{s_t,l},\ \mu_{s_t,l}$", size=5.9, color=COL["violet"])
    add_text(ax, 0.411, 0.520, "stable protection", size=4.5, color=COL["muted"])

    rounded_box(ax, 0.512, 0.508, 0.150, 0.074, face=COL["teal_light"], edge=COL["teal"], radius=0.010)
    for i, hh in enumerate([0.011, 0.020, 0.014, 0.026, 0.018]):
        ax.add_patch(Rectangle((0.529 + i * 0.015, 0.522), 0.009, hh, facecolor=COL["teal"], edgecolor="none", zorder=5))
    add_text(ax, 0.600, 0.562, "Recent activity", size=5.6, weight="bold", color=COL["teal"])
    add_text(ax, 0.600, 0.538, r"$R_{s_t,l}(t)$", size=6.1, color=COL["teal"])
    add_text(ax, 0.600, 0.520, "current request", size=4.5, color=COL["muted"])

    arrow(ax, (0.398, 0.503), (0.451, 0.455), color=COL["violet"], lw=1.15)
    arrow(ax, (0.587, 0.503), (0.535, 0.455), color=COL["teal"], lw=1.15)
    rounded_box(ax, 0.402, 0.388, 0.184, 0.071, face=COL["blue_light"], edge=COL["blue"], lw=1.15, radius=0.011)
    add_text(ax, 0.494, 0.439, "Dynamic selector", size=6.2, weight="bold", color=COL["blue"])
    add_text(ax, 0.494, 0.416, "protect first, then rank the non-core", size=4.8, color=COL["muted"])
    add_text(ax, 0.494, 0.398, r"$P=\lambda_p z(-I)+\lambda_r z(-R)$", size=5.4, color=COL["blue"])

    arrow(ax, (0.494, 0.383), (0.494, 0.354), color=COL["blue"], lw=1.25)
    rounded_box(ax, 0.328, 0.284, 0.329, 0.063, face=COL["white"], edge=COL["blue"], radius=0.010)
    add_text(ax, 0.352, 0.328, "temporary mask cache", size=5.4, weight="bold", color=COL["blue"], ha="left")
    draw_channel_vector(ax, 0.350, 0.296, ["protected", "kept", "pruned", "kept", "protected", "pruned", "kept", "pruned"], cell_w=0.026, cell_h=0.022)
    add_text(ax, 0.561, 0.306, r"$m_{s_t,l}(t)$", size=6.2, weight="bold", color=COL["blue"])

    rounded_box(ax, 0.327, 0.198, 0.100, 0.043, face=COL["white"], edge=COL["line"], radius=0.008)
    add_text(ax, 0.377, 0.226, "dense warmup", size=4.8, weight="bold", color=COL["muted"])
    add_text(ax, 0.377, 0.209, "observe before prune", size=4.1, color=COL["muted"])
    rounded_box(ax, 0.443, 0.198, 0.102, 0.043, face=COL["white"], edge=COL["line"], radius=0.008)
    add_text(ax, 0.494, 0.226, "refresh/cache", size=4.8, weight="bold", color=COL["muted"])
    add_text(ax, 0.494, 0.209, r"reuse until $\Delta_s$", size=4.1, color=COL["muted"])
    rounded_box(ax, 0.561, 0.198, 0.101, 0.043, face=COL["rose_light"], edge=COL["rose"], radius=0.008)
    add_text(ax, 0.6115, 0.226, "dense fallback", size=4.8, weight="bold", color=COL["rose"])
    add_text(ax, 0.6115, 0.209, "invalid protocol", size=4.1, color=COL["rose"])

    # b -> c runtime mask.
    arrow(ax, (0.657, 0.316), (0.724, 0.402), color=COL["blue"], lw=1.7, connectionstyle="arc3,rad=-0.08")
    add_text(ax, 0.686, 0.377, "mask", size=4.8, weight="bold", color=COL["blue"])

    # Panel c: exact MLP site.
    rounded_box(ax, 0.742, 0.607, 0.211, 0.063, face=COL["white"], edge=COL["line"], radius=0.010)
    add_text(ax, 0.759, 0.648, "input $x_{l,t}$", size=5.2, ha="left")
    rounded_box(ax, 0.824, 0.641, 0.105, 0.024, face=COL["blue_light"], edge=COL["blue"], radius=0.005)
    add_text(ax, 0.8765, 0.653, "gate projection", size=4.5)
    rounded_box(ax, 0.824, 0.611, 0.105, 0.024, face=COL["teal_light"], edge=COL["teal"], radius=0.005)
    add_text(ax, 0.8765, 0.623, "up projection", size=4.5)
    arrow(ax, (0.805, 0.640), (0.822, 0.653), color=COL["ink"], lw=0.75)
    arrow(ax, (0.805, 0.636), (0.822, 0.623), color=COL["ink"], lw=0.75)

    arrow(ax, (0.848, 0.607), (0.848, 0.573), color=COL["ink"], lw=0.9)
    add_text(ax, 0.848, 0.586, r"$\odot$", size=8.5, weight="bold")
    rounded_box(ax, 0.742, 0.511, 0.211, 0.054, face=COL["white"], edge=COL["blue"], radius=0.010)
    add_text(ax, 0.848, 0.552, "intermediate MLP channels", size=5.4, weight="bold")
    draw_channel_vector(ax, 0.767, 0.521, ["protected", "kept", "kept", "kept", "protected", "kept", "kept", "kept"], cell_w=0.0235, cell_h=0.020)

    arrow(ax, (0.848, 0.507), (0.848, 0.475), color=COL["blue"], lw=1.1)
    rounded_box(ax, 0.742, 0.407, 0.211, 0.061, face=COL["gold_light"], edge=COL["gold"], radius=0.010)
    add_text(ax, 0.848, 0.451, "apply temporary mask from (b)", size=5.3, weight="bold")
    draw_channel_vector(ax, 0.767, 0.419, ["protected", "kept", "pruned", "kept", "protected", "pruned", "kept", "pruned"], cell_w=0.0235, cell_h=0.020)

    arrow(ax, (0.848, 0.403), (0.848, 0.370), color=COL["ink"], lw=0.9)
    rounded_box(ax, 0.742, 0.320, 0.211, 0.045, face=COL["violet_light"], edge=COL["violet"], radius=0.008)
    add_text(ax, 0.848, 0.342, r"mean compensation: $\bar h=m\odot h+(1-m)\odot\mu$", size=4.9)

    arrow(ax, (0.848, 0.316), (0.848, 0.285), color=COL["ink"], lw=0.9)
    rounded_box(ax, 0.789, 0.252, 0.118, 0.030, face=COL["white"], edge=COL["ink"], radius=0.007)
    add_text(ax, 0.848, 0.267, "down projection $W_l^d$", size=4.8)
    arrow(ax, (0.908, 0.267), (0.938, 0.267), color=COL["ink"], lw=0.9)
    add_text(ax, 0.948, 0.267, "out", size=4.7, ha="right")

    # c -> b feedback.
    arrow(ax, (0.745, 0.530), (0.662, 0.548), color=COL["teal"], lw=1.6, connectionstyle="arc3,rad=0.10")
    add_text(ax, 0.699, 0.558, "activation\nfeedback", size=4.2, weight="bold", color=COL["teal"])

    # Direct legend and boundary statement.
    legend_y = 0.116
    draw_channel_vector(ax, 0.206, legend_y, ["protected"], cell_w=0.018, cell_h=0.015)
    add_text(ax, 0.226, legend_y + 0.0075, "protected", size=4.8, ha="left")
    draw_channel_vector(ax, 0.318, legend_y, ["kept"], cell_w=0.018, cell_h=0.015)
    add_text(ax, 0.338, legend_y + 0.0075, "kept active", size=4.8, ha="left")
    draw_channel_vector(ax, 0.430, legend_y, ["pruned"], cell_w=0.018, cell_h=0.015)
    add_text(ax, 0.450, legend_y + 0.0075, "masked and compensated", size=4.8, ha="left")
    add_text(
        ax,
        0.5,
        0.073,
        "Per request, RASP stores only stage state, recent activation windows, and mask cache; it does not train or create a separate pruned model for each instance.",
        size=5.2,
        color=COL["muted"],
    )

    return fig


def main():
    fig = build_figure_v4()
    stem = OUT_DIR / "rasp_lrm_method_overview"
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight", pad_inches=0.03)
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.03)
    fig.savefig(stem.with_suffix(".png"), dpi=420, bbox_inches="tight", pad_inches=0.03)
    fig.savefig(stem.with_suffix(".tiff"), dpi=600, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    print(stem)


if __name__ == "__main__":
    main()
