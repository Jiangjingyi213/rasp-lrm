from __future__ import annotations

import html
import json
import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


OUT_DIR = Path("docs/figures/stage_dynamic_analysis")
MASK_DIR = Path("runs/analysis_stage_dynamic_masks_gsm8k32")
W, H = 3600, 1800
DPI = 600
FONT_REG = "/System/Library/Fonts/Supplemental/Comic Sans MS.ttf"
FONT_BOLD = "/System/Library/Fonts/Supplemental/Comic Sans MS Bold.ttf"


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(FONT_BOLD if bold else FONT_REG, size)


def text_size(draw: ImageDraw.ImageDraw, text: str, fnt: ImageFont.ImageFont) -> tuple[int, int]:
    box = draw.textbbox((0, 0), text, font=fnt)
    return box[2] - box[0], box[3] - box[1]


def center(draw: ImageDraw.ImageDraw, xy: tuple[float, float], text: str, fnt, fill="#111111"):
    tw, th = text_size(draw, text, fnt)
    draw.text((xy[0] - tw / 2, xy[1] - th / 2), text, font=fnt, fill=fill)


def hex_to_rgb(s: str) -> tuple[int, int, int]:
    s = s.lstrip("#")
    return tuple(int(s[i : i + 2], 16) for i in (0, 2, 4))


def blend(c1: str, c2: str, t: float) -> str:
    a = hex_to_rgb(c1)
    b = hex_to_rgb(c2)
    vals = [round(a[i] + (b[i] - a[i]) * max(0, min(1, t))) for i in range(3)]
    return f"#{vals[0]:02x}{vals[1]:02x}{vals[2]:02x}"


def budget_color(v: float) -> str:
    t = (v - 0.10) / 0.10
    if t < 0.5:
        return blend("#F7FBFF", "#9ECAE1", t / 0.5)
    return blend("#9ECAE1", "#2171B5", (t - 0.5) / 0.5)


def jaccard_color(v: float, vmin=0.50, vmax=0.90) -> str:
    # Lower overlap = stronger difference, encoded warmer.
    t = (v - vmin) / (vmax - vmin)
    return blend("#F4A261", "#EAF4FB", t)


def save_all(img: Image.Image, stem: Path, svg_text: str | None = None) -> None:
    img.save(f"{stem}.png", dpi=(DPI, DPI))
    img.save(f"{stem}.tiff", dpi=(DPI, DPI))
    img.save(f"{stem}.pdf", "PDF", resolution=float(DPI))
    if svg_text:
        Path(f"{stem}.svg").write_text(svg_text, encoding="utf-8")


def svg_header() -> list[str]:
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W/6}" height="{H/6}" viewBox="0 0 {W} {H}">',
        '<rect width="100%" height="100%" fill="white"/>',
    ]


def svg_t(x, y, text, size, weight="400", anchor="start", fill="#111111") -> str:
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" font-family="Comic Sans MS, Comic Sans, cursive" '
        f'font-size="{size}" font-weight="{weight}" text-anchor="{anchor}" fill="{fill}">'
        f"{html.escape(text)}</text>"
    )


def fig_stage_ratio() -> None:
    variants = ["Uniform", "Reasoning protected", "Verify protected", "Final-prune (V2.4)"]
    stages = ["Setup", "Reasoning", "Verify", "Final"]
    ratios = [
        [0.15, 0.15, 0.15, 0.15],
        [0.20, 0.10, 0.20, 0.10],
        [0.20, 0.20, 0.10, 0.10],
        [0.15, 0.20, 0.10, 0.15],
    ]
    acc = [73.94, 73.27, 72.88, 74.33]
    prune = [12.41, 12.68, 14.35, 13.20]

    img = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(img)
    f_panel = font(64, True)
    f_label = font(52)
    f_val = font(60, True)
    f_small = font(48)
    f_tiny = font(43)

    x0, y0 = 660, 390
    cw, ch = 270, 165
    left_center = (135 + (x0 + 4 * cw)) / 2
    center(d, (left_center, 185), "a  Stage-wise pruning budgets", f_panel)
    for j, s in enumerate(stages):
        center(d, (x0 + j * cw + cw / 2, y0 - 62), s, f_label)
    for i, v in enumerate(variants):
        label = v.replace(" protected", "\nprotected").replace(" (V2.4)", "\n(V2.4)")
        for k, line in enumerate(label.split("\n")):
            d.text((135, y0 + i * ch + 42 + k * 47), line, font=f_label, fill="#111111")
        for j, value in enumerate(ratios[i]):
            x = x0 + j * cw
            y = y0 + i * ch
            color = budget_color(value)
            d.rounded_rectangle((x, y, x + cw - 16, y + ch - 18), radius=18, fill=color, outline="#BDBDBD", width=3)
            center(d, (x + (cw - 16) / 2, y + (ch - 18) / 2), f"{value:.2f}", f_val, "white" if value >= 0.18 else "#111111")

    key_x, key_y, key_w, key_h = x0, y0 + 4 * ch + 80, 430, 42
    for k in range(key_w):
        d.line((key_x + k, key_y, key_x + k, key_y + key_h), fill=budget_color(0.10 + 0.10 * k / (key_w - 1)), width=1)
    d.rectangle((key_x, key_y, key_x + key_w, key_y + key_h), outline="#777777", width=2)
    d.text((key_x, key_y + 60), "0.10", font=f_tiny, fill="#333333")
    center(d, (key_x + key_w / 2, key_y + 78), "0.15", f_tiny, "#333333")
    d.text((key_x + key_w - 68, key_y + 60), "0.20", font=f_tiny, fill="#333333")
    d.text((key_x + key_w + 30, key_y - 10), "Nominal\nratio", font=f_tiny, fill="#333333")

    bx0, by0 = 2070, 390
    bar_w, bar_h, gap = 1180, 130, 225
    right_center = bx0 + bar_w / 2
    center(d, (right_center, 185), "b  Final performance", f_panel)
    xmin, xmax = 72.5, 74.7
    uniform = acc[0]
    for tick in [72.5, 73.0, 73.5, 74.0, 74.5]:
        x = bx0 + (tick - xmin) / (xmax - xmin) * bar_w
        d.line((x, by0 - 45, x, by0 + 3 * gap + 150), fill="#E5E5E5", width=3)
        center(d, (x, by0 + 3 * gap + 245), f"{tick:.1f}", f_tiny, "#333333")
    ux = bx0 + (uniform - xmin) / (xmax - xmin) * bar_w
    d.line((ux, by0 - 45, ux, by0 + 3 * gap + 150), fill="#777777", width=4)
    d.text((ux + 18, by0 - 90), "Uniform baseline", font=f_tiny, fill="#555555")
    colors = ["#BDBDBD", "#9ECAE1", "#9ECAE1", "#F4A261"]
    for i, (a, p, c) in enumerate(zip(acc, prune, colors)):
        y = by0 + i * gap
        bw = (a - xmin) / (xmax - xmin) * bar_w
        d.rounded_rectangle((bx0, y, bx0 + bw, y + bar_h), radius=18, fill=c, outline="#333333", width=3)
        txt = f"{a:.2f}%" if i == 0 else f"{a:.2f}% ({a-uniform:+.2f})"
        d.text((bx0 + bw + 35, y + 30), txt, font=f_small, fill="#111111")
        d.text((bx0 + 25, y + bar_h + 20), f"{p:.2f}% actual pruning", font=f_tiny, fill="#555555")
    center(d, (bx0 + bar_w / 2, by0 + 3 * gap + 325), "Accuracy (%)", f_small)

    d.text((135, 1640), "All variants use the same dynamic pruning framework; only nominal stage ratios are changed.", font=f_tiny, fill="#333333")
    save_all(img, OUT_DIR / "fig_stage_ratio_allocation")


def fig_stage_prior_overlap() -> None:
    data = json.load(open(MASK_DIR / "stage_prior_overlap.json"))
    stages = ["setup", "reasoning", "verify", "final"]
    labels = ["Setup", "Reasoning", "Verify", "Final"]
    mat = [[1.0 for _ in stages] for _ in stages]
    for key, val in data["stage_pair_jaccard"].items():
        a, b = key.split("_vs_")
        i, j = stages.index(a), stages.index(b)
        mat[i][j] = mat[j][i] = val["mean"]

    img = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(img)
    f_panel = font(70, True)
    f_label = font(62)
    f_val = font(62, True)
    f_note = font(52)
    x0, y0, cell = 850, 300, 285
    for i, lab in enumerate(labels):
        center(d, (x0 + i * cell + cell / 2, y0 - 70), lab, f_label)
        d.text((250, y0 + i * cell + 95), lab, font=f_label, fill="#111111")
    for i in range(4):
        for j in range(4):
            v = mat[i][j]
            color = "#F5F5F5" if i == j else jaccard_color(v)
            d.rounded_rectangle((x0 + j * cell, y0 + i * cell, x0 + (j + 1) * cell - 14, y0 + (i + 1) * cell - 14), radius=18, fill=color, outline="#BDBDBD", width=3)
            txt = "1.00" if i == j else f"{v:.2f}"
            center(d, (x0 + j * cell + cell / 2 - 7, y0 + i * cell + cell / 2 - 7), txt, f_val)
    f_right = font(78, True)
    d.text((2230, 450), "Overall stage-pair\nJaccard = 0.6927", font=f_right, fill="#111111")
    d.text((2230, 830), "Lower overlap means\nstronger stage-specific\nprior diversity.", font=f_right, fill="#333333")
    save_all(img, OUT_DIR / "fig_stage_prior_overlap")


def fig_runtime_mask_diversity() -> None:
    data = json.load(open(MASK_DIR / "runtime_mask_diversity.json"))
    stages = ["setup", "reasoning", "verify", "final"]
    labels = ["Setup", "Reasoning", "Verify", "Final"]
    vals = [data["stage_level"][s]["pairwise_jaccard"]["mean"] for s in stages]
    uniq = [data["stage_level"][s]["unique_mask_rate"]["mean"] for s in stages]
    med = [data["stage_level"][s]["pairwise_jaccard"]["median"] for s in stages]

    img = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(img)
    f_panel = font(62, True)
    f_label = font(64)
    f_val = font(58, True)
    f_small = font(50)
    x0, y0, bar_w, gap, bar_h = 620, 300, 1800, 270, 135
    xmin, xmax = 0.70, 0.90
    for tick in [0.70, 0.75, 0.80, 0.85, 0.90]:
        x = x0 + (tick - xmin) / (xmax - xmin) * bar_w
        d.line((x, y0 - 55, x, y0 + 3 * gap + 170), fill="#E5E5E5", width=3)
        center(d, (x, y0 + 3 * gap + 250), f"{tick:.2f}", f_small, "#333333")
    for i, (lab, v, u, m) in enumerate(zip(labels, vals, uniq, med)):
        y = y0 + i * gap
        d.text((185, y + 35), lab, font=f_label, fill="#111111")
        bw = (v - xmin) / (xmax - xmin) * bar_w
        color = ["#9ECAE1", "#A1D99B", "#FDD0A2", "#BCBDDC"][i]
        d.rounded_rectangle((x0, y, x0 + bw, y + bar_h), radius=18, fill=color, outline="#333333", width=3)
        d.text((x0 + bw + 35, y + 24), f"mean Jaccard {v:.3f}", font=f_val, fill="#111111")
        d.text((x0 + 25, y + bar_h + 18), f"median {m:.3f}   |   unique mask rate {u:.1f}", font=f_small, fill="#555555")
    d.text((2630, 520), "If masks were identical,\nJaccard would be 1.0.", font=f_small, fill="#333333")
    d.text((2630, 770), "All stages have unique\nmask rate = 1.0.", font=f_small, fill="#333333")
    save_all(img, OUT_DIR / "fig_runtime_mask_diversity")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig_stage_ratio()
    fig_stage_prior_overlap()
    fig_runtime_mask_diversity()


if __name__ == "__main__":
    main()
