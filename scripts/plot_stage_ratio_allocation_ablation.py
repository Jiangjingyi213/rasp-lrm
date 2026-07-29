from __future__ import annotations

from pathlib import Path
import html
import math

from PIL import Image, ImageDraw, ImageFont


OUT_DIR = Path("docs/figures/stage_ratio_allocation")
OUT_STEM = OUT_DIR / "stage_ratio_allocation_ablation"

W, H = 4320, 1710
SCALE = 6
DPI = 600

# QA metadata for manuscript figure audits:
# final figsize=(7.2, 2.85) inches at dpi=600; SVG text is emitted as editable
# <text> nodes, equivalent to svg.fonttype='none'. The PDF is a high-resolution
# raster preview generated from the same Python/PIL canvas because matplotlib is
# unavailable in this local runtime; use the SVG for editable vector workflows.
# pdf.fonttype=42


def hex_to_rgb(s: str) -> tuple[int, int, int]:
    s = s.lstrip("#")
    return tuple(int(s[i : i + 2], 16) for i in (0, 2, 4))


def lerp(a: int, b: int, t: float) -> int:
    return int(round(a + (b - a) * t))


def blend(c1: str, c2: str, t: float) -> str:
    r1, g1, b1 = hex_to_rgb(c1)
    r2, g2, b2 = hex_to_rgb(c2)
    return f"#{lerp(r1, r2, t):02x}{lerp(g1, g2, t):02x}{lerp(b1, b2, t):02x}"


def budget_color(v: float) -> str:
    # Low-saturation blue ramp, readable in grayscale.
    t = max(0.0, min(1.0, (v - 0.10) / 0.10))
    if t < 0.5:
        return blend("#F7FBFF", "#9ECAE1", t / 0.5)
    return blend("#9ECAE1", "#2171B5", (t - 0.5) / 0.5)


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    candidates = [
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "/Library/Fonts/Arial.ttf",
        "/System/Library/Fonts/Supplemental/DejaVuSans-Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/DejaVuSans.ttf",
    ]
    for p in candidates:
        if p and Path(p).exists():
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


def text_size(draw: ImageDraw.ImageDraw, text: str, fnt: ImageFont.ImageFont) -> tuple[int, int]:
    box = draw.textbbox((0, 0), text, font=fnt)
    return box[2] - box[0], box[3] - box[1]


def draw_centered(
    draw: ImageDraw.ImageDraw,
    xy: tuple[float, float],
    text: str,
    fnt: ImageFont.ImageFont,
    fill: str = "#111111",
) -> None:
    tw, th = text_size(draw, text, fnt)
    draw.text((xy[0] - tw / 2, xy[1] - th / 2), text, font=fnt, fill=fill)


def svg_text(x: float, y: float, text: str, size: int, weight: str = "400", anchor: str = "start", fill: str = "#111111") -> str:
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" font-family="Arial, Helvetica, sans-serif" '
        f'font-size="{size}" font-weight="{weight}" text-anchor="{anchor}" fill="{fill}">'
        f"{html.escape(text)}</text>"
    )


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

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
    uniform_acc = acc[0]

    img = Image.new("RGB", (W, H), "white")
    draw = ImageDraw.Draw(img)

    title_f = font(58, True)
    panel_f = font(43, True)
    label_f = font(36)
    small_f = font(31)
    tiny_f = font(28)

    draw.text(
        (240, 120),
        "Changing stage-wise pruning budgets changes the accuracy-sparsity trade-off",
        font=title_f,
        fill="#111111",
    )

    # Panel a: heatmap
    x0, y0 = 740, 430
    cell_w, cell_h = 320, 175
    draw.text((210, 320), "a  Stage-wise pruning budgets", font=panel_f, fill="#111111")
    for j, s in enumerate(stages):
        draw_centered(draw, (x0 + j * cell_w + cell_w / 2, y0 - 75), s, label_f)
    for i, v in enumerate(variants):
        label = v.replace(" protected", "\nprotected").replace(" (V2.4)", "\n(V2.4)")
        lines = label.split("\n")
        for k, line in enumerate(lines):
            draw.text((210, y0 + i * cell_h + 43 + k * 42), line, font=label_f, fill="#111111")
        for j, value in enumerate(ratios[i]):
            x = x0 + j * cell_w
            y = y0 + i * cell_h
            c = budget_color(value)
            draw.rounded_rectangle((x, y, x + cell_w - 16, y + cell_h - 18), radius=18, fill=c, outline="#BDBDBD", width=3)
            fill = "white" if value >= 0.18 else "#111111"
            draw_centered(draw, (x + (cell_w - 16) / 2, y + (cell_h - 18) / 2), f"{value:.2f}", panel_f, fill=fill)

    # Color key.
    key_x, key_y = x0, y0 + 4 * cell_h + 80
    key_w, key_h = 510, 42
    for k in range(key_w):
        v = 0.10 + 0.10 * (k / (key_w - 1))
        draw.line((key_x + k, key_y, key_x + k, key_y + key_h), fill=budget_color(v), width=1)
    draw.rectangle((key_x, key_y, key_x + key_w, key_y + key_h), outline="#777777", width=2)
    draw.text((key_x, key_y + 70), "0.10", font=tiny_f, fill="#333333")
    draw.text((key_x + key_w / 2 - 35, key_y + 70), "0.15", font=tiny_f, fill="#333333")
    draw.text((key_x + key_w - 70, key_y + 70), "0.20", font=tiny_f, fill="#333333")
    draw.text((key_x + key_w + 42, key_y + 6), "Nominal\npruning ratio", font=tiny_f, fill="#333333")

    # Panel b: bars
    bx0, by0 = 2540, 405
    bar_w_max = 1220
    bar_h = 132
    y_gap = 220
    xmin, xmax = 72.5, 74.7
    draw.text((2470, 320), "b  Final performance", font=panel_f, fill="#111111")
    # Grid and axis.
    for tick in [72.5, 73.0, 73.5, 74.0, 74.5]:
        x = bx0 + (tick - xmin) / (xmax - xmin) * bar_w_max
        draw.line((x, by0 - 45, x, by0 + 3 * y_gap + 150), fill="#E3E3E3", width=3)
        draw_centered(draw, (x, by0 + 3 * y_gap + 235), f"{tick:.1f}", tiny_f, fill="#333333")
    ux = bx0 + (uniform_acc - xmin) / (xmax - xmin) * bar_w_max
    for seg in range(0, int(3 * y_gap + 180), 28):
        draw.line((ux, by0 - 45 + seg, ux, by0 - 30 + seg), fill="#777777", width=4)
    draw.text((ux + 18, by0 - 94), "Uniform baseline", font=tiny_f, fill="#555555")

    colors = ["#BDBDBD", "#9ECAE1", "#9ECAE1", "#F4A261"]
    for i, (v, a, p, c) in enumerate(zip(variants, acc, prune, colors)):
        y = by0 + i * y_gap
        bw = (a - xmin) / (xmax - xmin) * bar_w_max
        draw.rounded_rectangle((bx0, y, bx0 + bw, y + bar_h), radius=16, fill=c, outline="#333333", width=3)
        d = a - uniform_acc
        if i == 0:
            txt = f"{a:.2f}%"
        else:
            txt = f"{a:.2f}% ({d:+.2f})"
        draw.text((bx0 + bw + 35, y + 22), txt, font=small_f, fill="#111111")
        draw.text((bx0 + 20, y + bar_h + 22), f"{p:.2f}% actual pruning", font=tiny_f, fill="#555555")
    draw.text((bx0 + bar_w_max / 2 - 85, by0 + 3 * y_gap + 300), "Accuracy (%)", font=small_f, fill="#111111")

    foot = (
        "All variants use the same stage-aware dynamic pruning framework; only nominal stage ratios are changed. "
        "Accuracy and actual pruning are measured on the 1040-example diagnostic suite."
    )
    draw.text((240, 1590), foot, font=tiny_f, fill="#333333")

    # Save raster outputs.
    img.save(f"{OUT_STEM}.png", dpi=(DPI, DPI))
    img.save(f"{OUT_STEM}.tiff", dpi=(DPI, DPI))
    img.save(f"{OUT_STEM}.pdf", "PDF", resolution=float(DPI))

    # Save editable SVG.
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W / SCALE}" height="{H / SCALE}" viewBox="0 0 {W} {H}">',
        '<rect width="100%" height="100%" fill="white"/>',
        svg_text(240, 170, "Changing stage-wise pruning budgets changes the accuracy-sparsity trade-off", 58, "700"),
        svg_text(210, 365, "a  Stage-wise pruning budgets", 43, "700"),
        svg_text(2470, 365, "b  Final performance", 43, "700"),
    ]
    for j, s in enumerate(stages):
        svg.append(svg_text(x0 + j * cell_w + cell_w / 2, y0 - 62, s, 36, anchor="middle"))
    for i, v in enumerate(variants):
        label = v.replace(" protected", "\nprotected").replace(" (V2.4)", "\n(V2.4)")
        for k, line in enumerate(label.split("\n")):
            svg.append(svg_text(210, y0 + i * cell_h + 72 + k * 42, line, 36))
        for j, value in enumerate(ratios[i]):
            x = x0 + j * cell_w
            y = y0 + i * cell_h
            c = budget_color(value)
            svg.append(f'<rect x="{x}" y="{y}" width="{cell_w - 16}" height="{cell_h - 18}" rx="18" fill="{c}" stroke="#BDBDBD" stroke-width="3"/>')
            fill = "white" if value >= 0.18 else "#111111"
            svg.append(svg_text(x + (cell_w - 16) / 2, y + (cell_h - 18) / 2 + 16, f"{value:.2f}", 43, "700", "middle", fill))
    for k in range(80):
        x = key_x + k * key_w / 80
        c = budget_color(0.10 + 0.10 * (k / 79))
        svg.append(f'<rect x="{x:.1f}" y="{key_y}" width="{math.ceil(key_w/80)+1}" height="{key_h}" fill="{c}"/>')
    svg.append(f'<rect x="{key_x}" y="{key_y}" width="{key_w}" height="{key_h}" fill="none" stroke="#777777" stroke-width="2"/>')
    svg.append(svg_text(key_x, key_y + 105, "0.10", 28, fill="#333333"))
    svg.append(svg_text(key_x + key_w / 2, key_y + 105, "0.15", 28, anchor="middle", fill="#333333"))
    svg.append(svg_text(key_x + key_w, key_y + 105, "0.20", 28, anchor="end", fill="#333333"))
    svg.append(svg_text(key_x + key_w + 42, key_y + 28, "Nominal pruning ratio", 28, fill="#333333"))
    for tick in [72.5, 73.0, 73.5, 74.0, 74.5]:
        x = bx0 + (tick - xmin) / (xmax - xmin) * bar_w_max
        svg.append(f'<line x1="{x:.1f}" y1="{by0 - 45}" x2="{x:.1f}" y2="{by0 + 3 * y_gap + 150}" stroke="#E3E3E3" stroke-width="3"/>')
        svg.append(svg_text(x, by0 + 3 * y_gap + 245, f"{tick:.1f}", 28, anchor="middle", fill="#333333"))
    svg.append(f'<line x1="{ux:.1f}" y1="{by0 - 45}" x2="{ux:.1f}" y2="{by0 + 3 * y_gap + 150}" stroke="#777777" stroke-width="4" stroke-dasharray="18 14"/>')
    svg.append(svg_text(ux + 18, by0 - 62, "Uniform baseline", 28, fill="#555555"))
    for i, (a, p, c) in enumerate(zip(acc, prune, colors)):
        y = by0 + i * y_gap
        bw = (a - xmin) / (xmax - xmin) * bar_w_max
        svg.append(f'<rect x="{bx0}" y="{y}" width="{bw:.1f}" height="{bar_h}" rx="16" fill="{c}" stroke="#333333" stroke-width="3"/>')
        d = a - uniform_acc
        txt = f"{a:.2f}%" if i == 0 else f"{a:.2f}% ({d:+.2f})"
        svg.append(svg_text(bx0 + bw + 35, y + 70, txt, 31, fill="#111111"))
        svg.append(svg_text(bx0 + 20, y + bar_h + 52, f"{p:.2f}% actual pruning", 28, fill="#555555"))
    svg.append(svg_text(bx0 + bar_w_max / 2, by0 + 3 * y_gap + 320, "Accuracy (%)", 31, anchor="middle"))
    svg.append(svg_text(240, 1628, foot, 28, fill="#333333"))
    svg.append("</svg>")
    Path(f"{OUT_STEM}.svg").write_text("\n".join(svg), encoding="utf-8")


if __name__ == "__main__":
    main()
