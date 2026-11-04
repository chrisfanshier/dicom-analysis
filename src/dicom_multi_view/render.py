"""Compose section image + depth scale bar (cm) for display."""

from __future__ import annotations

from PIL import Image, ImageDraw, ImageFont

import numpy as np

from dicom_multi_view.section import LoadedSection, hu_to_u8

# Cap native compose height; GUI zooms this sheet uniformly (bar stays truthful).
BASE_MAX_HEIGHT_PX = 2400


def _font(size: int = 12) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("arial.ttf", size)
    except OSError:
        try:
            return ImageFont.truetype("DejaVuSans.ttf", size)
        except OSError:
            return ImageFont.load_default()


def _nice_step_cm(extent_cm: float, target_ticks: int = 6) -> float:
    """Nice tick step in cm."""
    if extent_cm <= 0:
        return 1.0
    raw = extent_cm / max(target_ticks, 1)
    exp = int(np.floor(np.log10(max(raw, 1e-6))))
    base = 10.0**exp
    for m in (1.0, 2.0, 5.0, 10.0):
        if m * base >= raw * 0.8:
            return m * base
    return 10.0 * base


def _draw_scale_bar_cm(
    height_px: int,
    depth_mm_per_px: float,
    *,
    top_depth_cm: float = 0.0,
    bar_width_px: int = 64,
) -> Image.Image:
    """
    Depth scale bar in cm. ``depth_mm_per_px`` is physical mm per image row
    at this compose resolution; zoom must scale section+bar together.
    """
    bar = Image.new("RGB", (bar_width_px, height_px), (32, 32, 36))
    draw = ImageDraw.Draw(bar)
    font = _font(11)
    font_sm = _font(10)

    depth_cm_per_px = depth_mm_per_px / 10.0
    extent_cm = height_px * depth_cm_per_px
    step = _nice_step_cm(extent_cm)

    x_rule = 12
    draw.line([(x_rule, 0), (x_rule, height_px - 1)], fill=(220, 220, 220), width=2)

    first = np.ceil(top_depth_cm / step) * step
    d = float(first)
    end = top_depth_cm + extent_cm + 1e-9
    while d <= end:
        y = (d - top_depth_cm) / max(depth_cm_per_px, 1e-12)
        if 0 <= y < height_px:
            yi = int(round(y))
            draw.line([(x_rule - 2, yi), (x_rule + 10, yi)], fill=(230, 230, 230), width=1)
            label = f"{d:.0f}" if abs(d - round(d)) < 1e-6 else f"{d:.1f}"
            draw.text((x_rule + 12, max(0, yi - 6)), label, fill=(210, 210, 215), font=font_sm)
        d += step

    draw.text((4, 2), "cm", fill=(170, 170, 180), font=font)
    draw.text((4, height_px - 14), "depth", fill=(120, 120, 130), font=font_sm)
    return bar


def compose_section_rgb(
    section: LoadedSection,
    *,
    top_depth_cm: float | None = None,
    bar_width_px: int = 64,
    win_lo: float | None = None,
    win_hi: float | None = None,
    max_height_px: int | None = BASE_MAX_HEIGHT_PX,
) -> Image.Image:
    """Greyscale section + cm depth scale bar on the right."""
    if top_depth_cm is None:
        top_depth_cm = section.top_depth_cm

    u8 = hu_to_u8(section.image_hu, win_lo, win_hi)
    h, w = u8.shape
    img = Image.fromarray(np.stack([u8, u8, u8], axis=-1), mode="RGB")

    if max_height_px and h > max_height_px:
        scale = max_height_px / h
        new_w = max(1, int(round(w * scale)))
        img = img.resize((new_w, max_height_px), Image.Resampling.BILINEAR)
        depth_mm_per_px = section.depth_mm_per_px / scale
        h, w = max_height_px, new_w
    else:
        depth_mm_per_px = section.depth_mm_per_px

    bar = _draw_scale_bar_cm(
        h,
        depth_mm_per_px,
        top_depth_cm=top_depth_cm,
        bar_width_px=bar_width_px,
    )
    out = Image.new("RGB", (w + bar_width_px, h), (20, 20, 22))
    out.paste(img, (0, 0))
    out.paste(bar, (w, 0))
    return out


def title_strip(text: str, width: int, height: int = 24) -> Image.Image:
    strip = Image.new("RGB", (width, height), (40, 42, 48))
    draw = ImageDraw.Draw(strip)
    draw.text((6, 5), text, fill=(230, 230, 235), font=_font(11))
    return strip
