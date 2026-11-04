"""Compose section image + depth scale bar (cm) for display."""

from __future__ import annotations

from dataclasses import dataclass

from PIL import Image, ImageDraw, ImageFont

import numpy as np

from dicom_multi_view.section import LoadedSection, hu_to_u8

# Cap native compose height for the tallest section; shared mm/px applied to all.
BASE_MAX_HEIGHT_PX = 2400
DEFAULT_BAR_WIDTH_PX = 72


def _font(size: int = 12) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("arial.ttf", size)
    except OSError:
        try:
            return ImageFont.truetype("DejaVuSans.ttf", size)
        except OSError:
            return ImageFont.load_default()


def shared_mm_per_px(
    sections: list[LoadedSection],
    *,
    max_height_px: int = BASE_MAX_HEIGHT_PX,
    max_width_px: int | None = 900,
) -> float:
    """
    One physical display scale for every section (mm per on-screen pixel).

    Chooses the coarsest scale that fits the tallest section in ``max_height_px``
    and (optionally) the widest section in ``max_width_px``, so shorter sections
    are shorter — not fatter — and diameters stay comparable.
    """
    if not sections:
        return 1.0
    heights = []
    widths = []
    for sec in sections:
        h_mm, w_mm = sec.physical_size_mm()
        heights.append(max(h_mm, 1e-6))
        widths.append(max(w_mm, 1e-6))
    mm = max(heights) / max(int(max_height_px), 1)
    if max_width_px is not None:
        mm = max(mm, max(widths) / max(int(max_width_px), 1))
    return float(max(mm, 1e-9))


def _draw_scale_bar_cm(
    height_px: int,
    depth_mm_per_px: float,
    *,
    top_depth_cm: float = 0.0,
    core_top_inset_cm: float = 0.0,
    core_bottom_inset_cm: float = 0.0,
    bar_width_px: int = DEFAULT_BAR_WIDTH_PX,
) -> Image.Image:
    """
    Depth scale bar in cm. ``depth_mm_per_px`` is physical mm per image row
    at this compose resolution; zoom must scale section+bar together.

    Ticks/labels span only between core-top and core-bottom markers.
    At the core-top line, depth = ``top_depth_cm``.
    Minor ticks every 1 cm; labeled major ticks every 10 cm.
    """
    bar = Image.new("RGB", (bar_width_px, height_px), (32, 32, 36))
    draw = ImageDraw.Draw(bar)
    font = _font(11)
    font_sm = _font(10)

    depth_cm_per_px = depth_mm_per_px / 10.0
    extent_cm = height_px * depth_cm_per_px
    top_inset = max(0.0, min(float(core_top_inset_cm), extent_cm))
    bottom_inset = max(0.0, min(float(core_bottom_inset_cm), extent_cm - top_inset))
    core_len = max(0.0, extent_cm - top_inset - bottom_inset)

    y_core_top = top_inset / max(depth_cm_per_px, 1e-12)
    y_core_bot = height_px - (bottom_inset / max(depth_cm_per_px, 1e-12))
    y0 = int(round(y_core_top))
    y1 = int(round(y_core_bot))
    if y1 <= y0:
        y1 = min(height_px - 1, y0 + 1)

    # Fixed physical ticks: 1 cm minor, label every 10 cm major.
    minor_cm = 1.0
    major_cm = 10.0

    x_rule = 12
    # Faint full-height rail, bright rule only over the core span.
    draw.line([(x_rule, 0), (x_rule, height_px - 1)], fill=(70, 70, 76), width=1)
    draw.line([(x_rule, y0), (x_rule, y1)], fill=(220, 220, 220), width=2)

    # Marker ticks on the bar matching image guides.
    for yi, color in ((y0, (80, 200, 255)), (y1, (255, 180, 80))):
        draw.line([(x_rule - 4, yi), (x_rule + 16, yi)], fill=color, width=2)

    first = np.ceil(top_depth_cm / minor_cm) * minor_cm
    d = float(first)
    end = top_depth_cm + core_len + 1e-9
    while d <= end:
        y = y_core_top + (d - top_depth_cm) / max(depth_cm_per_px, 1e-12)
        if y0 - 0.5 <= y <= y1 + 0.5:
            yi = int(round(y))
            is_major = abs(d / major_cm - round(d / major_cm)) < 1e-6
            if is_major:
                draw.line(
                    [(x_rule - 3, yi), (x_rule + 14, yi)],
                    fill=(230, 230, 230),
                    width=2,
                )
                label = f"{d:.0f}" if abs(d - round(d)) < 1e-6 else f"{d:.1f}"
                draw.text(
                    (x_rule + 16, max(0, min(height_px - 12, yi - 6))),
                    label,
                    fill=(210, 210, 215),
                    font=font_sm,
                )
            else:
                draw.line(
                    [(x_rule - 1, yi), (x_rule + 7, yi)],
                    fill=(160, 160, 165),
                    width=1,
                )
        d += minor_cm

    draw.text((4, 2), "cm", fill=(170, 170, 180), font=font)
    draw.text((4, height_px - 14), "depth", fill=(120, 120, 130), font=font_sm)
    return bar


def _draw_core_guides(
    img: Image.Image,
    *,
    depth_mm_per_px: float,
    core_top_inset_cm: float,
    core_bottom_inset_cm: float,
) -> None:
    """Horizontal core-top / core-bottom guides on the section image (in-place)."""
    w, h = img.size
    if h < 2:
        return
    depth_cm_per_px = depth_mm_per_px / 10.0
    extent_cm = h * depth_cm_per_px
    top_inset = max(0.0, min(float(core_top_inset_cm), extent_cm))
    bottom_inset = max(0.0, min(float(core_bottom_inset_cm), extent_cm - top_inset))
    y_top = int(round(top_inset / max(depth_cm_per_px, 1e-12)))
    y_bot = int(round(h - 1 - (bottom_inset / max(depth_cm_per_px, 1e-12))))
    y_top = int(np.clip(y_top, 0, h - 1))
    y_bot = int(np.clip(y_bot, 0, h - 1))
    draw = ImageDraw.Draw(img)
    # Core top = cyan; core bottom = amber.
    draw.line([(0, y_top), (w - 1, y_top)], fill=(80, 200, 255), width=2)
    draw.line([(0, y_bot), (w - 1, y_bot)], fill=(255, 180, 80), width=2)


@dataclass(frozen=True)
class ComposeLayout:
    """Pixel layout of one composed section tile (pre-zoom base sheet coords)."""

    img_w: int
    img_h: int
    bar_width: int
    title_h: int = 0

    @property
    def body_w(self) -> int:
        return self.img_w + self.bar_width

    @property
    def body_h(self) -> int:
        return self.img_h


def compose_section_rgb(
    section: LoadedSection,
    *,
    top_depth_cm: float | None = None,
    bar_width_px: int = DEFAULT_BAR_WIDTH_PX,
    win_lo: float | None = None,
    win_hi: float | None = None,
    mm_per_px: float | None = None,
    max_height_px: int | None = BASE_MAX_HEIGHT_PX,
    draw_guides: bool = True,
) -> tuple[Image.Image, ComposeLayout]:
    """
    Greyscale section + cm depth scale bar on the right.

    Prefer ``mm_per_px`` (shared across the layout). If omitted, falls back to
    per-section isotropic sizing capped by ``max_height_px`` (legacy).

    Returns ``(image, layout)`` where layout describes the image (not bar) size.
    """
    section.clamp_core_insets()
    if top_depth_cm is None:
        top_depth_cm = section.top_depth_cm
    if win_lo is None:
        win_lo = section.win_lo
    if win_hi is None:
        win_hi = section.win_hi

    u8 = hu_to_u8(section.image_hu, win_lo, win_hi)
    h0, w0 = u8.shape
    h_mm = h0 * float(section.depth_mm_per_px)
    w_mm = w0 * float(section.across_mm_per_px)

    if mm_per_px is None or mm_per_px <= 0:
        # Legacy single-section path: isotropic, then optional height cap.
        target_mm = min(
            max(float(section.depth_mm_per_px), 1e-12),
            max(float(section.across_mm_per_px), 1e-12),
        )
        iso_h = max(1, int(round(h_mm / target_mm)))
        iso_w = max(1, int(round(w_mm / target_mm)))
        if max_height_px and iso_h > max_height_px:
            scale = max_height_px / iso_h
            iso_h = max_height_px
            iso_w = max(1, int(round(iso_w * scale)))
            target_mm = target_mm / scale
        mm_per_px = target_mm
        out_h, out_w = iso_h, iso_w
    else:
        out_h = max(1, int(round(h_mm / mm_per_px)))
        out_w = max(1, int(round(w_mm / mm_per_px)))

    img = Image.fromarray(np.stack([u8, u8, u8], axis=-1), mode="RGB")
    if out_h != h0 or out_w != w0:
        img = img.resize((out_w, out_h), Image.Resampling.BILINEAR)

    if draw_guides:
        _draw_core_guides(
            img,
            depth_mm_per_px=float(mm_per_px),
            core_top_inset_cm=section.core_top_inset_cm,
            core_bottom_inset_cm=section.core_bottom_inset_cm,
        )

    bar = _draw_scale_bar_cm(
        out_h,
        float(mm_per_px),
        top_depth_cm=top_depth_cm,
        core_top_inset_cm=section.core_top_inset_cm,
        core_bottom_inset_cm=section.core_bottom_inset_cm,
        bar_width_px=bar_width_px,
    )
    out = Image.new("RGB", (out_w + bar_width_px, out_h), (20, 20, 22))
    out.paste(img, (0, 0))
    out.paste(bar, (out_w, 0))
    layout = ComposeLayout(img_w=out_w, img_h=out_h, bar_width=bar_width_px)
    return out, layout


def title_strip(text: str, width: int, height: int = 24) -> Image.Image:
    strip = Image.new("RGB", (width, height), (40, 42, 48))
    draw = ImageDraw.Draw(strip)
    draw.text((6, 5), text, fill=(230, 230, 235), font=_font(11))
    return strip
