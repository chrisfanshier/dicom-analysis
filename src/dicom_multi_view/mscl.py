"""Parse Geotek MSCL ``.out`` files and render depth-aligned plot strips.

Depth convention
----------------
The multi-view viewer labels depth in **cm** (``top_depth_cm``, scale bars).
MSCL ``SB DEPTH`` is detected from the units row:

* **cm** (newer Geotek exports) — used as-is
* **m** / **meters** (older exports) — multiplied by 100

Calibration rows (``SECT NUM`` containing ``cal``, e.g. ``cal sect``,
``before cal``) are dropped and never plotted.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Columns that are structural, not plottable channels by default.
_META_NAMES = {
    "sb depth",
    "sbdepth",
    "sect num",
    "sectnum",
    "sect depth",
    "sectdepth",
}

_DEPTH_ALIASES = ("sb depth", "sbdepth", "sb_depth", "depth sb", "depth")


def _font(size: int = 11) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("arial.ttf", size)
    except OSError:
        try:
            return ImageFont.truetype("DejaVuSans.ttf", size)
        except OSError:
            return ImageFont.load_default()


def _norm_name(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip().lower())


def _split_row(line: str) -> list[str]:
    """Split a Geotek .out row (tab-preferred; fall back to 2+ spaces)."""
    if "\t" in line:
        return [c.strip() for c in line.split("\t")]
    # Fixed-ish width / space-separated
    return [c for c in re.split(r"\s{2,}", line.strip()) if c != ""]


def _is_cal_section(sect_num: str) -> bool:
    return "cal" in (sect_num or "").lower()


def _parse_float(tok: str) -> float:
    tok = (tok or "").strip()
    if tok == "" or tok.lower() in {"nan", "na", "n/a", "-", "--"}:
        return float("nan")
    try:
        return float(tok)
    except ValueError:
        return float("nan")


def _looks_like_header(parts: list[str]) -> bool:
    norms = [_norm_name(p) for p in parts if p.strip()]
    if not norms:
        return False
    joined = " ".join(norms)
    if any(a in norms or a.replace(" ", "") in {n.replace(" ", "") for n in norms} for a in _DEPTH_ALIASES):
        return True
    return "sb depth" in joined or "sbdepth" in joined.replace(" ", "")


def _find_depth_col(names: list[str]) -> int:
    norms = [_norm_name(n) for n in names]
    for i, n in enumerate(norms):
        compact = n.replace(" ", "")
        if n in _DEPTH_ALIASES or compact in {"sbdepth", "sb_depth"}:
            return i
        if n.startswith("sb") and "depth" in n:
            return i
    for i, n in enumerate(norms):
        if n == "depth" or n.endswith(" depth"):
            return i
    raise ValueError("Could not find SB Depth column in MSCL header")


def _find_sect_num_col(names: list[str]) -> int | None:
    norms = [_norm_name(n) for n in names]
    for i, n in enumerate(norms):
        compact = n.replace(" ", "")
        if n in {"sect num", "sectnum", "section", "section num", "section number"}:
            return i
        if compact in {"sectnum", "sectionnum"}:
            return i
    return None


def _depth_unit_to_cm_factor(unit_cell: str) -> tuple[float, str]:
    u = _norm_name(unit_cell)
    if not u:
        return 1.0, "cm"  # assume cm if blank beside SB DEPTH (common in mixed rows)
    if u in {"m", "meter", "meters", "metre", "metres"}:
        return 100.0, "m"
    if u in {"cm", "centimeter", "centimeters", "centimetre", "centimetres"}:
        return 1.0, "cm"
    if u in {"mm", "millimeter", "millimeters"}:
        return 0.1, "mm"
    # Unknown — leave as-is but report the token
    return 1.0, u or "unknown"


@dataclass
class MsclColumn:
    name: str
    unit: str = ""

    @property
    def label(self) -> str:
        return f"{self.name} ({self.unit})" if self.unit else self.name


@dataclass
class MsclDataset:
    """Parsed MSCL table with depth in cm and calibration rows removed."""

    path: Path
    columns: list[MsclColumn]
    depth_cm: np.ndarray
    section_num: list[str]
    data: dict[str, np.ndarray]
    depth_unit_original: str
    header_style: str  # "legacy" | "modern"
    n_cal_filtered: int = 0
    notes: list[str] = field(default_factory=list)

    @property
    def channel_names(self) -> list[str]:
        """Plottable numeric channel names (excludes depth / section meta)."""
        out = []
        for col in self.columns:
            if _norm_name(col.name) in _META_NAMES:
                continue
            if col.name in self.data:
                out.append(col.name)
        return out

    def series_values(self, name: str) -> np.ndarray:
        return self.data[name]


@dataclass
class MsclPlotSpec:
    """User choices for one plotted channel."""

    name: str
    enabled: bool = True
    width_px: int = 72
    xmin: float | None = None  # None → auto from data in view
    xmax: float | None = None


def parse_mscl_out(path: Path | str) -> MsclDataset:
    """
    Load a Geotek MSCL ``.out`` file (legacy or modern header).

    Handles optional ``Processing Parameters`` preamble, tab- or
    space-delimited columns, and SB DEPTH units of m or cm.
    """
    path = Path(path)
    raw = path.read_text(encoding="utf-8", errors="replace")
    lines = raw.splitlines()

    header_idx: int | None = None
    header_parts: list[str] = []
    for i, line in enumerate(lines):
        if not line.strip():
            continue
        parts = _split_row(line)
        if _looks_like_header(parts):
            header_idx = i
            header_parts = parts
            break
    if header_idx is None:
        raise ValueError(f"No MSCL column header (SB DEPTH) found in {path.name}")

    # Drop trailing empty header cells
    while header_parts and header_parts[-1] == "":
        header_parts.pop()
    names = [p.strip() for p in header_parts]
    n_cols = len(names)

    # Units row: first non-empty line after header that is not clearly data
    units = [""] * n_cols
    data_start = header_idx + 1
    style = "legacy"
    for j in range(header_idx + 1, min(header_idx + 4, len(lines))):
        line = lines[j]
        if not line.strip():
            data_start = j + 1
            continue
        parts = _split_row(line)
        # Units row: mostly non-numeric tokens / unit strings
        numericish = sum(1 for p in parts[: min(3, len(parts))] if re.fullmatch(r"[-+]?\d+(\.\d+)?", p.strip() or ""))
        if numericish >= 1 and any("cal" in p.lower() or re.fullmatch(r"\d+", p.strip() or "") for p in parts[1:3] if p):
            # Actual data row
            data_start = j
            break
        # Treat as units
        for k in range(min(n_cols, len(parts))):
            units[k] = parts[k].strip()
        data_start = j + 1
        # Modern files often have a blank line after units
        if j + 1 < len(lines) and not lines[j + 1].strip():
            data_start = j + 2
        break

    # Preamble with "Processing Parameters" ⇒ modern-ish export
    preamble = "\n".join(lines[:header_idx]).lower()
    if "processing parameters" in preamble or any(
        _norm_name(u) == "cm" for u in units[:1]
    ):
        style = "modern"
    elif units and _norm_name(units[0]) in {"m", "meter", "meters"}:
        style = "legacy"

    depth_i = _find_depth_col(names)
    sect_i = _find_sect_num_col(names)
    factor, depth_unit = _depth_unit_to_cm_factor(units[depth_i] if depth_i < len(units) else "")

    # If units row missing for depth but values look like meters (typical |d|<50 for whole cores in m),
    # we still trust the units cell; if blank, probe first data depths later.

    columns = [MsclColumn(name=names[k], unit=units[k] if k < len(units) else "") for k in range(n_cols)]

    depth_raw: list[float] = []
    sect_nums: list[str] = []
    col_vals: dict[str, list[float]] = {c.name: [] for c in columns if _norm_name(c.name) not in _META_NAMES}
    # Also keep SECT DEPTH if present as optional channel? Skip — meta only.
    n_cal = 0
    n_bad = 0

    for line in lines[data_start:]:
        if not line.strip():
            continue
        parts = _split_row(line)
        if not parts:
            continue
        # Pad / trim
        if len(parts) < n_cols:
            parts = parts + [""] * (n_cols - len(parts))
        elif len(parts) > n_cols:
            parts = parts[:n_cols]

        # Skip stray header repeats
        if _looks_like_header(parts):
            continue

        d = _parse_float(parts[depth_i])
        if np.isnan(d):
            n_bad += 1
            continue

        sn = parts[sect_i].strip() if sect_i is not None and sect_i < len(parts) else ""
        if _is_cal_section(sn):
            n_cal += 1
            continue

        depth_raw.append(d)
        sect_nums.append(sn)
        for k, col in enumerate(columns):
            if col.name not in col_vals:
                continue
            col_vals[col.name].append(_parse_float(parts[k]))

    if not depth_raw:
        raise ValueError(f"No non-calibration data rows in {path.name}")

    depth_arr = np.asarray(depth_raw, dtype=np.float64)

    # If unit was blank/unknown, infer m vs cm from magnitude
    notes: list[str] = []
    if depth_unit in {"unknown", ""} or (factor == 1.0 and not units[depth_i].strip()):
        med = float(np.nanmedian(np.abs(depth_arr)))
        if med < 40.0 and float(np.nanmax(depth_arr)) < 80.0:
            # Likely meters for a multi-meter core logged as m
            # But modern short cores in cm also have med < 40...
            # Prefer: if header said nothing and max < 30 and file looks legacy (has ST col / Geotek 7.0)
            first = lines[0].lower() if lines else ""
            if "mscl 7.0" in first or "mscl 6" in first or any(
                _norm_name(c.name) == "st" for c in columns
            ):
                factor, depth_unit = 100.0, "m (inferred)"
                notes.append("Inferred SB DEPTH unit = m from legacy-style file")
            else:
                depth_unit = depth_unit or "cm"
                notes.append("Assumed SB DEPTH unit = cm (blank units cell)")
        else:
            depth_unit = depth_unit or "cm"

    depth_cm = depth_arr * factor
    data = {k: np.asarray(v, dtype=np.float64) for k, v in col_vals.items()}

    # Drop channels that are entirely NaN
    data = {k: v for k, v in data.items() if np.any(np.isfinite(v))}
    columns = [c for c in columns if _norm_name(c.name) in _META_NAMES or c.name in data]

    notes.append(f"SB DEPTH unit: {depth_unit} -> cm (x{factor:g})")
    if n_cal:
        notes.append(f"Filtered {n_cal} calibration row(s)")

    return MsclDataset(
        path=path,
        columns=columns,
        depth_cm=depth_cm,
        section_num=sect_nums,
        data=data,
        depth_unit_original=depth_unit,
        header_style=style,
        n_cal_filtered=n_cal,
        notes=notes,
    )


def depth_at_image_y(
    y_px: float,
    *,
    top_depth_cm: float,
    core_top_inset_cm: float,
    mm_per_px: float,
) -> float:
    """Map image-row y (0 = top of section image) to depth cm."""
    depth_cm_per_px = max(float(mm_per_px), 1e-12) / 10.0
    return float(top_depth_cm - core_top_inset_cm + y_px * depth_cm_per_px)


def image_y_at_depth(
    depth_cm: float,
    *,
    top_depth_cm: float,
    core_top_inset_cm: float,
    mm_per_px: float,
) -> float:
    depth_cm_per_px = max(float(mm_per_px), 1e-12) / 10.0
    return (depth_cm - (top_depth_cm - core_top_inset_cm)) / depth_cm_per_px


def render_mscl_strip(
    dataset: MsclDataset,
    spec: MsclPlotSpec,
    *,
    height_px: int,
    top_depth_cm: float,
    core_top_inset_cm: float,
    depth_extent_cm: float,
    mm_per_px: float,
    background: tuple[int, int, int] = (24, 26, 30),
) -> Image.Image:
    """
    Draw one vertical MSCL channel strip aligned to a section image height.

    Depth increases downward. X-range uses ``spec.xmin/xmax`` or auto from
    finite samples that fall inside the image depth span.
    """
    w = max(24, int(spec.width_px))
    h = max(1, int(height_px))
    img = Image.new("RGB", (w, h), background)
    draw = ImageDraw.Draw(img)

    d0 = top_depth_cm - core_top_inset_cm
    d1 = d0 + depth_extent_cm

    depths = dataset.depth_cm
    values = dataset.series_values(spec.name)
    mask = np.isfinite(depths) & np.isfinite(values) & (depths >= d0 - 1e-6) & (depths <= d1 + 1e-6)
    if not np.any(mask):
        draw.text((4, 4), spec.name, fill=(140, 140, 150), font=_font(10))
        draw.text((4, 18), "no data", fill=(100, 100, 110), font=_font(9))
        return img

    yd = depths[mask]
    yv = values[mask]

    xmin = spec.xmin
    xmax = spec.xmax
    if xmin is None or xmax is None:
        lo = float(np.nanpercentile(yv, 2))
        hi = float(np.nanpercentile(yv, 98))
        if not np.isfinite(lo) or not np.isfinite(hi) or abs(hi - lo) < 1e-12:
            lo = float(np.nanmin(yv))
            hi = float(np.nanmax(yv))
        if abs(hi - lo) < 1e-12:
            lo -= 1.0
            hi += 1.0
        pad = 0.05 * (hi - lo)
        if xmin is None:
            xmin = lo - pad
        if xmax is None:
            xmax = hi + pad
    if xmax <= xmin:
        xmax = xmin + 1.0

    # Plot area insets for label
    pad_l, pad_r, pad_t, pad_b = 2, 2, 16, 2
    plot_w = max(1, w - pad_l - pad_r)

    def x_of(v: float) -> float:
        t = (v - xmin) / (xmax - xmin)
        return pad_l + t * plot_w

    def y_of(depth: float) -> float:
        y = image_y_at_depth(
            depth,
            top_depth_cm=top_depth_cm,
            core_top_inset_cm=core_top_inset_cm,
            mm_per_px=mm_per_px,
        )
        return float(y)

    # Frame
    draw.rectangle([pad_l, pad_t, w - pad_r - 1, h - pad_b - 1], outline=(70, 74, 82))
    # Zero / mid guide
    mid_x = x_of(0.0) if xmin < 0 < xmax else None
    if mid_x is not None and pad_l <= mid_x <= w - pad_r:
        draw.line([(mid_x, pad_t), (mid_x, h - pad_b)], fill=(50, 55, 62), width=1)

    # Polyline (skip gaps where consecutive depths jump a lot)
    pts: list[tuple[float, float]] = []
    prev_d: float | None = None
    max_gap = max(2.0, 0.05 * (d1 - d0))  # cm

    def flush() -> None:
        nonlocal pts
        if len(pts) >= 2:
            draw.line(pts, fill=(120, 200, 230), width=1)
        elif len(pts) == 1:
            x, y = pts[0]
            draw.ellipse([x - 1, y - 1, x + 1, y + 1], fill=(120, 200, 230))
        pts = []

    order = np.argsort(yd)
    for i in order:
        d = float(yd[i])
        v = float(yv[i])
        if prev_d is not None and abs(d - prev_d) > max_gap:
            flush()
        x = x_of(v)
        y = y_of(d)
        if y < -2 or y > h + 2:
            flush()
            prev_d = d
            continue
        pts.append((x, y))
        prev_d = d
    flush()

    # Title + xlim hint
    title = spec.name
    draw.text((3, 2), title[:18], fill=(210, 210, 220), font=_font(10))
    rng = f"{xmin:g}…{xmax:g}"
    draw.text((3, h - 12), rng[:16], fill=(110, 110, 120), font=_font(8))
    return img


def render_mscl_strips_for_section(
    dataset: MsclDataset,
    specs: list[MsclPlotSpec],
    *,
    height_px: int,
    top_depth_cm: float,
    core_top_inset_cm: float,
    depth_extent_cm: float,
    mm_per_px: float,
    gap: int = 4,
) -> Image.Image | None:
    """Concatenate enabled strips horizontally; ``None`` if nothing enabled."""
    enabled = [s for s in specs if s.enabled and s.name in dataset.data]
    if not enabled:
        return None
    strips = [
        render_mscl_strip(
            dataset,
            spec,
            height_px=height_px,
            top_depth_cm=top_depth_cm,
            core_top_inset_cm=core_top_inset_cm,
            depth_extent_cm=depth_extent_cm,
            mm_per_px=mm_per_px,
        )
        for spec in enabled
    ]
    total_w = sum(s.width for s in strips) + gap * max(len(strips) - 1, 0)
    out = Image.new("RGB", (total_w, height_px), (20, 20, 22))
    x = 0
    for s in strips:
        out.paste(s, (x, 0))
        x += s.width + gap
    return out
