"""Core scale block for the JSON sidecar: where the core ends, in pixels and cm.

The exported JPEG/TIFF stay clean (no burned-in ruler). The sidecar carries the numbers
needed to build a ruler or measure along the core later. Field names follow
``dicom-multi-view`` (``core_top_inset_cm``, ``core_bottom_inset_cm``, ``core_length_cm``).

Positions are pixel-edge coordinates: row 0 is the top edge of the first pixel row and
row ``image_rows`` the bottom edge of the last, so bottom - top is exactly the core length.

Conventions: the exported image is upright with the core running down the rows; row 0
is the top (shallow) end. Insets are measured in from the image ends, along the rows.
"""

from __future__ import annotations

import numpy as np

SCHEMA = "core_scale/1"


def auto_bounds(
    mask: np.ndarray, box: tuple[int, int, int, int], mm_per_px: float
) -> tuple[float, float]:
    """Insets (cm) from the exported image's top/bottom edges to the first/last core row."""
    r0, r1, _c0, _c1 = box
    rows = np.where(mask.any(axis=1))[0]
    if rows.size == 0:
        return 0.0, 0.0
    top_px = max(0, int(rows[0]) - r0)
    bot_px = max(0, (r1 - 1) - int(rows[-1]))
    return top_px * mm_per_px / 10.0, bot_px * mm_per_px / 10.0


def clamp_insets(top_cm: float, bottom_cm: float, extent_cm: float) -> tuple[float, float]:
    top = float(np.clip(top_cm, 0.0, max(extent_cm, 0.0)))
    bot = float(np.clip(bottom_cm, 0.0, max(extent_cm - top - 0.01, 0.0)))
    return top, bot


def length_check(measured_cm: float, expected_cm: float | None) -> dict:
    """Record how the measured length compares with the master list. No threshold is applied:
    ``length_check`` is ``compared`` (numbers recorded) or ``no_reference``."""
    if not expected_cm or expected_cm <= 0:
        return {
            "expected_length_cm": None,
            "length_difference_cm": None,
            "length_difference_pct": None,
            "length_check": "no_reference",
        }
    diff = measured_cm - expected_cm
    return {
        "expected_length_cm": round(float(expected_cm), 3),
        "length_difference_cm": round(float(diff), 3),
        "length_difference_pct": round(100.0 * diff / expected_cm, 2),
        "length_check": "compared",
    }


def dpi_for(mm_per_px: float) -> float:
    return 25.4 / max(float(mm_per_px), 1e-9)


def core_scale_block(
    shape_hw: tuple[int, int],
    mm_per_px: float,
    top_inset_cm: float,
    bottom_inset_cm: float,
    *,
    source: str,
    expected_length_cm: float | None = None,
    center_col_px: float | None = None,
) -> dict:
    h, w = int(shape_hw[0]), int(shape_hw[1])
    extent_cm = h * mm_per_px / 10.0
    top_cm, bot_cm = clamp_insets(top_inset_cm, bottom_inset_cm, extent_cm)
    length_cm = max(extent_cm - top_cm - bot_cm, 0.0)
    px_per_cm = 10.0 / mm_per_px
    top_row = top_cm * px_per_cm
    bot_row = h - bot_cm * px_per_cm
    col = w / 2.0 if center_col_px is None else float(center_col_px)
    block = {
        "schema": SCHEMA,
        "mm_per_px": float(mm_per_px),
        "px_per_cm": px_per_cm,
        "depth_axis": "rows",
        "row_zero_is": "core_top",
        "image_rows": h,
        "image_cols": w,
        "image_depth_extent_cm": round(extent_cm, 4),
        "core_top_inset_cm": round(top_cm, 4),
        "core_bottom_inset_cm": round(bot_cm, 4),
        "core_length_cm": round(length_cm, 4),
        "core_top_row_px": round(top_row, 2),
        "core_bottom_row_px": round(bot_row, 2),
        "core_center_col_px": round(col, 2),
        "endpoints_px": {
            "top": [round(top_row, 2), round(col, 2)],
            "bottom": [round(bot_row, 2), round(col, 2)],
        },
        "bounds_source": source,
    }
    block.update(length_check(length_cm, expected_length_cm))
    return block
