"""Write a down-core profile as a CSV with a commented header that records every choice and the input file.

The header lines start with ``#`` (``pandas.read_csv(path, comment="#")`` and most spreadsheet imports skip them).
They are human-readable ``key: value`` lines, followed by one ``# settings_json:`` line holding the same settings as
JSON so a run can be repeated by a program. Then a plain header row and one line per row (depth step) of the image
between the core-end guides.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

COLUMNS = ("depth_from_image_top_mm", "depth_from_core_top_cm", "mean_hu", "sd_hu", "n_voxels")
AXIS_COLUMNS = ("axis_x_mm", "axis_y_mm")


def _fmt(v: float, nd: int) -> str:
    return "" if v is None or not math.isfinite(float(v)) else f"{float(v):.{nd}f}"


def _flat(meta: dict, prefix: str = "") -> list[tuple[str, object]]:
    out: list[tuple[str, object]] = []
    for k, v in meta.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.extend(_flat(v, key + "."))
        else:
            out.append((key, v))
    return out


def _text(v: object) -> str:
    if isinstance(v, bool):
        return "yes" if v else "no"
    if v is None:
        return "-"
    if isinstance(v, float):
        return f"{v:.4g}"
    return str(v).replace("\n", " ")


def write_profile_csv(
    path: str | Path,
    mean: np.ndarray,
    sd: np.ndarray,
    cnt: np.ndarray,
    row_mm: float,
    top_mm: float,
    bottom_mm: float,
    meta: dict,
    axis: tuple[np.ndarray, np.ndarray] | None = None,
) -> int:
    """Write the profile for the rows between ``top_mm`` and ``bottom_mm`` (depth from the top of the image).

    ``mean``, ``sd`` and ``cnt`` have one value per image row; row i is centred at (i + 0.5) * row_mm. ``axis`` =
    (x, y) of the fitted cylinder axis per row, written as two extra columns. Returns the number of data rows.
    """
    depth = (np.arange(len(mean)) + 0.5) * row_mm
    keep = np.flatnonzero((depth >= top_mm) & (depth <= bottom_mm))
    cols = COLUMNS + (AXIS_COLUMNS if axis is not None else ())
    lines = ["# core-profile down-core profile"]
    lines += [f"# {k}: {_text(v)}" for k, v in _flat(meta)]
    lines.append("# columns: depth_from_image_top_mm = row centre measured from the top of the image; depth_from_core_top_cm = same, from the top core-end guide")
    lines.append("# columns: mean_hu / sd_hu / n_voxels = mean, standard deviation and count of the masked voxels in that row over every coronal slice (blank where there were too few)")
    if axis is not None:
        lines.append("# columns: axis_x_mm / axis_y_mm = where the cylinder axis was at that row (across the slice / along the stack)")
    lines.append("# settings_json: " + json.dumps(meta, default=str, sort_keys=False))
    lines.append(",".join(cols))
    for i in keep:
        row = [_fmt(depth[i], 3), _fmt((depth[i] - top_mm) / 10.0, 4), _fmt(mean[i], 2), _fmt(sd[i], 2), str(int(cnt[i]))]
        if axis is not None:
            row += [_fmt(axis[0][i], 3), _fmt(axis[1][i], 3)]
        lines.append(",".join(row))
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    return int(len(keep))
