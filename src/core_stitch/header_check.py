"""Does the scanner (DICOM header) placement of TOP and BOTTOM look right?

Uses one middle slice per half, so it takes well under a second.  Three checks, in physical units:

* across : the two core centres should land at the same X
* depth  : the two stacks should start at the same depth
* along  : the headers imply an overlap; the two core profiles (mean HU down the core) should
           agree there, and no other slide within +/-150 mm should fit clearly better

The same logic ran over the whole 2026-11 release (tools/survey_headers_fast.py).
"""

from __future__ import annotations

import numpy as np

from core_stitch.itk_stitch import orientation_difference_deg
from core_stitch.segment import mid_slice, segment_core

ACROSS_TOL_MM = 5.0
DEPTH_TOL_MM = 5.0
MIN_OVERLAP_MM = 10.0


def _profile(h):
    img = mid_slice(h.series.volume)
    m = segment_core(img).mask
    _, xs = np.nonzero(m)
    origin = np.asarray(h.image.GetOrigin())
    col_pitch, row_pitch = h.spacing_xyz[0], h.spacing_xyz[1]
    # where the core's centre column sits on this half's first slice plane (3-D, mm).  Measuring across and depth
    # there, and not at the image corner, keeps halves cut in slightly different planes comparable.
    across = origin + float(xs.mean() if xs.size else 0.0) * col_pitch * h.row_dir
    prof = np.full(m.shape[0], np.nan)
    for r in np.where(m.any(axis=1))[0]:
        prof[r] = img[r, m[r]].mean()
    along = float(np.dot(origin, h.col_dir)) + np.arange(m.shape[0]) * row_pitch
    return across, along, prof


def _ncc(a, b):
    k = np.isfinite(a) & np.isfinite(b)
    if k.sum() < 15:
        return float("nan")
    a, b = a[k], b[k]
    a = (a - a.mean()) / (a.std() + 1e-6)
    b = (b - b.mean()) / (b.std() + 1e-6)
    return float(np.dot(a, b) / len(a))


def _shifted(x, sh):
    out = np.full_like(x, np.nan)
    if sh >= 0:
        out[sh:] = x[: len(x) - sh] if sh else x
    else:
        out[:sh] = x[-sh:]
    return out


def header_check(top, bottom) -> dict:
    """Return the measurements and a verdict: ``headers_good`` or ``needs_image_stitch``."""
    xa, za, pa = _profile(top)
    xb, zb, pb = _profile(bottom)
    out = {
        "orientations_equal": bool(np.allclose(np.r_[top.row_dir, top.col_dir], np.r_[bottom.row_dir, bottom.col_dir], atol=1e-3)),
        "across_diff_mm": round(float(np.dot(xb - xa, top.row_dir)), 2),
        "depth_diff_mm": round(float(np.dot(xb - xa, top.normal)), 2),
    }
    g = np.arange(min(za.min(), zb.min()), max(za.max(), zb.max()), 1.0)

    def on_grid(z, p):
        f = np.interp(g, z, np.nan_to_num(p, nan=-1e9))
        f[np.interp(g, z, np.isfinite(p).astype(float)) < 0.99] = np.nan
        f[(g < z.min()) | (g > z.max())] = np.nan  # np.interp clamps outside the image
        return f

    fa, fb = on_grid(za, pa), on_grid(zb, pb)
    both = np.isfinite(fa) & np.isfinite(fb)
    out["overlap_mm"] = int(both.sum())
    at_header = _ncc(fa, fb) if both.sum() >= 15 else float("nan")
    out["ncc_at_header"] = None if not np.isfinite(at_header) else round(at_header, 3)
    best, best_shift = float("nan"), None
    for sh in range(-150, 151):
        v = _ncc(fa, _shifted(fb, sh))
        if np.isfinite(v) and (best_shift is None or v > best):
            best, best_shift = v, sh
    out["best_shift_mm"] = best_shift
    out["ncc_best"] = None if best_shift is None else round(best, 3)

    reasons = []
    out["orientation_diff_deg"] = None if out["orientations_equal"] else round(orientation_difference_deg(top, bottom), 2)
    notes = []
    if not out["orientations_equal"]:  # information, not a reason to doubt the placement: the headers describe both planes
        notes.append(
            f"The scanner cut the two halves in planes {out['orientation_diff_deg']:.1f} degrees apart; the coarser half is "
            "resampled into the other's orientation."
        )
    out["notes"] = notes
    if abs(out["across_diff_mm"]) > ACROSS_TOL_MM:
        reasons.append(f"core centres are {out['across_diff_mm']:+.0f} mm apart sideways")
    if abs(out["depth_diff_mm"]) > DEPTH_TOL_MM:
        reasons.append(f"the stacks are offset {out['depth_diff_mm']:+.0f} mm in depth")
    if out["overlap_mm"] < MIN_OVERLAP_MM:
        reasons.append("the headers imply no overlap")
    elif best_shift is not None and abs(best_shift) > 10 and best - (out["ncc_at_header"] if out["ncc_at_header"] is not None else -1) > 0.1:
        reasons.append(f"the core profiles fit better at {best_shift:+d} mm")
    out["verdict"] = "needs_image_stitch" if reasons else "headers_good"
    out["reasons"] = reasons
    return out
