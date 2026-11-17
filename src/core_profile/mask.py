"""Sediment mask from the picture itself: no round-core assumption, so whole-round and split cores work alike.

How: in the HU values, air is lowest, the liner (PVC) is brighter, sediment is brighter again. Two Otsu steps find the
two thresholds: air | everything else, then liner | sediment among the non-air values. Per coronal slice the sediment
is the pixels at or above the second threshold, the biggest blob, holes optionally filled (air pockets inside the
sediment), shrunk by a margin (partial-volume pixels at the wall and surface).

An optional cylinder (circle in the axial view, running along the core) removes the liner, so then the threshold
only has to separate air from sediment. The circle is fitted to the liner's outer edge and shrunk by the liner
thickness; the mask is the threshold AND the cylinder.

Limits: if the liner reads brighter than low-density sediment, one threshold cannot separate them; the threshold is
a number the user can move. Edge pixels of a split core's uneven surface are as reliable as the scan's resolution.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi
from skimage.filters import threshold_otsu


def auto_thresholds(volume: np.ndarray, max_samples: int = 2_000_000) -> tuple[float, float]:
    """(air | rest, liner | sediment) from a sample of the whole volume."""
    flat = volume.reshape(-1)
    step = max(1, flat.size // max_samples)
    v = flat[::step]
    v = v[np.isfinite(v)]
    t_air = float(threshold_otsu(v))
    rest = v[v >= t_air]
    t_sed = float(threshold_otsu(rest)) if rest.size > 100 and rest.min() < rest.max() else t_air
    return t_air, t_sed


MIN_BLOB_PX = 25  # smallest patch kept when a cylinder is used (about 6 mm2 at 0.5 mm pixels)


def circle_spans(k: int, circle: tuple, dz_mm: float, dc_mm: float, n_cols: int) -> tuple[np.ndarray, np.ndarray]:
    """Columns [c0, c1) of coronal slice ``k`` inside the circle ``(cx, cy, r)``; each of cx, cy is a number or one value
    per row (a core whose axis wobbles). Units mm; cx across the slice, cy along the stack, slice k at (k + 0.5) * dz.
    The cylinder runs along the rows. Rows where the slice misses the circle get c0 = c1 = 0."""
    cx, cy, r = circle
    cx, cy = np.asarray(cx, float), np.asarray(cy, float)
    dy = (k + 0.5) * dz_mm - cy
    half = np.sqrt(np.maximum(r * r - dy * dy, 0.0))
    c0 = np.clip(np.ceil((cx - half) / dc_mm - 0.5), 0, n_cols).astype(int)
    c1 = np.clip(np.floor((cx + half) / dc_mm - 0.5) + 1, 0, n_cols).astype(int)
    ok = (np.abs(dy) < r) & (c1 > c0)
    return np.where(ok, c0, 0), np.where(ok, c1, 0)


def circle_span(k: int, circle: tuple, dz_mm: float, dc_mm: float, n_cols: int) -> tuple[int, int] | None:
    """``circle_spans`` for one fixed circle: (c0, c1), or None when the slice misses it."""
    c0, c1 = circle_spans(k, circle, dz_mm, dc_mm, n_cols)
    return (int(c0), int(c1)) if int(c1) > int(c0) else None


def mask_2d(img: np.ndarray, t_sed: float, margin_px: int = 1, fill_holes: bool = True, span: tuple | None = None) -> np.ndarray:
    """Sediment of one coronal slice. ``span`` = columns kept by a cylinder (everything outside is dropped first).

    Without a cylinder only the biggest blob is kept (the liner ties the sediment together); with one, every blob of
    at least MIN_BLOB_PX pixels is kept.
    """
    m = img >= t_sed
    if span is not None:  # (c0, c1): numbers, or one value per row
        c0, c1 = np.asarray(span[0]).reshape(-1, 1), np.asarray(span[1]).reshape(-1, 1)
        cols = np.arange(img.shape[1])[None, :]
        m &= (cols >= c0) & (cols < c1)
    lab, n = ndi.label(m)
    if n == 0:
        return np.zeros(img.shape, bool)
    if n > 1:
        sizes = np.bincount(lab.ravel())[1:]
        if span is None:
            m = lab == (1 + int(np.argmax(sizes)))  # the liner joins everything into one blob; drop stray specks
        else:
            # the cylinder has cut the liner away, so patches of sediment along an uneven surface are separate blobs
            m = np.isin(lab, 1 + np.flatnonzero(sizes >= MIN_BLOB_PX))
    if fill_holes:
        m = ndi.binary_fill_holes(m)
    if margin_px > 0:
        m = ndi.binary_erosion(m, structure=ndi.generate_binary_structure(2, 1), iterations=margin_px, border_value=1)
    return m


def mask_slice(img: np.ndarray, k: int, t_sed: float, margin_mm: float, dz_mm: float, dc_mm: float, fill_holes: bool = True, circle=None) -> np.ndarray:
    """Mask of coronal slice ``k``: threshold, optionally inside the cylinder, biggest blob, holes, margin."""
    margin_px = int(round(margin_mm / dc_mm)) if margin_mm > 0 else 0
    if circle is None:
        return mask_2d(img, t_sed, margin_px, fill_holes)
    c0, c1 = circle_spans(k, circle, dz_mm, dc_mm, img.shape[1])
    if not (c1 > c0).any():
        return np.zeros(img.shape, bool)
    return mask_2d(img, t_sed, margin_px, fill_holes, (c0, c1))


def mask_3d(volume: np.ndarray, t_sed: float, margin_mm: float, pixel_mm: float, fill_holes: bool = True, circle=None, dz_mm: float = 1.0) -> np.ndarray:
    """Stack of per-coronal-slice masks, same shape as ``volume`` (slices, rows, cols)."""
    return np.stack([mask_slice(volume[k], k, t_sed, margin_mm, dz_mm, pixel_mm, fill_holes, circle) for k in range(len(volume))])


# ---------------------------------------------------------------------------------------------------------
# Cylinder: fit a circle to the outer edge of the liner (air | liner), which is the strongest edge in the scan.

def edge_points(volume: np.ndarray, row: int, t_air: float, dz_mm: float, dc_mm: float, half_rows: int = 2) -> np.ndarray:
    """(x, y) in mm of the air | core edge on both sides of every coronal slice, in the axial cut at ``row``.

    x runs across the slice (sub-pixel, interpolated at the threshold), y is the slice position (k + 0.5) * dz.
    Rows ``row - half_rows .. row + half_rows`` are averaged to calm the noise. Slices with no core, or whose core
    touches the image edge, give no point on that side.
    """
    r0, r1 = max(0, row - half_rows), min(volume.shape[1], row + half_rows + 1)
    v = np.asarray(volume[:, r0:r1, :], np.float64).mean(axis=1)  # (slices, cols)
    nz, nc = v.shape
    above = v >= t_air
    both = above[:, :-1] & above[:, 1:]  # two neighbours above: a speck of noise does not count
    has = both.any(axis=1)
    k = np.arange(nz)
    y = (k + 0.5) * dz_mm
    i = both.argmax(axis=1)  # first column of the first pair
    j = (nc - 2 - both[:, ::-1].argmax(axis=1)) + 1  # last column of the last pair
    pts = []
    sel = has & (i > 0)
    a, b = v[k[sel], i[sel] - 1], v[k[sel], i[sel]]
    f = np.where(b != a, (t_air - a) / np.where(b != a, b - a, 1.0), 0.5)
    pts.append(np.c_[(i[sel] - 1 + f + 0.5) * dc_mm, y[sel]])
    sel = has & (j < nc - 1)
    a, b = v[k[sel], j[sel]], v[k[sel], j[sel] + 1]
    f = np.where(a != b, (a - t_air) / np.where(a != b, a - b, 1.0), 0.5)
    pts.append(np.c_[(j[sel] + f + 0.5) * dc_mm, y[sel]])
    return np.vstack(pts).reshape(-1, 2)


def _kasa(p: np.ndarray) -> tuple[float, float, float] | None:
    """Least-squares circle through points (algebraic fit)."""
    A = np.c_[2 * p[:, 0], 2 * p[:, 1], np.ones(len(p))]
    b = (p**2).sum(axis=1)
    try:
        sol = np.linalg.lstsq(A, b, rcond=None)[0]
    except np.linalg.LinAlgError:
        return None
    cx, cy = float(sol[0]), float(sol[1])
    r2 = float(sol[2] + cx * cx + cy * cy)
    return (cx, cy, float(np.sqrt(r2))) if r2 > 0 else None


def ransac_circle(pts: np.ndarray, tol_mm: float = 1.5, iters: int = 500, seed: int = 0, min_inliers: int = 8) -> tuple[float, float, float, int] | None:
    """Circle that most points lie on (within ``tol_mm``), ignoring the rest. Returns (cx, cy, r, inliers) or None.

    The outliers are what makes this work for split cores: only the liner's arc is a circle, the flat cut face and
    the uneven sediment surface are not.
    """
    n = len(pts)
    if n < max(min_inliers, 3):
        return None
    rng = np.random.default_rng(seed)
    tri = np.array([rng.choice(n, 3, replace=False) for _ in range(iters)])
    a, b, c = pts[tri[:, 0]], pts[tri[:, 1]], pts[tri[:, 2]]
    d = 2 * (a[:, 0] * (b[:, 1] - c[:, 1]) + b[:, 0] * (c[:, 1] - a[:, 1]) + c[:, 0] * (a[:, 1] - b[:, 1]))
    ok = np.abs(d) > 1e-9
    a2, b2, c2 = (a**2).sum(1), (b**2).sum(1), (c**2).sum(1)
    with np.errstate(divide="ignore", invalid="ignore"):
        ux = (a2 * (b[:, 1] - c[:, 1]) + b2 * (c[:, 1] - a[:, 1]) + c2 * (a[:, 1] - b[:, 1])) / d
        uy = (a2 * (c[:, 0] - b[:, 0]) + b2 * (a[:, 0] - c[:, 0]) + c2 * (b[:, 0] - a[:, 0])) / d
    r = np.hypot(ux - a[:, 0], uy - a[:, 1])
    extent = float(np.ptp(pts, axis=0).max())
    ok &= np.isfinite(r) & (r > 0.1 * extent) & (r < 2.0 * extent)
    if not ok.any():
        return None
    ux, uy, r = ux[ok], uy[ok], r[ok]
    dist = np.hypot(pts[None, :, 0] - ux[:, None], pts[None, :, 1] - uy[:, None])
    inl = np.abs(dist - r[:, None]) < tol_mm
    best = int(np.argmax(inl.sum(1)))
    mask = inl[best]
    for _ in range(3):  # refine on the inliers, then re-select them
        fit = _kasa(pts[mask])
        if fit is None:
            return None
        cx, cy, rr = fit
        mask = np.abs(np.hypot(pts[:, 0] - cx, pts[:, 1] - cy) - rr) < tol_mm
        if mask.sum() < 3:
            return None
    if int(mask.sum()) < min_inliers:
        return None
    return cx, cy, rr, int(mask.sum())


def fit_outer_circle(volume: np.ndarray, row: int, t_air: float, dz_mm: float, dc_mm: float) -> tuple[float, float, float, int] | None:
    """Outer circle (cx, cy, r in mm, inlier count) of the core + liner in the axial cut at ``row``."""
    return ransac_circle(edge_points(volume, row, t_air, dz_mm, dc_mm))


def fit_axis_path(volume: np.ndarray, t_air: float, dz_mm: float, dr_mm: float, dc_mm: float, row0: int, row1: int,
                  step_mm: float = 10.0, smooth_mm: float = 30.0) -> tuple[np.ndarray, np.ndarray, float, int] | None:
    """Where the core's axis is at every row: (cx per row, cy per row, outer radius, rows fitted), or None.

    The outer circle is fitted at rows ``step_mm`` apart between ``row0`` and ``row1``; the radius is the median (the
    liner has one diameter), the centre is refitted with that radius fixed (steadier for a split core, where only
    an arc is visible), rows whose radius disagrees by more than 2 mm are dropped, and the centres are interpolated to
    every row and smoothed over ``smooth_mm``. Beyond the first and last fitted rows the axis is held constant.
    """
    from scipy.optimize import least_squares

    nr = volume.shape[1]
    row0, row1 = max(0, int(row0)), min(nr - 1, int(row1))
    step = max(1, int(round(step_mm / dr_mm)))
    half = max(2, int(round(2.5 / dr_mm)))
    fits = []
    for row in range(row0, row1 + 1, step):
        pts = edge_points(volume, row, t_air, dz_mm, dc_mm, half)
        res = ransac_circle(pts)
        if res is not None:
            fits.append((row, pts, res))
    if len(fits) < 3:
        return None
    r_med = float(np.median([f[2][2] for f in fits]))
    rows, cxs, cys = [], [], []
    for row, pts, (cx, cy, r, _n) in fits:
        if abs(r - r_med) > 2.0:
            continue
        near = pts[np.abs(np.hypot(pts[:, 0] - cx, pts[:, 1] - cy) - r) < 1.5]
        if len(near) >= 6:
            sol = least_squares(lambda c: np.hypot(near[:, 0] - c[0], near[:, 1] - c[1]) - r_med, (cx, cy))
            cx, cy = float(sol.x[0]), float(sol.x[1])
        rows.append(row)
        cxs.append(cx)
        cys.append(cy)
    if len(rows) < 3:
        return None
    all_rows = np.arange(nr)
    cx_r, cy_r = np.interp(all_rows, rows, cxs), np.interp(all_rows, rows, cys)
    size = max(1, int(round(smooth_mm / dr_mm)))
    if size > 1:
        cx_r = ndi.uniform_filter1d(cx_r, size, mode="nearest")
        cy_r = ndi.uniform_filter1d(cy_r, size, mode="nearest")
    return cx_r, cy_r, r_med, len(rows)


def row_profile(volume: np.ndarray, mask: np.ndarray, min_count: int = 20) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mean, SD and voxel count per row (depth) over every masked voxel of every slice. NaN where fewer than ``min_count``."""
    cnt = mask.sum(axis=(0, 2))
    s1 = (volume * mask).sum(axis=(0, 2), dtype=np.float64)
    s2 = (volume.astype(np.float64) ** 2 * mask).sum(axis=(0, 2))
    ok = cnt >= min_count
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.where(ok, s1 / np.maximum(cnt, 1), np.nan)
        var = np.where(ok, s2 / np.maximum(cnt, 1) - mean**2, np.nan)
    return mean, np.sqrt(np.maximum(var, 0)), cnt
