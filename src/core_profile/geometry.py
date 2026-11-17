"""Where the core is, slice by slice: centre, outer radius, and the inner wall of the liner.

The engine works on a *core-aligned axial volume*: ``data[i]`` is the i-th axial slice (along the core, top
first), values are whatever the scanner calibration gives (HU, or a CT-specific unit). Nothing here assumes the
unit except ``core_thr``, the value that separates "core or liner" from the air around it.

Method (plain language): smooth each slice a little, keep everything above ``core_thr``, fill holes (a void
inside the core must not shrink the circle), keep the biggest blob, and take its centre of mass and the radius
of a disc with the same area. The inner wall of the liner is where the median radial profile of the whole
section changes fastest, a few millimetres inside the outer edge; the sediment radius then follows the outer
radius slice by slice, so a core that wobbles in the scanner is followed.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage as ndi


@dataclass
class Geometry:
    """Per-slice centre (row, col, pixels), outer radius and sediment (inner) radius in pixels."""

    center: np.ndarray  # (n, 2)
    r_out: np.ndarray  # (n,)
    r_in: np.ndarray  # (n,)
    voxel_mm: float
    wall_px: float  # outer radius minus inner radius of the liner (median)

    def radius(self, i: int, shape: tuple[int, int]) -> np.ndarray:
        """Distance (pixels) of every pixel of slice ``i`` from the core axis."""
        yy, xx = np.indices(shape, dtype=np.float32)
        return np.hypot(yy - self.center[i, 0], xx - self.center[i, 1])

    def inside(self, i: int, shape: tuple[int, int], shrink_px: float = 0.0) -> np.ndarray:
        return self.radius(i, shape) < (self.r_in[i] - shrink_px)


def _slice_disc(sl: np.ndarray, core_thr: float) -> tuple[float, float, float]:
    """Centre (row, col) and equal-area radius of the core+liner disc in one slice; NaN if none."""
    m = ndi.gaussian_filter(sl, 1.5) > core_thr
    m = ndi.binary_fill_holes(m)
    lab, k = ndi.label(m)
    if k == 0:
        return np.nan, np.nan, np.nan
    sz = np.bincount(lab.ravel())
    sz[0] = 0
    big = lab == int(np.argmax(sz))
    cy, cx = ndi.center_of_mass(big)
    return float(cy), float(cx), float(np.sqrt(big.sum() / np.pi))


def _fill_nan(a: np.ndarray) -> np.ndarray:
    ok = np.isfinite(a)
    if not ok.any():
        raise ValueError("no slice contains a core above core_thr")
    idx = np.arange(len(a))
    return np.where(ok, a, np.interp(idx, idx[ok], a[ok]))


def radial_profile(sl: np.ndarray, cy: float, cx: float, rmax: int) -> np.ndarray:
    yy, xx = np.indices(sl.shape)
    r = np.hypot(yy - cy, xx - cx).astype(int)
    m = r < rmax
    s = np.bincount(r[m], sl[m], rmax)
    c = np.bincount(r[m], minlength=rmax)
    return s / np.maximum(c, 1)


def fit_geometry(
    data: np.ndarray,
    voxel_mm: float,
    core_thr: float,
    r_in_mm: float | None = None,
    wall_search_mm: tuple[float, float] = (3.4, 12.5),
    margin_mm: float = 1.1,
    smooth_slices: int = 25,
) -> Geometry:
    """Fit centre / outer radius per slice and the inner liner wall.

    ``r_in_mm``: if given, the sediment radius is fixed to it (median), skipping the wall search.
    ``wall_search_mm``: where, inside the outer edge, to look for the inner wall of the liner.
    ``margin_mm``: safety margin taken off the sediment radius (partial-volume at the wall).
    """
    n = len(data)
    fit = np.array([_slice_disc(data[i], core_thr) for i in range(n)])
    cy, cx, ro = (_fill_nan(fit[:, k]) for k in range(3))
    sm = lambda a: ndi.median_filter(a, smooth_slices, mode="nearest")
    cy, cx, ro = sm(cy), sm(cx), sm(ro)
    ro_med = float(np.median(ro))
    if r_in_mm is not None:
        wall = ro_med - r_in_mm / voxel_mm
    else:
        pick = np.arange(0, n, max(1, n // 60))
        prof = np.nanmedian([radial_profile(data[i], cy[i], cx[i], int(ro_med) + 3) for i in pick], axis=0)
        # radial_profile is relative to each slice's own centre, so align on the outer edge of the median disc
        a = int(ro_med - wall_search_mm[1] / voxel_mm)
        b = int(ro_med - wall_search_mm[0] / voxel_mm)
        # a bin k holds radii [k, k+1), so its centre is k+0.5; the wall is the centroid of the steepest step
        g = np.abs(np.diff(ndi.gaussian_filter1d(prof, 0.7)))  # g[k] is the step between bin centres k+0.5 and k+1.5
        k0 = a + int(np.argmax(g[a:b]))
        w = g[max(k0 - 2, 0) : k0 + 3]
        r0 = float(np.sum((np.arange(max(k0 - 2, 0), k0 + 3) + 1.0) * w) / w.sum())
        wall = ro_med - r0
    r_in = ro - wall - margin_mm / voxel_mm
    return Geometry(np.c_[cy, cx], ro, r_in, voxel_mm, float(wall))
