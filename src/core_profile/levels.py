"""Expected matrix level for every voxel: the value a plain piece of sediment would have there.

Why not a threshold: cupping and beam hardening make the same sediment read higher or lower depending on its
distance from the axis, and the level drifts along the core. A single global threshold then calls the rim "clast".

The level is a product of two pieces, each taken as a *mode* (the most common value, not the mean, so clasts
and voids cannot drag it):

* a radial **shape** (cupping): the median over a long stretch of core of each ring's mode, relative to the
  slice mean. A clast would have to fill more than half of a ~12 cm stretch to move it.
* a per-slice **scalar**: the mode of the slice after dividing by the shape, median-filtered along the core.
  A clast would have to fill more than half of the disc to move it.

Two passes: the second recomputes the scalar with first-pass clasts and voids left out.

Provenance: assembled from standard pieces (modes, median filters, MAD noise); it is not a published method.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage as ndi

from .geometry import Geometry

NB = 24  # radial rings


@dataclass
class Level:
    level: np.ndarray  # (n, h, w) expected matrix value per voxel
    scalar: np.ndarray  # (n,) per-slice level
    shape: np.ndarray  # (n, NB) radial shape (mean 1)
    sigma: np.ndarray  # (n,) noise (MAD of the below-level residual), same unit as data


def _bins(data: np.ndarray, geom: Geometry, n_bins: int = 220) -> np.ndarray:
    """Histogram bins spanning the bulk of the in-core values (1st-99th percentile, padded)."""
    pick = np.arange(0, len(data), max(1, len(data) // 40))
    v = np.concatenate([data[i][geom.inside(i, data.shape[1:])][::7] for i in pick])
    lo, hi = np.percentile(v, [0.5, 99.5])
    pad = 0.25 * (hi - lo)
    return np.linspace(lo - pad, hi + pad, n_bins + 1)


def _mode(vals: np.ndarray, bins: np.ndarray, smooth: int = 3) -> float:
    h = ndi.uniform_filter1d(np.histogram(vals, bins)[0].astype(float), smooth)
    k = int(np.argmax(h))
    return 0.5 * (bins[k] + bins[k + 1])


def interp_rings(rings: np.ndarray, r: np.ndarray, r_in: float) -> np.ndarray:
    """Linear interpolation of per-ring values (centres at (k+.5)/NB of the inner radius) to every pixel."""
    x = np.clip(r / r_in * NB - 0.5, 0, NB - 1)
    lo = x.astype(int)
    hi = np.minimum(lo + 1, NB - 1)
    f = x - lo
    return rings[lo] * (1 - f) + rings[hi] * f


def ring_modes(data: np.ndarray, geom: Geometry, bins: np.ndarray) -> np.ndarray:
    nv = len(bins) - 1
    out = np.zeros((len(data), NB), np.float32)
    for i in range(len(data)):
        r = geom.radius(i, data.shape[1:])
        m = r < geom.r_in[i]
        b = np.minimum((r[m] / geom.r_in[i] * NB).astype(int), NB - 1)
        vi = np.clip(np.digitize(data[i][m], bins) - 1, 0, nv - 1)
        h = np.bincount(b * nv + vi, minlength=NB * nv).reshape(NB, nv).astype(float)
        h = ndi.uniform_filter1d(h, 3, axis=1)
        out[i] = 0.5 * (bins[np.argmax(h, 1)] + bins[np.argmax(h, 1) + 1])
    return out


def noise_sigma(resid: np.ndarray, geom: Geometry, win: int) -> np.ndarray:
    """Per-slice noise: 1.4826 x median of the below-level residuals (a clast cannot inflate it)."""
    s = np.full(len(resid), np.nan, np.float32)
    for i in range(len(resid)):
        v = resid[i][geom.inside(i, resid.shape[1:])]
        v = v[v < 0]
        if v.size:
            s[i] = 1.4826 * np.median(-v)
    return ndi.median_filter(np.nan_to_num(s, nan=np.nanmedian(s)), win, mode="nearest")


def estimate_level(
    data: np.ndarray,
    geom: Geometry,
    step_mm: float,
    shape_cm: float = 12.0,
    scalar_cm: float = 3.0,
    passes: int = 2,
    k: float = 4.0,
    min_vox: int = 8,
    zero_offset: float = 0.0,
) -> tuple[Level, np.ndarray]:
    """Expected matrix level and the first-pass clast mask used to refine it (bool, same shape as data).

    Cupping is multiplicative in attenuation, so the shape is divided out of ``data + zero_offset``. Use 0 for an
    attenuation-like unit (positive for sediment) and 1000 for HU (so water is 1000, air 0).
    """
    data = data + np.float32(zero_offset) if zero_offset else data
    n = len(data)
    bins = _bins(data, geom)
    w_shape = max(3, int(shape_cm * 10 / step_mm) | 1)
    w_scalar = max(3, int(scalar_cm * 10 / step_mm) | 1)
    modes = ring_modes(data, geom, bins)
    shape0 = np.median(modes / modes.mean(1, keepdims=True), 0)
    shape0 /= shape0.mean()

    def flat_of(shape_rows: np.ndarray) -> np.ndarray:
        out = np.empty_like(data)
        for i in range(n):
            out[i] = data[i] / interp_rings(shape_rows[i], geom.radius(i, data.shape[1:]), geom.r_in[i])
        return out

    def scalar_of(flat: np.ndarray, ok: np.ndarray | None) -> np.ndarray:
        fb = np.linspace(*np.percentile(flat[:: max(1, n // 40)][:, ::2, ::2], [0.5, 99.5]), 221)
        s = np.zeros(n, np.float32)
        for i in range(n):
            m = geom.inside(i, data.shape[1:])
            if ok is not None:
                m &= ok[i]
            s[i] = _mode(flat[i][m], fb, 5) if m.any() else np.nan
        s = np.where(np.isfinite(s), s, np.nanmedian(s))
        return ndi.median_filter(s, w_scalar, mode="nearest")

    s = scalar_of(flat_of(np.tile(shape0, (n, 1))), None)
    delta = ndi.median_filter(modes / (s[:, None] * shape0[None]), size=(w_shape, 1), mode="nearest")
    delta = ndi.uniform_filter1d(delta, 3, axis=1, mode="nearest")
    shp = shape0[None] * delta
    flat = flat_of(shp)
    s = scalar_of(flat, None)
    level = np.empty_like(data)
    clast1 = np.zeros(data.shape, bool)
    sig = None
    for p in range(passes):
        for i in range(n):
            level[i] = s[i] * interp_rings(shp[i], geom.radius(i, data.shape[1:]), geom.r_in[i])
        sig = noise_sigma(data - level, geom, w_scalar)
        resid = data - level
        flag = resid > k * sig[:, None, None]
        lab, nl = ndi.label(flag)
        if nl:
            sz = np.bincount(lab.ravel())
            keep = sz >= min_vox
            keep[0] = False
            clast1 = ndi.binary_dilation(keep[lab], iterations=1)
        else:
            clast1 = flag
        if p < passes - 1:
            ok = ~ndi.binary_dilation(clast1, iterations=3) & ~(resid < -k * sig[:, None, None])
            s = scalar_of(flat, ok)
    if zero_offset:
        level -= np.float32(zero_offset)
        s = s - zero_offset  # per-slice level in the caller's unit (the shape is relative to the offset scale)
    return Level(level, s, shp, sig), clast1
