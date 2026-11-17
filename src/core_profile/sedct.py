"""Profile part of SedCT (Reilly et al. 2017), ported from the published MATLAB description and source.

Per depth: draw a random sample of voxel values from the sediment region, sort, cull background, cut at the first
sorted step larger than ``maxslope`` (the plateau of the distribution), keep the longest unbroken run, trim its
ends, and report the mean, SD and count. It is a distribution-plateau estimator, close to a matrix-only mean.

Parity with the MATLAB program is NOT verified (no MATLAB reference output was available). Known quirks that are
reproduced on purpose: the trim removes ``trim + 1`` low and ``trim - 1`` high values, and the run after the last
NaN is never a candidate. Sampling is random: pass a seed, and compare statistically, not value by value.
"""

from __future__ import annotations

import numpy as np

from .geometry import Geometry
from .reduce import Region


def _run_trim(col: np.ndarray, trim: int) -> np.ndarray:
    """One depth's sorted, NaN-culled column; MATLAB 1-based indexing emulated."""
    F = np.where(np.isnan(col))[0] + 1
    D = np.diff(F)
    if D.size == 0:
        return np.full_like(col, np.nan)
    L = int(np.argmax(D)) + 1
    M = D.max()
    s = D[: L - 1].sum()
    i1 = s + 2 + trim
    i2 = s + 2 + M - trim
    out = col.copy()
    out[: min(i1, out.size)] = np.nan
    if i2 < 1:
        i2 = 1
    out[i2 - 1 :] = np.nan
    return out


def sedct_profile(y: np.ndarray, backv: float = 0.0, maxslope: float = 1.0, trim: int = 500, pxcut: float = 10, minpx: int = 0):
    """y: (samples, depths) values. Returns mean, SD, count per depth. Units follow the data (HU in the original)."""
    z = np.sort(y.astype(float), axis=0)
    z[z <= backv] = np.nan
    z = z[np.nansum(z, axis=1) > 0]
    dz = np.diff(z, axis=0)
    z = z[:-1]
    z[dz > maxslope] = np.nan
    z = np.vstack([np.full((1, z.shape[1]), np.nan), z])
    for j in range(z.shape[1]):
        z[:, j] = _run_trim(z[:, j], trim)
    z = np.sort(z, axis=0)
    z = z[np.nansum(z, axis=1) != 0]
    with np.errstate(all="ignore"):
        gs = np.nanmean(z, axis=0)
        sd = np.nanstd(z, axis=0, ddof=1)
        px = np.sum(~np.isnan(z), axis=0)
    cut = px < px.mean() - pxcut * px.std(ddof=1)
    gs[cut] = np.nan
    sd[cut] = np.nan
    gs[px < minpx] = np.nan
    gs[gs <= 0] = np.nan
    return gs, sd, px


def sedct_from_axial(
    vol: np.ndarray, geom: Geometry, region: Region = Region(), samples: int = 5000, seed: int = 0, **kw
):
    """Sample ``samples`` voxels per slice from the region (with replacement only if the region is smaller) and run the profile."""
    rng = np.random.default_rng(seed)
    y = np.empty((samples, len(vol)))
    for i in range(len(vol)):
        v = vol[i][region.mask(i, vol.shape[1:], geom)]
        y[:, i] = v[rng.choice(v.size, samples, replace=v.size < samples)]
    return sedct_profile(y, **kw)
