"""Down-core profiles: choose a region of each slice, choose which classes count, reduce to one number per slice.

``Region`` (where in the slice): the whole sediment disc, a central disc (fraction of the radius) or a vertical
plane / band through the axis (the MSCL gamma beam samples a narrow central strip, so a 2 mm band is the natural
comparison). ``use`` (which classes count): matrix only is the default; any set of ``Cls`` is allowed, so
"everything" gives the plain mean and a single class gives a class-only profile (clast, burrow, ...).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage as ndi

from .classes import Cls
from .geometry import Geometry


@dataclass(frozen=True)
class Region:
    kind: str = "whole"  # whole | central | band
    frac: float = 0.5  # central: fraction of the inner radius
    thickness_mm: float = 2.0  # band: full width, mm
    offset_mm: float = 0.0  # band: shift of the plane from the axis, mm (positive = +col)

    def mask(self, i: int, shape: tuple[int, int], geom: Geometry) -> np.ndarray:
        r = geom.radius(i, shape)
        m = r < geom.r_in[i]
        if self.kind == "central":
            m &= r < self.frac * geom.r_in[i]
        elif self.kind == "band":
            xx = np.arange(shape[1], dtype=np.float32)[None, :] - geom.center[i, 1]
            half = 0.5 * self.thickness_mm / geom.voxel_mm
            m &= np.abs(xx - self.offset_mm / geom.voxel_mm) <= half
        elif self.kind != "whole":
            raise ValueError(f"unknown region kind {self.kind!r}")
        return m


@dataclass
class Profile:
    depth_cm: np.ndarray
    mean: np.ndarray
    std: np.ndarray
    count: np.ndarray  # voxels used per slice
    coverage: np.ndarray  # share of the region's voxels that were used


def down_core(
    vol: np.ndarray,
    cls: np.ndarray | None,
    geom: Geometry,
    step_mm: float,
    region: Region = Region(),
    use: tuple[Cls, ...] | None = (Cls.MATRIX,),
    top_cm: float = 0.0,
    min_count: int = 20,
) -> Profile:
    """One value per slice. ``use=None`` or ``cls=None`` means every voxel of the region counts."""
    n = len(vol)
    mean = np.full(n, np.nan)
    std = np.full(n, np.nan)
    cnt = np.zeros(n, int)
    cov = np.zeros(n)
    for i in range(n):
        reg = region.mask(i, vol.shape[1:], geom)
        m = reg if (cls is None or use is None) else reg & np.isin(cls[i], [int(c) for c in use])
        k = int(m.sum())
        cnt[i] = k
        cov[i] = k / max(int(reg.sum()), 1)
        if k >= min_count:
            v = vol[i][m]
            mean[i] = v.mean()
            std[i] = v.std()
    return Profile(top_cm + np.arange(n) * step_mm / 10, mean, std, cnt, cov)


def class_fractions(cls: np.ndarray, geom: Geometry, region: Region = Region()) -> dict[str, np.ndarray]:
    """Per-slice share of the region taken by each class."""
    n = len(cls)
    out = {c.name.lower(): np.zeros(n) for c in Cls}
    for i in range(n):
        reg = region.mask(i, cls.shape[1:], geom)
        tot = max(int(reg.sum()), 1)
        for c in Cls:
            out[c.name.lower()][i] = ((cls[i] == c) & reg).sum() / tot
    return out


def footprint_smooth(depth_cm: np.ndarray, y: np.ndarray, fwhm_mm: float) -> np.ndarray:
    """NaN-aware Gaussian smooth along depth to a detector footprint (e.g. a gamma-beam FWHM)."""
    step = float(np.median(np.diff(depth_cm))) * 10
    s = fwhm_mm / 2.3548 / step
    ok = np.isfinite(y)
    num = ndi.gaussian_filter1d(np.where(ok, y, 0.0), s, mode="nearest")
    den = ndi.gaussian_filter1d(ok.astype(float), s, mode="nearest")
    return np.where(den > 0.2, num / np.maximum(den, 1e-9), np.nan)
