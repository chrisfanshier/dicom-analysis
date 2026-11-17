"""Voxel classes: matrix, void, clast, low-density body, dense patch, wall contact.

Everything is judged against the expected matrix level (see ``levels``), not a global threshold.

* void: below an absolute floor ``void_below`` (an air-like value; without it there is no void class).
* clast: above level + k sigma, at least ``min_vox`` connected voxels, grown by one voxel (the edge ramp).
* low-density body (burrow, organic): after a small Gaussian smooth, below level by 2 sigma_s, seeded at 3.5 sigma_s
  (hysteresis), at least ``min_body_vox`` voxels, away from clasts and voids. Bodies that reach the liner wall band
  are called *wall contact* instead (gaps, cracks, edge effects).
* dense patch: the same on the positive side (diffuse dense bodies a clast test misses).

Provenance: standard pieces (MAD noise, hysteresis, component size filter); not a published method.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

import numpy as np
from scipy import ndimage as ndi
from skimage.filters import apply_hysteresis_threshold

from .geometry import Geometry
from .levels import Level


class Cls(IntEnum):
    MATRIX = 0
    VOID = 1
    CLAST = 2
    LOWDENS = 3
    DENSE = 4
    WALL = 5


@dataclass
class ClassParams:
    k: float = 4.0  # clast / void-by-residual cut, in sigma
    min_vox: int = 8
    void_below: float | None = None  # absolute air-like floor
    smooth: float = 1.5  # voxels, for the body detectors
    lo: float = 2.0
    hi: float = 3.5
    min_body_vox: int = 60
    wall_band_px: float = 12.0


def _label_filter(flag: np.ndarray, min_vox: int) -> np.ndarray:
    lab, n = ndi.label(flag)
    if not n:
        return np.zeros(flag.shape, bool)
    sz = np.bincount(lab.ravel())
    keep = sz >= min_vox
    keep[0] = False
    return keep[lab]


def _bodies(vol, lev, geom, cls, valid, sign, p: ClassParams):
    vs = ndi.gaussian_filter(vol, p.smooth)
    rel = sign * (vs - lev.level)  # positive where the voxel is on the side being searched (sign=-1: below matrix)
    m = valid & (cls == Cls.MATRIX)
    # noise of the smoothed matrix, from the side NOT being searched (a body cannot inflate it)
    other = -rel[m]
    other = other[other > 0]
    if other.size == 0:
        return np.zeros(vol.shape, bool)
    sig_s = 1.4826 * np.median(other)
    near = ndi.binary_dilation(cls == Cls.CLAST, iterations=3) | ndi.binary_dilation(cls == Cls.VOID, iterations=2)
    cand = apply_hysteresis_threshold(rel, p.lo * sig_s, p.hi * sig_s)
    return _label_filter(cand & valid & ~near, p.min_body_vox)


def classify(
    vol: np.ndarray,
    geom: Geometry,
    lev: Level,
    step_mm: float,
    params: ClassParams | None = None,
    trim_cm: float = 5.0,
    trim_in_px: float = 6.0,
) -> np.ndarray:
    """uint8 class volume (``Cls`` codes). Voxels outside the sediment radius stay 0 (use ``inside`` to restrict).

    ``trim_cm``: the section ends are excluded from the body detectors (their level is least certain).
    """
    p = params or ClassParams()
    n = len(vol)
    inside = np.stack([geom.inside(i, vol.shape[1:]) for i in range(n)])
    resid = vol - lev.level
    sig = lev.sigma[:, None, None]
    cls = np.zeros(vol.shape, np.uint8)
    clast = ndi.binary_dilation(_label_filter(resid > p.k * sig, p.min_vox), iterations=1)
    if p.void_below is not None:
        cls[(vol < p.void_below) & inside] = Cls.VOID
    cls[clast & inside & (cls == 0)] = Cls.CLAST
    z_cm = np.arange(n) * step_mm / 10
    valid = inside & (z_cm > trim_cm)[:, None, None] & (z_cm < z_cm[-1] - trim_cm)[:, None, None]
    valid &= np.stack([geom.radius(i, vol.shape[1:]) < geom.r_in[i] - trim_in_px for i in range(n)])
    low = _bodies(vol, lev, geom, cls, valid, -1, p)
    dense = _bodies(vol, lev, geom, cls, valid, +1, p) & ~low
    band = np.stack([geom.radius(i, vol.shape[1:]) > geom.r_in[i] - p.wall_band_px for i in range(n)])
    lab, _ = ndi.label(low)
    touch = np.unique(lab[band & low])
    touch = touch[touch > 0]
    wall = np.isin(lab, touch) if touch.size else np.zeros(low.shape, bool)
    low &= ~wall
    cls[low & (cls == 0)] = Cls.LOWDENS
    cls[dense & (cls == 0)] = Cls.DENSE
    cls[wall & (cls == 0)] = Cls.WALL
    cls[~inside] = 0
    return cls
