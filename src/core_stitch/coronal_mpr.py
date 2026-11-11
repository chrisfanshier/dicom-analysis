"""Coronal (or any oblique) MPR planes built from an axial DICOM stack with SimpleITK, plus the measures used to
compare them with the scanner's own MPR.

What is built
    One output plane is the **mean of a slab** of the axial volume, centred on the plane, ``thickness`` mm thick
    (measured along the plane normal), sampled at the centre of every output pixel.  The slab is sampled every
    ``step_mm`` along the normal, each sample a single ``sitk.Resample`` (identity transform, linear interpolation)
    onto a reference grid that carries the plane's full orientation, origin and pixel size.  Nothing is smoothed first.

    This is the recipe that reproduces the Toshiba Aquilon MPR (2 mm mean slab, pixel-centre sampling, plane taken
    from the DICOM header) closely on real sections: typically within a few HU, with some planes differing by up to the
    noise level, and a small constant offset from the scanner rounding down to whole numbers.  The checks are made by
    ``tools/coronal_compare.py``.

Conventions
    DICOM and ITK are both LPS and both place a voxel's position at its centre, so ``ImagePositionPatient`` is the
    ITK origin.  The axial stack must be scanner-aligned (orientation ``1 0 0 0 1 0``, no gantry tilt); anything
    else raises ``ValueError`` instead of silently producing a wrong plane.

Read-only on the DICOM; this module writes nothing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pydicom
import SimpleITK as sitk
from scipy import ndimage as ndi

from core_stitch.load import _hu, sort_slices

CORE_HU = -300.0  # his > CORE_HU, hole-filled, is "the core" (air is about -1000, the liner and sediment are above)
AIR_HU = -900.0


# ---------------------------------------------------------------------------------------------- reading


def _header_items(folder: Path) -> list[tuple[Path, pydicom.Dataset]]:
    items = []
    for p in sorted(Path(folder).iterdir()):
        if not p.is_file():
            continue
        try:
            ds = pydicom.dcmread(p, stop_before_pixels=True, force=True)
        except Exception:
            continue
        if "ImageOrientationPatient" in ds and "ImagePositionPatient" in ds:
            items.append((p, ds))
    if not items:
        raise FileNotFoundError(f"No DICOM images with a position and orientation in {folder}")
    uids: dict[str, list] = {}
    for it in items:
        uids.setdefault(str(it[1].get("SeriesInstanceUID", "")), []).append(it)
    items = max(uids.values(), key=len)  # a folder with several series: the one with most images
    return sort_slices(items)


@dataclass
class CoronalSeries:
    """The scanner's own MPR planes of one series (headers only; pixels are read plane by plane)."""

    folder: Path
    paths: list[Path]
    ipps: np.ndarray  # (N, 3) ImagePositionPatient of every plane, sorted along the plane normal
    r: np.ndarray  # direction of increasing column index
    c: np.ndarray  # direction of increasing row index
    n: np.ndarray  # plane normal r x c
    px: float
    rows: int
    cols: int
    thickness: float
    description: str

    def plane_hu(self, k: int) -> np.ndarray:
        return _hu(pydicom.dcmread(self.paths[k], force=True))

    @property
    def n_planes(self) -> int:
        return len(self.paths)


def read_coronal(folder: Path | str) -> CoronalSeries:
    items = _header_items(Path(folder))
    ds0 = items[0][1]
    r = np.array([float(x) for x in ds0.ImageOrientationPatient[:3]])
    c = np.array([float(x) for x in ds0.ImageOrientationPatient[3:]])
    return CoronalSeries(
        folder=Path(folder),
        paths=[p for p, _ in items],
        ipps=np.array([[float(x) for x in d.ImagePositionPatient] for _, d in items]),
        r=r, c=c, n=np.cross(r, c),
        px=float(ds0.PixelSpacing[0]), rows=int(ds0.Rows), cols=int(ds0.Columns),
        thickness=float(ds0.get("SliceThickness", 0.0) or 0.0),
        description=str(ds0.get("SeriesDescription", "")).strip(),
    )


@dataclass
class AxialGeometry:
    paths: list[Path]
    z: np.ndarray
    x0: float
    y0: float
    px_x: float
    px_y: float
    ncols: int
    nrows: int
    thickness: float
    description: str


def read_axial_geometry(folder: Path | str) -> AxialGeometry:
    items = _header_items(Path(folder))
    ds0 = items[0][1]
    iop = [round(float(x), 3) for x in ds0.ImageOrientationPatient]
    if iop != [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]:
        raise ValueError(f"The axial stack must be scanner-aligned (orientation 1 0 0 0 1 0); got {iop}")
    return AxialGeometry(
        paths=[p for p, _ in items],
        z=np.array([float(d.ImagePositionPatient[2]) for _, d in items]),
        x0=float(ds0.ImagePositionPatient[0]), y0=float(ds0.ImagePositionPatient[1]),
        px_x=float(ds0.PixelSpacing[1]), px_y=float(ds0.PixelSpacing[0]),
        ncols=int(ds0.Columns), nrows=int(ds0.Rows),
        thickness=float(ds0.get("SliceThickness", 0.0) or 0.0),
        description=str(ds0.get("SeriesDescription", "")).strip(),
    )


def plane_window(cor: CoronalSeries, geom: AxialGeometry, margin_mm: float = 3.0, y_margin_mm: float = 4.0):
    """Which of the coronal's columns fall inside the axial field of view, and which axial rows (y) are needed.

    Returns ``(j_lo, j_hi, y_min, y_max)``: columns ``j_lo:j_hi`` of the coronal, axial y range in mm.
    """
    V = (cor.rows - 1) * cor.px
    u = np.arange(cor.cols) * cor.px
    x1 = cor.ipps[:, 0][:, None] + cor.r[0] * u[None, :]
    x2 = x1 + cor.c[0] * V
    lo = np.minimum(x1, x2).min(axis=0)
    hi = np.maximum(x1, x2).max(axis=0)
    xa = geom.x0 - margin_mm
    xb = geom.x0 + (geom.ncols - 1) * geom.px_x + margin_mm
    keep = np.where((hi >= xa) & (lo <= xb))[0]
    if keep.size == 0:
        raise ValueError("The coronal planes do not overlap the axial field of view in x")
    j_lo, j_hi = int(keep[0]), int(keep[-1]) + 1
    u_ext = np.array([j_lo * cor.px, (j_hi - 1) * cor.px])
    ys = (cor.ipps[:, 1][:, None, None] + cor.r[1] * u_ext[None, :, None] + cor.c[1] * np.array([0.0, V])[None, None, :])
    return j_lo, j_hi, float(ys.min() - y_margin_mm), float(ys.max() + y_margin_mm)


@dataclass
class AxialStack:
    """The axial volume as a SimpleITK image (HU, float32), cropped in y to the rows the planes need."""

    image: sitk.Image
    ones: sitk.Image  # same grid, all ones: resampled to find which output pixels have axial data
    geom: AxialGeometry
    rows: tuple[int, int]
    filled_z: list[float] = field(default_factory=list)  # slices missing from the series, filled by interpolation

    @property
    def z_range(self) -> tuple[float, float]:
        z0 = self.image.GetOrigin()[2]
        return z0, z0 + (self.image.GetSize()[2] - 1) * self.image.GetSpacing()[2]


def load_axial(geom: AxialGeometry, y_range: tuple[float, float]) -> AxialStack:
    """Read the axial slices, keeping only the rows ``y_range`` (mm) needs; a missing slice is filled by interpolation."""
    j0 = max(int(np.floor((y_range[0] - geom.y0) / geom.px_y)), 0)
    j1 = min(int(np.ceil((y_range[1] - geom.y0) / geom.px_y)) + 1, geom.nrows)
    if j1 <= j0:
        raise ValueError("The coronal planes do not overlap the axial field of view in y")
    dz = float(np.median(np.diff(geom.z)))
    z0 = float(geom.z.min())
    grid_n = int(round((float(geom.z.max()) - z0) / dz)) + 1
    vol = np.empty((grid_n, j1 - j0, geom.ncols), np.float32)
    have = np.zeros(grid_n, bool)
    for p, z in zip(geom.paths, geom.z):
        k = int(round((z - z0) / dz))
        vol[k] = _hu(pydicom.dcmread(p, force=True))[j0:j1, :]
        have[k] = True
    filled = []
    for k in np.where(~have)[0]:
        lo = int(np.max(np.where(have[:k])[0]))
        hi = int(k + np.min(np.where(have[k:])[0]))
        w = (k - lo) / (hi - lo)
        vol[k] = (1 - w) * vol[lo] + w * vol[hi]
        filled.append(z0 + k * dz)
    img = sitk.GetImageFromArray(vol)
    img.SetSpacing((geom.px_x, geom.px_y, dz))
    img.SetOrigin((geom.x0, geom.y0 + j0 * geom.px_y, z0))
    return AxialStack(image=img, ones=sitk.Cast(img * 0 + 1, sitk.sitkFloat32), geom=geom, rows=(j0, j1), filled_z=filled)


# ---------------------------------------------------------------------------------------------- rendering


class PlaneRenderer:
    """Renders planes of an ``AxialStack`` onto the pixel grid of a ``CoronalSeries`` (columns ``j_lo:j_hi`` only)."""

    def __init__(self, stack: AxialStack, cor: CoronalSeries, j_lo: int, j_hi: int) -> None:
        self.stack, self.cor, self.j_lo, self.j_hi = stack, cor, j_lo, j_hi
        ref = sitk.Image(j_hi - j_lo, cor.rows, 1, sitk.sitkFloat32)
        ref.SetSpacing((cor.px, cor.px, 1.0))
        ref.SetDirection(tuple(float(x) for x in np.column_stack([cor.r, cor.c, cor.n]).flatten()))
        self._ref = ref

    def render(self, ipp, thickness: float = 2.0, mode: str = "mean", dy: float = 0.0, step_mm: float = 0.1):
        """One plane: ``(values, valid)``, both shaped ``(rows, j_hi - j_lo)``.

        ``ipp`` is the plane's ImagePositionPatient; ``dy`` shifts the slab centre along the plane normal (mm).
        ``valid`` is False wherever any slab sample falls outside the axial volume.
        """
        cor, ref = self.cor, self._ref
        base = np.asarray(ipp, float) + cor.r * (self.j_lo * cor.px)
        N = 1 if thickness <= 0 else max(1, int(round(thickness / step_mm)))
        offs = [0.0] if thickness <= 0 else list((np.arange(N) + 0.5) / N * thickness - thickness / 2)
        acc = cnt = mx = None
        for o in offs:
            ref.SetOrigin(tuple(float(x) for x in base + cor.n * (o + dy)))
            v = sitk.GetArrayFromImage(
                sitk.Resample(self.stack.image, ref, sitk.Transform(), sitk.sitkLinear, 0.0, sitk.sitkFloat32))[0]
            ok = sitk.GetArrayFromImage(
                sitk.Resample(self.stack.ones, ref, sitk.Transform(), sitk.sitkNearestNeighbor, 0.0, sitk.sitkFloat32))[0] > 0.5
            v = np.where(ok, v, 0.0)
            acc = v if acc is None else acc + v
            cnt = ok.astype(np.float32) if cnt is None else cnt + ok
            w = np.where(ok, v, -np.inf)
            mx = w if mx is None else np.maximum(mx, w)
        valid = cnt >= len(offs) - 0.5
        out = acc / np.maximum(cnt, 1) if mode == "mean" else np.where(np.isfinite(mx), mx, 0.0)
        return np.where(valid, out, np.nan), valid

    def render_plane(self, k: int, **kw):
        return self.render(self.cor.ipps[k], **kw)

    def his_plane(self, k: int) -> np.ndarray:
        return self.cor.plane_hu(k)[:, self.j_lo:self.j_hi]


# ---------------------------------------------------------------------------------------------- measures


def core_mask(his: np.ndarray, valid: np.ndarray, px: float, trim_rows: int = 6):
    """The core's interior in his plane (bright material, away from the edges), where the axial exists.

    The ``trim_rows`` rows nearest each end of the axial's coverage are left out (end-of-stack effect).
    Returns ``(mask, valid_rows)``.
    """
    rows = np.where(valid.any(axis=1))[0]
    if rows.size < 50:
        return np.zeros(valid.shape, bool), rows
    filled = ndi.binary_fill_holes(his > CORE_HU)
    core = ndi.binary_erosion(filled, iterations=max(2, int(round(2.8 / px)))) & valid
    core[: rows[0] + trim_rows] = False
    core[rows[-1] - trim_rows + 1:] = False
    return core, rows


def core_area_estimate(his: np.ndarray, px: float) -> int:
    """Size of the core in his plane (no rendering needed); used to pick a plane that really cuts the core."""
    filled = ndi.binary_fill_holes(his > CORE_HU)
    return int(ndi.binary_erosion(filled, iterations=max(2, int(round(2.8 / px)))).sum())


def pick_plane(areas: list[int], min_fraction: float = 0.5) -> int:
    """The middle plane among those whose core area is at least ``min_fraction`` of the largest."""
    a = np.asarray(areas, float)
    good = np.where(a >= min_fraction * a.max())[0]
    return int(good[len(good) // 2])


def difference_stats(ours: np.ndarray, his: np.ndarray, core: np.ndarray) -> dict:
    m = core & np.isfinite(ours)
    d = (ours - his)[m]
    p5, p95 = np.percentile(d, [5, 95])
    return dict(n=int(m.sum()), mean=float(d.mean()), median=float(np.median(d)), sd=float(d.std()),
                p5=float(p5), p95=float(p95), max_abs=float(np.abs(d).max()),
                corr=float(np.corrcoef(ours[m], his[m])[0, 1]),
                frac_gt10=float(np.mean(np.abs(d) > 10)), frac_gt20=float(np.mean(np.abs(d) > 20)))


def fit_terms(his: np.ndarray, ours: np.ndarray, gy: np.ndarray, core: np.ndarray) -> dict:
    """Least-squares explanation of (ours - his): sub-pixel shifts across / along the core, slab offset, constant.

    ``gy`` is d(ours)/d(slab position) in HU per mm, i.e. (render at dy=+0.25 minus dy=-0.25) / 0.5.
    Shifts are in pixels of his grid: a positive ``sx`` means ours is displaced by +sx pixels along x.
    """
    gx = ndi.sobel(his, axis=1) / 8.0
    gz = ndi.sobel(his, axis=0) / 8.0
    ok = core & np.isfinite(ours) & np.isfinite(gy)
    X = np.column_stack([gx[ok], gz[ok], gy[ok], np.ones(ok.sum())])
    y = (ours - his)[ok]
    co, *_ = np.linalg.lstsq(X, y, rcond=None)
    return dict(sx=float(co[0]), sz=float(co[1]), sy=float(co[2]), const=float(co[3]), sd_after=float(np.std(y - X @ co)))


_B = 8
_RR, _CC = np.mgrid[0:_B, 0:_B]
_XP = np.column_stack([_RR.ravel(), _CC.ravel(), np.ones(_B * _B)])


def flat_blocks(his: np.ndarray, core: np.ndarray, rows: np.ndarray) -> list[tuple[int, int]]:
    """Top-left corners of 8x8-pixel blocks that are flat in his image (no visible gradient, no pore or crack)."""
    inner = ndi.binary_erosion(core, iterations=6)
    out = []
    for r0 in range(int(rows[0]) + 12, int(rows[-1]) - _B - 12, _B):
        for c0 in range(0, his.shape[1] - _B, _B):
            if not inner[r0:r0 + _B, c0:c0 + _B].all():
                continue
            h = his[r0:r0 + _B, c0:c0 + _B].ravel()
            co, *_ = np.linalg.lstsq(_XP, h, rcond=None)
            if np.hypot(co[0], co[1]) > 4.0 or np.abs(h - _XP @ co).max() > 90:
                continue
            out.append((r0, c0))
    return out


def block_sd(img: np.ndarray, blocks: list[tuple[int, int]]) -> np.ndarray:
    """Noise SD per block (what is left after removing a plane fit)."""
    out = []
    for r0, c0 in blocks:
        v = img[r0:r0 + _B, c0:c0 + _B].ravel()
        co, *_ = np.linalg.lstsq(_XP, v, rcond=None)
        out.append((v - _XP @ co).std(ddof=3))
    return np.array(out)


BLOCK = _B
_OFFS = np.arange(-8, 8.001, 0.25)  # profile positions relative to the 50 % point, in pixels


def _sample(a: np.ndarray, rr, cc):
    return ndi.map_coordinates(np.nan_to_num(a, nan=-1000.0), [rr, cc], order=1, mode="nearest")


def _profile_at(images: dict[str, np.ndarray], his: np.ndarray, coords, minstep: float):
    """Normalised, aligned profiles of every image along one line.

    ``coords(t)`` gives (rows, cols) at distance ``t`` (pixels) along the line, positive ``t`` towards the bright side.
    The profiles are scaled so the dark plateau is 0 and the bright plateau is 1 (plateaus measured in ``his``) and
    aligned at his 50 % crossing.  Returns None if the line has no clear edge or leaves the image.
    """
    t = np.arange(-10, 10.001, 0.25)
    r_, c_ = coords(t)
    if r_.min() < 0 or c_.min() < 0 or r_.max() > his.shape[0] - 1 or c_.max() > his.shape[1] - 1:
        return None
    ph = _sample(his, r_, c_)
    lo, hi = np.percentile(ph[t < -3], 50), np.percentile(ph[t > 3], 50)
    if hi - lo < minstep:
        return None
    mid = lo + 0.5 * (hi - lo)
    k = np.where((ph[:-1] - mid) * (ph[1:] - mid) <= 0)[0]
    if k.size == 0:
        return None
    k = k[np.argmin(np.abs(t[k]))]
    t50 = t[k] + (mid - ph[k]) / (ph[k + 1] - ph[k]) * 0.25
    r2, c2 = coords(t50 + _OFFS)
    if r2.min() < 0 or c2.min() < 0 or r2.max() > his.shape[0] - 1 or c2.max() > his.shape[1] - 1:
        return None
    return {name: (_sample(a, r2, c2) - lo) / (hi - lo) for name, a in images.items()}


def edge_profiles(images: dict[str, np.ndarray], his: np.ndarray, core: np.ndarray, rows: np.ndarray, direction: str):
    """Normalised edge profiles across the sediment boundary, aligned at his 50 % point.

    ``direction``: ``left`` / ``right`` (the core's side walls, scanned along each row) or ``bottom`` (the end of a
    piece of core, scanned along each column).
    Returns ``(profiles, info)``: ``profiles[name]`` is (n_profiles, n_offsets); ``info`` has the line index and edge
    position of each profile and the position axis ``offs`` (pixels).
    """
    med = float(np.median(his[core]))
    thr, minstep = 0.5 * (med + AIR_HU), 0.4 * (med - AIR_HU)
    sm = ndi.gaussian_filter(his, 1.0)
    prof = {k: [] for k in images}
    info = dict(line=[], edge=[], offs=_OFFS, direction=direction)
    lines = range(int(rows[0]) + 12, int(rows[-1]) - 12, 2) if direction in ("left", "right") else range(0, his.shape[1], 2)
    for ln in lines:
        if direction in ("left", "right"):
            idx = np.where(sm[ln, :] > thr)[0]
            if idx.size == 0:
                continue
            if direction == "left":
                e = int(idx[0]); coords = lambda tt, ln=ln, e=e: (np.full_like(tt, ln, float), e + tt)
            else:
                e = int(idx[-1]); coords = lambda tt, ln=ln, e=e: (np.full_like(tt, ln, float), e - tt)
        else:
            lo_r, hi_r = int(rows[0]) + 20, int(rows[-1]) - 20
            idx = np.where(sm[lo_r:hi_r, ln] > thr)[0]
            if idx.size == 0:
                continue
            e = lo_r + int(idx[-1]); coords = lambda tt, ln=ln, e=e: (e - tt, np.full_like(tt, ln, float))
        res = _profile_at(images, his, coords, minstep)
        if res is None:
            continue
        for name in images:
            prof[name].append(res[name])
        info["line"].append(ln); info["edge"].append(e)
    return {k: np.array(v) for k, v in prof.items()}, info


def across_core_edge(images: dict[str, np.ndarray], his: np.ndarray, core: np.ndarray, segment: tuple[int, int], min_cols: int = 8):
    """The strongest edge that runs ACROSS the core inside ``segment`` (e.g. the top or bottom of a void or of a gap).

    In every column of the core the sharpest step along the core is found; the edge is the row where most columns agree
    (within 4 rows, same direction).  Profiles are taken along the core, one per agreeing column, so they measure the
    sharpness along the core.  Returns ``(profiles, info)`` with ``info['row']``, ``info['cols']``, ``info['offs']``,
    or ``(None, None)`` if the segment has no such edge.
    """
    s0, s1 = segment
    med = float(np.median(his[core]))
    minstep = 0.4 * (med - AIR_HU)
    sm = ndi.gaussian_filter(his, 1.0)
    cand = []
    for c in range(his.shape[1]):
        if core[s0:s1, c].sum() < 20:
            continue
        i = int(np.argmax(np.abs(np.gradient(sm[s0:s1, c]))))
        r = s0 + i
        if r - 12 < s0 or r + 13 > s1:
            continue
        above, below = float(np.median(sm[r - 8:r - 3, c])), float(np.median(sm[r + 4:r + 9, c]))
        if abs(above - below) < minstep:
            continue
        cand.append((c, r, 1 if above > below else -1))
    if len(cand) < min_cols:
        return None, None
    best = None
    for sign in (1, -1):
        rr = np.array([r for _c, r, sg in cand if sg == sign])
        if rr.size < min_cols:
            continue
        for r0 in np.unique(rr):
            n = int(np.sum(np.abs(rr - r0) <= 4))
            if best is None or n > best[0]:
                best = (n, int(r0), sign)
    if best is None or best[0] < min_cols:
        return None, None
    _n, r_mode, sign = best
    prof = {k: [] for k in images}
    cols, rws = [], []
    for c, r, sg in cand:
        if sg != sign or abs(r - r_mode) > 4:
            continue
        coords = (lambda tt, c=c, r=r: (r - tt, np.full_like(tt, c, float))) if sign > 0 else \
                 (lambda tt, c=c, r=r: (r + tt, np.full_like(tt, c, float)))
        res = _profile_at(images, his, coords, minstep)
        if res is None:
            continue
        for name in images:
            prof[name].append(res[name])
        cols.append(c); rws.append(r)
    if len(cols) < min_cols:
        return None, None
    return {k: np.array(v) for k, v in prof.items()}, dict(row=int(np.median(rws)), cols=cols, offs=_OFFS, sign=sign)


def edge_width(profile: np.ndarray, offs: np.ndarray = _OFFS) -> float:
    """10-90 % rise distance (pixels) of an averaged, normalised edge profile."""
    def cross(level):
        k = np.where((profile[:-1] - level) * (profile[1:] - level) <= 0)[0]
        if k.size == 0:
            return np.nan
        k = k[np.argmin(np.abs(offs[k]))]
        return offs[k] + (level - profile[k]) / (profile[k + 1] - profile[k]) * (offs[1] - offs[0])
    return float(cross(0.9) - cross(0.1))


def peak_gradient_ratio(ours: np.ndarray, his: np.ndarray, core: np.ndarray) -> float:
    def g(a):
        a = np.nan_to_num(a, nan=-1000.0)
        return np.hypot(ndi.sobel(a, axis=1), ndi.sobel(a, axis=0)) / 8.0
    gh, go = g(his), g(ours)
    sel = core & (gh > 60)
    return float(np.median(go[sel] / gh[sel])) if sel.sum() > 200 else float("nan")


def downcore_profile(img: np.ndarray, core: np.ndarray, min_pixels: int = 20) -> np.ndarray:
    """Mean HU along the core, one value per image row (NaN where fewer than ``min_pixels`` core pixels)."""
    m = core & np.isfinite(img)
    cnt = m.sum(axis=1)
    s = np.where(m, img, 0.0).sum(axis=1)
    return np.where(cnt >= min_pixels, s / np.maximum(cnt, 1), np.nan)


def slab_sweep(renderer: PlaneRenderer, k: int, his: np.ndarray, core: np.ndarray, thicknesses=(0.0, 1.0, 1.5, 2.0, 2.5, 3.0)) -> dict:
    """SD of (ours - his) for several slab thicknesses (mean) and for a 2 mm MIP: the recipe check."""
    out = {}
    for T in thicknesses:
        a, _ = renderer.render_plane(k, thickness=T)
        out[f"mean {T:g} mm"] = difference_stats(a, his, core)["sd"]
    a, _ = renderer.render_plane(k, thickness=2.0, mode="max")
    out["MIP 2 mm"] = difference_stats(a, his, core)["sd"]
    return out
