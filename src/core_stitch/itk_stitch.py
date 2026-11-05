"""Header-geometry stitch of two MPR halves with SimpleITK (experimental engine).

Procedure (each step is a standard ITK operation):

1. Each series becomes a 3-D SimpleITK image in *physical* space: spacing, origin
   (``ImagePositionPatient`` of the first slice) and direction (from
   ``ImageOrientationPatient``) all come from the DICOM headers.  Two halves from the
   same scan therefore already sit where the scanner put them: placing them needs no
   search.
2. The half with the finer in-plane pixel size is the **reference**.  Its voxels are
   copied into the output unchanged; the output grid is the reference lattice extended
   to cover both halves.
3. The other half is resampled **once**, straight from its original voxels onto the
   reference grid (linear interpolation, or nearest-neighbour when its voxels already
   fall on the reference lattice).
4. In the overlap the two halves are averaged, except within ``trim_rows`` of either
   half's image edge, where only the half that has data is used.  Outside the core the
   output is real air.

An optional ``move_mm`` shifts the *bottom* half relative to the header placement
(manual nudges, or the image-based fallback start); it is given as
(along the core, across the core, through the stack) in millimetres.

This module does not read or write DICOM exports; see ``tools/itk_compare.py``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pydicom
import SimpleITK as sitk
from scipy.ndimage import binary_erosion

from core_stitch.geometry import air_value
from core_stitch.load import MprSeries, load_series
from core_stitch.pipeline import _content_mask
from core_stitch.segment import mid_slice, segment_core

PLACEMENTS = ("whole_pixel", "subpixel")
INTERPOLATORS = {"linear": sitk.sitkLinear, "bspline": sitk.sitkBSpline, "nearest": sitk.sitkNearestNeighbor}
SAME_TOL = 1e-3


@dataclass
class Half:
    name: str  # "top" or "bottom"
    series: MprSeries
    image: sitk.Image
    row_dir: np.ndarray  # direction of increasing column index
    col_dir: np.ndarray  # direction of increasing row index (down the core)
    normal: np.ndarray
    spacing_xyz: tuple[float, float, float]  # (column pitch, row pitch, slice step) mm
    threshold: float
    air: float
    core_cols: tuple[int, int] = (0, 0)  # first/last column holding core in the middle slice


@dataclass
class ItkStitch:
    volume: np.ndarray  # fused HU, (slices, rows, cols)
    grid: sitk.Image  # geometry of ``volume``
    reference: Half
    moving: Half
    ref_vol: np.ndarray  # reference voxels on the output grid
    mov_vol: np.ndarray  # moving voxels on the output grid
    ref_footprint: np.ndarray
    mov_footprint: np.ndarray
    ref_core: np.ndarray
    mov_core: np.ndarray
    info: dict = field(default_factory=dict)


def _direction(row: np.ndarray, col: np.ndarray) -> tuple[np.ndarray, tuple[float, ...]]:
    normal = np.cross(row, col)
    d = np.column_stack([row, col, normal])
    return normal, tuple(float(x) for x in d.flatten())


def load_half(name: str, folder: Path | str, uid: str | None) -> Half:
    series = load_series(folder, uid)
    ds = pydicom.dcmread(series.slice_paths[0], stop_before_pixels=True, force=True)
    iop = np.asarray(ds.ImageOrientationPatient, float)
    row, col = iop[:3], iop[3:]
    normal, direction = _direction(row, col)
    z_mm, row_mm, col_mm = series.spacing_mm  # row_mm: step between rows; col_mm: step between columns
    img = sitk.GetImageFromArray(np.ascontiguousarray(series.volume, dtype=np.float32))
    img.SetSpacing((float(col_mm), float(row_mm), float(z_mm)))
    img.SetOrigin(tuple(float(x) for x in ds.ImagePositionPatient))
    img.SetDirection(direction)
    mid = mid_slice(series.volume)
    seg = segment_core(mid)
    cols = np.where(seg.mask.any(axis=0))[0]
    return Half(
        name=name, series=series, image=img, row_dir=row, col_dir=col, normal=normal,
        spacing_xyz=(float(col_mm), float(row_mm), float(z_mm)),
        threshold=float(seg.threshold), air=air_value(mid),
        core_cols=(int(cols.min()), int(cols.max())) if cols.size else (0, int(mid.shape[1]) - 1),
    )


def _corners_index(half: Half, ref: sitk.Image) -> np.ndarray:
    """Continuous index (in the reference lattice) of the 8 corner voxel centres of ``half``."""
    nx, ny, nz = half.image.GetSize()
    pts = []
    for i in (0, nx - 1):
        for j in (0, ny - 1):
            for k in (0, nz - 1):
                p = half.image.TransformIndexToPhysicalPoint((i, j, k))
                pts.append(ref.TransformPhysicalPointToContinuousIndex(p))
    return np.asarray(pts)


def _same_lattice(a: Half, b: Half) -> bool:
    ds = np.abs(np.asarray(a.spacing_xyz) - np.asarray(b.spacing_xyz)).max()
    dd = np.abs(np.asarray(a.image.GetDirection()) - np.asarray(b.image.GetDirection())).max()
    return ds < SAME_TOL and dd < SAME_TOL


def stitch_pair(
    top_folder: Path | str,
    top_uid: str | None,
    bottom_folder: Path | str,
    bottom_uid: str | None,
    *,
    move_mm: tuple[float, float, float] = (0.0, 0.0, 0.0),
    placement: str = "whole_pixel",
    interpolator: str = "linear",
    trim_rows: int = 3,
    crop_pad_mm: float | None = None,
    halves: tuple[Half, Half] | None = None,
) -> ItkStitch:
    """Stitch TOP and BOTTOM using the header geometry.

    ``move_mm`` = shift of the bottom half from its header position, as
    (along the core, across the core, through the stack).
    ``placement`` ("whole_pixel" | "subpixel") only matters when both halves share a pixel
    lattice: "whole_pixel" snaps the shift to a whole pixel so every voxel stays original.
    """
    if placement not in PLACEMENTS:
        raise ValueError(f"placement must be one of {PLACEMENTS}")
    if interpolator not in INTERPOLATORS:
        raise ValueError(f"interpolator must be one of {tuple(INTERPOLATORS)}")
    top, bottom = halves if halves else (load_half("top", top_folder, top_uid), load_half("bottom", bottom_folder, bottom_uid))
    if not np.allclose(np.r_[top.row_dir, top.col_dir], np.r_[bottom.row_dir, bottom.col_dir], atol=SAME_TOL):
        raise ValueError("The two halves have different orientations; header placement does not apply")

    # reference = finer in-plane pixel (top wins a tie)
    pitch = lambda h: min(h.spacing_xyz[0], h.spacing_xyz[1])
    ref, mov = (top, bottom) if pitch(top) <= pitch(bottom) + SAME_TOL else (bottom, top)
    sign = 1.0 if mov is bottom else -1.0  # move_mm is for the bottom half
    move = sign * (move_mm[0] * ref.col_dir + move_mm[1] * ref.row_dir + move_mm[2] * ref.normal)

    rimg, mimg = ref.image, mov.image
    same = _same_lattice(ref, mov)
    snapped = False
    snap_residual_mm = 0.0
    if same and placement == "whole_pixel":
        c = np.asarray(rimg.TransformPhysicalPointToContinuousIndex(tuple(np.asarray(mimg.GetOrigin()) + move)))
        r = np.round(c)
        p0 = np.asarray(rimg.TransformContinuousIndexToPhysicalPoint(tuple(c)))
        p1 = np.asarray(rimg.TransformContinuousIndexToPhysicalPoint(tuple(r)))
        snap_residual_mm = float(np.linalg.norm(p1 - p0))
        move = move + (p1 - p0)
        snapped = True

    # output grid: the reference lattice, integer-extended to cover both halves
    allc = np.vstack([_corners_index(ref, rimg),
                      np.asarray([rimg.TransformPhysicalPointToContinuousIndex(tuple(np.asarray(mimg.TransformIndexToPhysicalPoint(ix)) + move))
                                  for ix in [(i, j, k) for i in (0, mimg.GetSize()[0] - 1) for j in (0, mimg.GetSize()[1] - 1) for k in (0, mimg.GetSize()[2] - 1)]])])
    # headers carry ~1e-5 mm rounding: do not let that add a whole extra slice or row
    lo = np.floor(allc.min(axis=0) + 1e-3).astype(int)
    hi = np.ceil(allc.max(axis=0) - 1e-3).astype(int)
    if crop_pad_mm is not None:  # keep every row and slice, but only the core's width (+ margin)
        ci = []
        for h, sh in ((ref, np.zeros(3)), (mov, move)):
            for col in h.core_cols:
                for row in (0, h.image.GetSize()[1] - 1):
                    p = np.asarray(h.image.TransformIndexToPhysicalPoint((int(col), int(row), 0))) + sh
                    ci.append(rimg.TransformPhysicalPointToContinuousIndex(tuple(p))[0])
        pad = crop_pad_mm / rimg.GetSpacing()[0]
        lo[0] = max(lo[0], int(np.floor(min(ci) - pad)))
        hi[0] = min(hi[0], int(np.ceil(max(ci) + pad)))
    size = (hi - lo + 1).tolist()
    grid = sitk.Image(size, sitk.sitkFloat32)
    grid.SetSpacing(rimg.GetSpacing())
    grid.SetDirection(rimg.GetDirection())
    grid.SetOrigin(rimg.TransformContinuousIndexToPhysicalPoint(tuple(float(x) for x in lo)))

    interp_hu = sitk.sitkNearestNeighbor if (snapped or interpolator == "nearest") else INTERPOLATORS[interpolator]
    air = ref.air

    def put(img: sitk.Image, shift: np.ndarray, interp: int) -> tuple[np.ndarray, np.ndarray]:
        tx = sitk.TranslationTransform(3, tuple(float(x) for x in -shift))  # output point -> source point
        hu = sitk.Resample(img, grid, tx, interp, float(air), sitk.sitkFloat32)
        ones = sitk.Image(img.GetSize(), sitk.sitkUInt8) + 1
        ones.CopyInformation(img)
        fp = sitk.Resample(ones, grid, tx, sitk.sitkNearestNeighbor, 0, sitk.sitkUInt8)
        return sitk.GetArrayFromImage(hu), sitk.GetArrayFromImage(fp).astype(bool)

    ref_vol, ref_fp = put(rimg, np.zeros(3), sitk.sitkNearestNeighbor)  # exact copy: same lattice
    mov_vol, mov_fp = put(mimg, move, interp_hu)

    # core masks (same recipe as the existing pipeline: threshold + largest component per slice)
    def core(vol: np.ndarray, fp: np.ndarray, h: Half) -> np.ndarray:
        out = np.zeros(vol.shape, bool)
        for z in range(vol.shape[0]):
            if fp[z].any():
                out[z] = _content_mask(vol[z], h.threshold, h.air) & fp[z]
        return out

    ref_core = core(ref_vol, ref_fp, ref)
    mov_core = core(mov_vol, mov_fp, mov)

    # fuse: mean in the interior of the overlap; edge rows of either half use the other half
    k = 2 * int(trim_rows) + 1
    st = np.ones((1, k, 1), bool)
    fr = binary_erosion(ref_fp, st) if trim_rows > 0 else ref_fp
    fm = binary_erosion(mov_fp, st) if trim_rows > 0 else mov_fp
    out = np.full(ref_vol.shape, air, np.float32)
    only_r = ref_core & ~mov_core
    only_m = mov_core & ~ref_core
    both = ref_core & mov_core
    out[only_r] = ref_vol[only_r]
    out[only_m] = mov_vol[only_m]
    mean = both & fr & fm
    out[mean] = 0.5 * (ref_vol[mean] + mov_vol[mean])
    a = both & fr & ~fm
    out[a] = ref_vol[a]
    b = both & ~fr & fm
    out[b] = mov_vol[b]
    c = both & ~fr & ~fm
    out[c] = ref_vol[c]

    ov_rows = np.where((ref_fp & mov_fp).any(axis=(0, 2)))[0]
    px = float(rimg.GetSpacing()[1])
    info = dict(
        reference=ref.name, moving=mov.name,
        ref_pixel_mm=float(rimg.GetSpacing()[0]), moving_pixel_mm=float(mimg.GetSpacing()[0]),
        upsample_factor_moving=float(mimg.GetSpacing()[0] / rimg.GetSpacing()[0]),
        same_lattice=bool(same), placement=placement, snapped_to_pixel=bool(snapped),
        snap_residual_mm=snap_residual_mm,
        interpolation="nearest (voxels on reference lattice)" if snapped else interpolator,
        move_mm_bottom=[float(x) for x in move_mm], applied_move_mm=[float(x) for x in move],
        output_size_xyz=[int(s) for s in size], output_offset_in_reference_index=[int(v) for v in lo], output_origin=[float(x) for x in grid.GetOrigin()],
        output_spacing=[float(x) for x in grid.GetSpacing()],
        overlap_rows=None if ov_rows.size == 0 else [int(ov_rows[0]), int(ov_rows[-1])],
        overlap_mm=0.0 if ov_rows.size == 0 else float((ov_rows[-1] - ov_rows[0] + 1) * px),
        trim_rows=int(trim_rows), crop_pad_mm=crop_pad_mm,
        sitk_version=sitk.Version_VersionString(),
    )
    return ItkStitch(volume=out, grid=grid, reference=ref, moving=mov, ref_vol=ref_vol, mov_vol=mov_vol,
                     ref_footprint=ref_fp, mov_footprint=mov_fp, ref_core=ref_core, mov_core=mov_core, info=info)


def image_based_move_mm(
    top_folder: Path | str,
    top_uid: str | None,
    bottom_folder: Path | str,
    bottom_uid: str | None,
) -> tuple[float, float, float]:
    """Fallback start for scans whose headers are way off.

    Runs the existing core-mask + overlap-profile alignment (``coarse_align``) and returns the
    shift of the *bottom* half from its header placement that reproduces that alignment, as
    (along the core, across the core, through the stack) in millimetres, ready for
    ``stitch_pair(move_mm=...)``.  Depth is left at 0: that alignment pairs slice k with slice k.
    """
    from core_stitch.pipeline import coarse_align

    st = coarse_align(top_folder, bottom_folder, top_uid, bottom_uid)
    heads = []
    for series in (st.fixed.series, st.mobile_base.series):
        ds = pydicom.dcmread(series.slice_paths[0], stop_before_pixels=True, force=True)
        iop = np.asarray(ds.ImageOrientationPatient, float)
        heads.append((iop[:3], iop[3:], np.asarray(ds.ImagePositionPatient, float)))
    row, col, ipp_a = heads[0]
    ipp_b = heads[1][2]
    delta = ipp_b - ipp_a
    mm = st.spacing_mm
    img_along = st.total_lag_px * mm
    img_across = (st.mobile_base.lateral_dcol + st.total_col_offset_px) * mm
    return (float(img_along - np.dot(delta, col)), float(img_across - np.dot(delta, row)), 0.0)
