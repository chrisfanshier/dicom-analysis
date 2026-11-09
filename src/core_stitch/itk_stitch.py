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
from core_stitch.load import MprSeries, load_series, turned_series
from core_stitch.pipeline import _content_mask
from core_stitch.segment import mid_slice, segment_core

PLACEMENTS = ("whole_pixel", "subpixel")
INTERPOLATORS = {"linear": sitk.sitkLinear, "bspline": sitk.sitkBSpline, "nearest": sitk.sitkNearestNeighbor}
SAME_TOL = 1e-3
ORIENT_MAX_DEG = 30.0  # halves whose image planes differ by more than this are not the same view of the core


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


@dataclass
class Plan:
    """Everything decided before any voxel is resampled (cheap to compute)."""

    top: Half
    bottom: Half
    ref: Half
    mov: Half
    move: np.ndarray  # shift of the moving half in physical space (mm), after any whole-pixel snap
    move_mm_bottom: tuple[float, float, float]
    placement: str
    same_lattice: bool
    snapped: bool
    snap_residual_mm: float
    snap_vec: np.ndarray  # physical vector the whole-pixel snap added to the shift (zero if none)
    lo: np.ndarray  # first output index, in reference-lattice units
    size: list[int]  # output size (x, y, z)
    crop_pad_mm: float | None
    orientation_diff_deg: float = 0.0  # angle between the two halves' image planes (0 for the usual identical pair)

    @property
    def n_slices(self) -> int:
        return int(self.size[2])


def _direction(row: np.ndarray, col: np.ndarray) -> tuple[np.ndarray, tuple[float, ...]]:
    normal = np.cross(row, col)
    d = np.column_stack([row, col, normal])
    return normal, tuple(float(x) for x in d.flatten())


def load_half(name: str, folder: Path | str, uid: str | None, *, turn180: bool = False) -> Half:
    """Load one half.  ``turn180=True`` turns every slice 180 degrees in place (for a half scanned upside down)."""
    series = load_series(folder, uid)
    if turn180:
        series = turned_series(series)
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


def orientation_difference_deg(a: Half, b: Half) -> float:
    """Largest angle between the two halves' row, column or normal directions, in degrees (0 = identical planes)."""
    if np.allclose(np.r_[a.row_dir, a.col_dir], np.r_[b.row_dir, b.col_dir], atol=SAME_TOL):
        return 0.0
    angs = [np.degrees(np.arccos(np.clip(float(np.dot(u, v)), -1.0, 1.0)))
            for u, v in ((a.row_dir, b.row_dir), (a.col_dir, b.col_dir), (a.normal, b.normal))]
    return float(max(angs))


def plan_stitch(
    top: Half,
    bottom: Half,
    *,
    move_mm: tuple[float, float, float] = (0.0, 0.0, 0.0),
    placement: str = "whole_pixel",
    crop_pad_mm: float | None = None,
) -> Plan:
    """Choose the reference, apply the optional shift and size the output grid (no resampling).

    ``move_mm`` = shift of the bottom half from its header position, as
    (along the core, across the core, through the stack).
    ``placement`` ("whole_pixel" | "subpixel") only matters when both halves share a pixel
    lattice: "whole_pixel" snaps the shift to a whole pixel so every voxel stays original.
    """
    if placement not in PLACEMENTS:
        raise ValueError(f"placement must be one of {PLACEMENTS}")
    orient_deg = orientation_difference_deg(top, bottom)
    if orient_deg > ORIENT_MAX_DEG:
        raise ValueError(
            f"The two halves are cut in planes {orient_deg:.0f} degrees apart, so they are not the same view of the "
            "core (for example coronal against sagittal) and cannot be joined by their headers."
        )
    # reference = finer in-plane pixel (top wins a tie)
    pitch = lambda h: min(h.spacing_xyz[0], h.spacing_xyz[1])
    ref, mov = (top, bottom) if pitch(top) <= pitch(bottom) + SAME_TOL else (bottom, top)
    sign = 1.0 if mov is bottom else -1.0  # move_mm is for the bottom half
    move = sign * (move_mm[0] * ref.col_dir + move_mm[1] * ref.row_dir + move_mm[2] * ref.normal)

    rimg, mimg = ref.image, mov.image
    same = _same_lattice(ref, mov)
    snapped = False
    snap_residual_mm = 0.0
    snap_vec = np.zeros(3)
    if same and placement == "whole_pixel":
        c = np.asarray(rimg.TransformPhysicalPointToContinuousIndex(tuple(np.asarray(mimg.GetOrigin()) + move)))
        r = np.round(c)
        p0 = np.asarray(rimg.TransformContinuousIndexToPhysicalPoint(tuple(c)))
        p1 = np.asarray(rimg.TransformContinuousIndexToPhysicalPoint(tuple(r)))
        snap_residual_mm = float(np.linalg.norm(p1 - p0))
        snap_vec = p1 - p0
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
    return Plan(top=top, bottom=bottom, ref=ref, mov=mov, move=move, move_mm_bottom=tuple(float(x) for x in move_mm),
                placement=placement, same_lattice=bool(same), snapped=bool(snapped), snap_residual_mm=snap_residual_mm,
                snap_vec=snap_vec, lo=lo, size=size, crop_pad_mm=crop_pad_mm, orientation_diff_deg=orient_deg)


def _core_slice(image: np.ndarray, threshold: float, air: float) -> np.ndarray:
    """Core = pixels above the half's threshold, largest connected piece.

    Unlike the older ``_content_mask`` this never flips to "below threshold" when most of the slice is
    bright: the output is cropped to the core's width, so a wide core can fill over 55% of the picture,
    and that flip then made the mask the air around it (the core went black, depending on small shifts).
    """
    from core_stitch.segment import largest_component

    thr = threshold if np.isfinite(threshold) else air + 50.0
    raw = np.isfinite(image) & (image >= thr)
    res = largest_component(raw)
    if res.n_components == 0:
        return np.isfinite(image) & (image > air + 30.0)
    return res.mask


def render_plan(
    plan: Plan,
    *,
    interpolator: str = "linear",
    trim_rows: int = 3,
    z_keep: tuple[int, int] | None = None,
    y_keep: tuple[int, int] | None = None,
    threshold_masks: bool = False,
) -> ItkStitch:
    """Resample both halves onto the planned grid and fuse them.

    ``z_keep=(first, last)`` renders only those output slices (fast preview of one slice).
    ``y_keep=(first, last)`` renders only those output rows (a band, or a single row for a cross-section).
    ``threshold_masks`` marks the core by a plain threshold instead of the per-slice largest component,
    which is what a one-row render needs.
    """
    if interpolator not in INTERPOLATORS:
        raise ValueError(f"interpolator must be one of {tuple(INTERPOLATORS)}")
    ref, mov, move = plan.ref, plan.mov, plan.move
    rimg, mimg = ref.image, mov.image
    placement, same, snapped = plan.placement, plan.same_lattice, plan.snapped
    snap_residual_mm, move_mm, crop_pad_mm = plan.snap_residual_mm, plan.move_mm_bottom, plan.crop_pad_mm
    lo = plan.lo.copy()
    size = list(plan.size)
    if z_keep is not None:
        z0 = int(np.clip(z_keep[0], 0, plan.n_slices - 1))
        z1 = int(np.clip(z_keep[1], z0, plan.n_slices - 1))
        lo[2] = lo[2] + z0
        size[2] = z1 - z0 + 1
    row_offset = 0
    if y_keep is not None:
        y0 = int(np.clip(y_keep[0], 0, plan.size[1] - 1))
        y1 = int(np.clip(y_keep[1], y0, plan.size[1] - 1))
        lo[1] = lo[1] + y0
        size[1] = y1 - y0 + 1
        row_offset = y0
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
        if threshold_masks:
            return (vol > h.threshold) & fp
        out = np.zeros(vol.shape, bool)
        for z in range(vol.shape[0]):
            if fp[z].any():
                out[z] = _core_slice(vol[z], h.threshold, h.air) & fp[z]
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
        orientation_diff_deg=float(plan.orientation_diff_deg),
        snap_residual_mm=snap_residual_mm,
        interpolation="nearest (voxels on reference lattice)" if snapped else interpolator,
        move_mm_bottom=[float(x) for x in move_mm], applied_move_mm=[float(x) for x in move],
        output_size_xyz=[int(s) for s in size], output_offset_in_reference_index=[int(v) for v in lo], output_origin=[float(x) for x in grid.GetOrigin()],
        output_spacing=[float(x) for x in grid.GetSpacing()],
        overlap_rows=None if ov_rows.size == 0 else [int(ov_rows[0]), int(ov_rows[-1])],
        overlap_mm=0.0 if ov_rows.size == 0 else float((ov_rows[-1] - ov_rows[0] + 1) * px),
        output_row_offset=int(row_offset),
        trim_rows=int(trim_rows), crop_pad_mm=crop_pad_mm,
        sitk_version=sitk.Version_VersionString(),
    )
    return ItkStitch(volume=out, grid=grid, reference=ref, moving=mov, ref_vol=ref_vol, mov_vol=mov_vol,
                     ref_footprint=ref_fp, mov_footprint=mov_fp, ref_core=ref_core, mov_core=mov_core, info=info)


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
    z_keep: tuple[int, int] | None = None,
) -> ItkStitch:
    """Stitch TOP and BOTTOM using the header geometry (plan, then render)."""
    top, bottom = halves if halves else (load_half("top", top_folder, top_uid), load_half("bottom", bottom_folder, bottom_uid))
    plan = plan_stitch(top, bottom, move_mm=move_mm, placement=placement, crop_pad_mm=crop_pad_mm)
    return render_plan(plan, interpolator=interpolator, trim_rows=trim_rows, z_keep=z_keep)


def image_based_move_mm(
    top_folder: Path | str,
    top_uid: str | None,
    bottom_folder: Path | str,
    bottom_uid: str | None,
    require_below: bool = False,
    turn_top: bool = False,
    turn_bottom: bool = False,
) -> tuple[float, float, float]:
    """Fallback start for scans whose headers are way off.

    ``require_below=True`` forces the bottom half below the top half (the order is not searched).


    Runs the existing core-mask + overlap-profile alignment (``coarse_align``) and returns the
    shift of the *bottom* half from its header placement that reproduces that alignment, as
    (along the core, across the core, through the stack) in millimetres, ready for
    ``stitch_pair(move_mm=...)``.  Depth is left at 0: that alignment pairs slice k with slice k.
    """
    from core_stitch.pipeline import coarse_align

    st = coarse_align(top_folder, bottom_folder, top_uid, bottom_uid, require_below=require_below,
                      turn_a=turn_top, turn_b=turn_bottom)
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


@dataclass
class RefineOutcome:
    """Result of ``refine_translation``; shifts are for the BOTTOM half, in mm."""

    along_mm: float
    across_mm: float
    score_before: float  # correlation over the overlap core, before / after
    score_after: float
    applied: bool
    note: str
    iterations: int = 0


def _local(img: sitk.Image, direction: np.ndarray, origin: np.ndarray) -> sitk.Image:
    """Same voxels, re-expressed in the reference frame: axes = (across, along, depth).

    A translation found in this frame is directly (across, along, depth) in mm.  The image keeps its own
    orientation relative to the reference (identity for the reference itself; a small rotation for a half that was
    cut in a plane a few degrees away), so the registration sees exactly the placement the stitch uses.
    """
    out = sitk.Image(img)
    d_img = np.asarray(img.GetDirection(), float).reshape(3, 3)
    out.SetDirection(tuple(float(x) for x in (direction.T @ d_img).flatten()))
    out.SetOrigin(tuple(float(x) for x in direction.T @ (np.asarray(img.GetOrigin()) - origin)))
    return out


def refine_translation(
    plan: Plan,
    *,
    max_shift_mm: float = 15.0,
    slab: int = 24,
    margin_rows: int = 20,
    edge_rows: int = 4,
) -> RefineOutcome:
    """Fine-tune the bottom half's position by intensity registration (SimpleITK).

    A rigid translation along and across the core (depth is left alone) that maximises the
    correlation of the two halves over the overlap core, starting from the current placement.
    Metric: correlation (insensitive to overall brightness differences); linear interpolation;
    two resolution levels; regular-step gradient descent.  The shift is applied only if it improves
    the score and is no larger than ``max_shift_mm``.
    """
    mid = plan.n_slices // 2
    z0 = max(0, mid - slab // 2)
    z1 = min(plan.n_slices - 1, z0 + slab - 1)
    res = render_plan(plan, trim_rows=0, z_keep=(z0, z1))
    both = res.ref_core & res.mov_core & res.ref_footprint & res.mov_footprint
    rows = np.where(both.any(axis=(0, 2)))[0]
    if rows.size < 2 * edge_rows + 8:
        return RefineOutcome(0.0, 0.0, float("nan"), float("nan"), False, "The halves barely overlap, so there is nothing to refine on.")
    r0 = max(int(rows.min()) - margin_rows, 0)
    r1 = min(int(rows.max()) + margin_rows + 1, both.shape[1])
    mask = binary_erosion(both[:, r0:r1, :], np.ones((1, 2 * edge_rows + 1, 1), bool))
    if mask.sum() < 500:
        return RefineOutcome(0.0, 0.0, float("nan"), float("nan"), False, "Too little overlap core to register on.")

    direction = np.column_stack([plan.ref.row_dir, plan.ref.col_dir, plan.ref.normal])
    o_ref = np.asarray(res.grid.GetOrigin())
    fixed = sitk.GetImageFromArray(np.ascontiguousarray(res.ref_vol[:, r0:r1, :], dtype=np.float32))
    fixed.SetSpacing(res.grid.GetSpacing())
    fixed.SetDirection(res.grid.GetDirection())
    fixed.SetOrigin(res.grid.TransformIndexToPhysicalPoint((0, r0, 0)))
    m = sitk.GetImageFromArray(mask.astype(np.uint8))
    m.CopyInformation(fixed)
    moving = sitk.Image(plan.mov.image)
    moving.SetOrigin(tuple(float(x) for x in np.asarray(moving.GetOrigin()) + plan.move))  # current placement baked in

    fixed_l, mask_l = _local(fixed, direction, o_ref), _local(m, direction, o_ref)
    moving_l = _local(moving, direction, o_ref)

    reg = sitk.ImageRegistrationMethod()
    reg.SetMetricAsCorrelation()
    reg.SetMetricFixedMask(mask_l)
    reg.SetMetricSamplingStrategy(reg.REGULAR)
    reg.SetMetricSamplingPercentage(0.3, 1)
    reg.SetInterpolator(sitk.sitkLinear)
    reg.SetOptimizerAsRegularStepGradientDescent(
        learningRate=2.0, minStep=1e-3, numberOfIterations=120, relaxationFactor=0.5, gradientMagnitudeTolerance=1e-7
    )
    reg.SetOptimizerScalesFromPhysicalShift()
    reg.SetOptimizerWeights([1.0, 1.0, 0.0])  # across, along; depth stays where you put it
    reg.SetShrinkFactorsPerLevel([2, 1])
    reg.SetSmoothingSigmasPerLevel([1.0, 0.0])
    reg.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    start = sitk.TranslationTransform(3)
    reg.SetInitialTransform(start, inPlace=False)
    tx = reg.Execute(fixed_l, moving_l)

    # Score both positions the same way: full resolution, the same pixels, and only pixels that
    # lie inside the moving image at BOTH positions (so a shift cannot be penalised by air at an edge).
    fixed_arr = sitk.GetArrayFromImage(fixed_l)
    mask_arr = sitk.GetArrayFromImage(mask_l) > 0
    ones = sitk.Image(moving_l.GetSize(), sitk.sitkUInt8) + 1
    ones.CopyInformation(moving_l)

    def placed(offset):
        t = sitk.TranslationTransform(3, tuple(float(x) for x in offset))
        r = sitk.Resample(moving_l, fixed_l, t, sitk.sitkLinear, float(plan.ref.air), sitk.sitkFloat32)
        v = sitk.Resample(ones, fixed_l, t, sitk.sitkNearestNeighbor, 0, sitk.sitkUInt8)
        return sitk.GetArrayFromImage(r), sitk.GetArrayFromImage(v) > 0

    def ncc(a, b, m) -> float:
        a, b = a[m].astype(np.float64), b[m].astype(np.float64)
        a = (a - a.mean()) / (a.std() + 1e-9)
        b = (b - b.mean()) / (b.std() + 1e-9)
        return float((a * b).mean())

    o = np.asarray(tx.GetParameters(), float)  # fixed point p corresponds to moving point p + o
    (m0, v0), (m1, v1) = placed((0.0, 0.0, 0.0)), placed(o)
    common = mask_arr & v0 & v1
    before, after = ncc(fixed_arr, m0, common), ncc(fixed_arr, m1, common)
    content = -o  # so the moving half's content must move by -o  (across, along, depth)
    sign = 1.0 if plan.mov is plan.bottom else -1.0
    along, across = float(sign * content[1]), float(sign * content[0])
    iters = int(reg.GetOptimizerIteration())
    if max(abs(along), abs(across)) > max_shift_mm:
        return RefineOutcome(0.0, 0.0, before, after, False,
                             f"Refine wanted to move the bottom half {along:+.1f} mm along / {across:+.1f} mm across, "
                             f"more than the {max_shift_mm:.0f} mm limit, so nothing was changed.", iters)
    if not (after > before + 1e-4):
        return RefineOutcome(0.0, 0.0, before, after, False, "Refine found no better position; nothing was changed.", iters)
    return RefineOutcome(along, across, before, after, True, "", iters)


def join_quality(
    res: ItkStitch,
    *,
    edge_rows: int = 4,
    max_px: int = 3,
    max_slices: int = 3,
) -> dict:
    """How well do the two halves agree over the overlap, and is any other position clearly better?

    Works on any render that contains the overlap (a row band is enough).  It reports

    * the correlation of the two halves over the overlap core at the placement as rendered, and
    * the same correlation with the bottom half's voxels shifted by up to ``max_px`` pixels along the
      core, ``max_px`` across it, and ``max_slices`` slices in depth (one axis at a time).

    At a good join the best shift is 0 on every axis.  An axis is flagged when the best shift is at least
    2 pixels (1 slice in depth) away and beats the current placement by more than a per-axis margin.  The
    margins differ because the axes differ in sensitivity: on real cores one slice of depth error costs
    ~0.05 in correlation, a pixel across the core ~0.005, a pixel along the core only ~0.0002 (layering
    is smooth down the core).  A shift of one pixel is never flagged: a placement that is right to
    sub-pixel precision often has its best whole-pixel shift at 0 or 1.  Only whole voxels are tried, so
    this is a check, not a refinement.
    """
    keep = edge_rows + max_px
    both_all = res.ref_core & res.mov_core & res.ref_footprint & res.mov_footprint
    rows_all = np.where(both_all.any(axis=(0, 2)))[0]
    if rows_all.size < 2 * keep + 8:
        return {"evaluated": False, "reason": "too little overlap to evaluate"}
    # work on the overlap band only (a full volume has tens of millions of voxels)
    r0 = max(int(rows_all.min()) - keep - 2, 0)
    r1 = min(int(rows_all.max()) + keep + 3, both_all.shape[1])
    band = np.s_[:, r0:r1, :]
    ref_vol, mov_vol = res.ref_vol[band], res.mov_vol[band]
    mr = (res.ref_core & res.ref_footprint)[band]
    mm = (res.mov_core & res.mov_footprint)[band]
    both = both_all[band]
    rows = np.where(both.any(axis=(0, 2)))[0]
    window = np.zeros(both.shape, bool)  # interior rows of the overlap, away from either half's edge
    window[:, rows.min() + keep : rows.max() - keep + 1, :] = True

    def shifted(a: np.ndarray, axis: int, s: int) -> np.ndarray:
        out = np.zeros_like(a)
        src = [slice(None)] * 3
        dst = [slice(None)] * 3
        if s > 0:
            src[axis], dst[axis] = slice(None, -s), slice(s, None)
        elif s < 0:
            src[axis], dst[axis] = slice(-s, None), slice(None, s)
        else:
            return a
        out[tuple(dst)] = a[tuple(src)]
        return out

    def ncc(axis: int, s: int) -> float | None:
        m = window & mr & shifted(mm, axis, s)
        if m.sum() < 500:
            return None
        a = ref_vol[m].astype(np.float64)
        b = shifted(mov_vol, axis, s)[m].astype(np.float64)
        a = (a - a.mean()) / (a.std() + 1e-9)
        b = (b - b.mean()) / (b.std() + 1e-9)
        return float((a * b).mean())

    base = ncc(0, 0)
    if base is None:
        return {"evaluated": False, "reason": "too little overlap core to evaluate"}
    voxels = int((window & mr & mm).sum())
    names = {0: "depth_slices", 1: "along_px", 2: "across_px"}
    limit = {0: max_slices, 1: max_px, 2: max_px}
    min_gain = {"depth_slices": 0.005, "along_px": 1e-4, "across_px": 1e-3}
    min_shift = {"depth_slices": 1, "along_px": 2, "across_px": 2}
    best_shift, best_corr, scans, flags = {}, {}, {}, []
    for axis in (0, 1, 2):
        vals = {s: ncc(axis, s) for s in range(-limit[axis], limit[axis] + 1)}
        vals = {s: v for s, v in vals.items() if v is not None}
        b = max(vals, key=vals.get)
        best_shift[names[axis]], best_corr[names[axis]] = int(b), round(vals[b], 4)
        scans[names[axis]] = {str(s): round(v, 4) for s, v in sorted(vals.items())}
        if abs(b) >= min_shift[names[axis]] and vals[b] - base > min_gain[names[axis]]:
            px = float(res.grid.GetSpacing()[0])
            dz = float(res.grid.GetSpacing()[2])
            what = {"depth_slices": f"{b:+d} slices ({b * dz:+.1f} mm) in depth",
                    "along_px": f"{b:+d} px ({b * px:+.2f} mm) along the core",
                    "across_px": f"{b:+d} px ({b * px:+.2f} mm) across the core"}[names[axis]]
            flags.append(f"the halves agree better if the bottom half is moved {what} (correlation {base:.3f} -> {vals[b]:.3f})")
    return {
        "evaluated": True,
        "correlation": round(base, 4),
        "voxels": voxels,
        "slices": int(both.shape[0]),
        "best_shift": best_shift,
        "correlation_at_best_shift": best_corr,
        "scan": scans,
        "consistent": not flags,
        "flags": flags,
        "note": "correlation of the two halves over the overlap core; shifts are whole voxels of the output grid, one axis at a time",
    }
