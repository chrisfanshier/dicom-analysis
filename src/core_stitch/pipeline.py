"""Orchestrate Coron TOP/BTM coarse align → flip/roll → overlap → fuse."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from skimage.transform import resize

from core_stitch.geometry import (
    CoreGeometry,
    air_value,
    core_geometry,
    resample_mask_to_mm,
    resample_to_mm,
    rotate_image,
    rotate_mask,
    shift_image,
)
from core_stitch.load import MprSeries, load_series
from core_stitch.overlap import (
    OverlapResult,
    alignment_rgb,
    apply_flip,
    apply_roll,
    compose_canvases,
    fuse_preview,
    lateral_align_to,
    search_overlap,
    upright_angle,
)
from core_stitch.analyze import (
    DowncoreProfile,
    SedimentRoi,
    build_sediment_mask_2d,
    build_sediment_mask_3d,
    detect_sediment_from_profile,
    mean_hu_per_row,
    overlay_sediment_circle_xz,
    overlay_sediment_rgb,
    render_downcore_plot_rgb,
)
from core_stitch.refine import RefineResult, _shift_stack_z, refine_join_volumes
from core_stitch.segment import mid_slice, segment_core


@dataclass
class PosedVolumes:
    """Cached upright/flip/roll stacks (nudge_z applied at compose time)."""

    vol_a: np.ndarray
    mask_a: np.ndarray
    vol_b: np.ndarray
    mask_b: np.ndarray
    z_mm: float

    @property
    def n_slices(self) -> int:
        return int(self.vol_a.shape[0])


@dataclass
class HalfPrepared:
    series: MprSeries
    image: np.ndarray  # upright, laterally roughly placed, float HU
    mask: np.ndarray
    geom: CoreGeometry
    target_mm: float
    air: float
    upright_deg: float = 0.0  # CCW rotation applied to make long axis +row
    lateral_dcol: float = 0.0  # column shift after upright (px)
    seg_threshold: float = 0.0  # Otsu thr on native mid-slice (HU)


@dataclass
class StitchState:
    fixed: HalfPrepared
    mobile_base: HalfPrepared  # after upright + lateral, before flip/roll/lag
    flip: bool = False
    roll_deg: float = 0.0
    # Manual / refine nudge of mobile (B) on top of auto overlap (mm)
    nudge_x_mm: float = 0.0  # +X = columns right
    nudge_y_mm: float = 0.0  # +Y = rows down (along upright core axis)
    nudge_z_mm: float = 0.0  # +Z = B toward higher stack index
    # Overlap pixel combine: max | mean | blend | feather | min | a | b
    fuse_mode: str = "max"
    match_hu: bool = False
    # Greyscale display/export window (HU); None → auto 1st–99th percentile
    window_lo: float | None = None
    window_hi: float | None = None
    overlap: OverlapResult | None = None
    refine: RefineResult | None = None
    posed: PosedVolumes | None = None
    view_z: int | None = None  # None → mid-slice 2D path; int → Z from posed stack
    canvas_a: np.ndarray | None = None
    canvas_b: np.ndarray | None = None
    mask_a: np.ndarray | None = None
    mask_b: np.ndarray | None = None
    sediment: SedimentRoi | None = None
    show_sediment: bool = True
    show_downcore_plot: bool = False
    # Cached fused XZ plane for compact QC panel: key, HU (Z,W), mask (Z,W)
    _xz_cache_key: tuple | None = None
    _xz_hu: np.ndarray | None = None
    _xz_mask: np.ndarray | None = None
    # Cached downcore mean HU inside sediment cylinder
    _downcore_cache_key: tuple | None = None
    _downcore_profile: DowncoreProfile | None = None
    warnings: list[str] = field(default_factory=list)

    @property
    def low_confidence(self) -> bool:
        return bool(self.overlap and self.overlap.low_confidence)

    @property
    def spacing_mm(self) -> float:
        return self.fixed.target_mm

    @property
    def total_lag_px(self) -> int:
        base = 0 if self.overlap is None else self.overlap.offset_px
        return int(base + round(self.nudge_y_mm / self.spacing_mm))

    @property
    def total_col_offset_px(self) -> int:
        return int(round(self.nudge_x_mm / self.spacing_mm))


def _prepare_half(series: MprSeries, target_mm: float) -> tuple[HalfPrepared, list[str]]:
    warns: list[str] = []
    img0 = mid_slice(series.volume)
    seg = segment_core(img0)
    if seg.n_components == 0:
        raise ValueError(f"No core mask found in {series.path}")
    if seg.ambiguous:
        warns.append(f"{series.path.name}: secondary component nearly as large as core")

    _z, row_mm, col_mm = series.spacing_mm
    image = resample_to_mm(img0, (row_mm, col_mm), target_mm)
    mask = resample_mask_to_mm(seg.mask, (row_mm, col_mm), target_mm)
    geom = core_geometry(mask, target_mm)

    # Rotate upright (long axis → +row)
    ang = upright_angle(geom)
    air = air_value(image)
    image = rotate_image(image, ang, cval=air, order=1)
    mask = rotate_mask(mask, ang)
    geom = core_geometry(mask, target_mm)

    return (
        HalfPrepared(
            series=series,
            image=image,
            mask=mask,
            geom=geom,
            target_mm=target_mm,
            air=air,
            upright_deg=ang,
            lateral_dcol=0.0,
            seg_threshold=float(seg.threshold),
        ),
        warns,
    )


def coarse_align(path_a: Path | str, path_b: Path | str) -> StitchState:
    """Load A/B, resample, segment, upright, lateral-align B to A, run overlap."""
    sa = load_series(path_a)
    sb = load_series(path_b)
    target = float(
        min(sa.spacing_mm[1], sa.spacing_mm[2], sb.spacing_mm[1], sb.spacing_mm[2])
    )

    fixed, wa = _prepare_half(sa, target)
    mobile, wb = _prepare_half(sb, target)
    warns = wa + wb

    # Lateral align mobile centroid column to fixed
    dcol = float(fixed.geom.centroid_rc[1] - mobile.geom.centroid_rc[1])
    mob_img, mob_mask = lateral_align_to(
        mobile.image, mobile.mask, mobile.geom, fixed.geom.centroid_rc[1]
    )
    mob_geom = core_geometry(mob_mask, target)
    mobile_base = HalfPrepared(
        series=sb,
        image=mob_img,
        mask=mob_mask,
        geom=mob_geom,
        target_mm=target,
        air=air_value(mob_img),
        upright_deg=mobile.upright_deg,
        lateral_dcol=dcol,
        seg_threshold=mobile.seg_threshold,
    )

    state = StitchState(fixed=fixed, mobile_base=mobile_base, warnings=warns)
    recompute_overlap(state)
    return state


def _mobile_posed(state: StitchState) -> tuple[np.ndarray, np.ndarray]:
    img = state.mobile_base.image
    mask = state.mobile_base.mask
    if state.flip:
        img, mask = apply_flip(img, mask)
    if abs(state.roll_deg) > 1e-6:
        img, mask = apply_roll(img, mask, state.roll_deg, cval=state.mobile_base.air)
        # keep laterally centered after roll
        g = core_geometry(mask, state.spacing_mm)
        img, mask = lateral_align_to(
            img, mask, g, state.fixed.geom.centroid_rc[1]
        )
    return img, mask


def invalidate_posed(state: StitchState) -> None:
    state.posed = None
    state._xz_cache_key = None
    state._xz_hu = None
    state._xz_mask = None
    state._downcore_cache_key = None
    state._downcore_profile = None


def invalidate_downcore_profile(state: StitchState) -> None:
    state._downcore_cache_key = None
    state._downcore_profile = None


def prepare_posed_volumes(
    state: StitchState,
    *,
    apply_nudge_z: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Upright / flip / roll / Z-resample both halves.

    If apply_nudge_z, shift B along Z by state.nudge_z_mm.
    Returns vol_a, mask_a, vol_b, mask_b, z_spacing_mm (same Z length, center-cropped).
    """
    za = float(state.fixed.series.spacing_mm[0])
    zb = float(state.mobile_base.series.spacing_mm[0])
    z_mm = float(min(za, zb))

    vol_a, mask_a = _transform_half_volume(state.fixed)
    vol_b, mask_b = _transform_half_volume(
        state.mobile_base,
        flip=state.flip,
        roll_deg=state.roll_deg,
        relateral_to_col=(
            state.fixed.geom.centroid_rc[1] if abs(state.roll_deg) > 1e-6 else None
        ),
    )

    vol_a = _resample_volume_z(vol_a, za, z_mm, state.fixed.air)
    mask_a = _resample_mask_z(mask_a, za, z_mm)
    vol_b = _resample_volume_z(vol_b, zb, z_mm, state.mobile_base.air)
    mask_b = _resample_mask_z(mask_b, zb, z_mm)

    nz = min(vol_a.shape[0], vol_b.shape[0])
    vol_a = _center_crop_z(vol_a, nz)
    mask_a = _center_crop_z(mask_a, nz)
    vol_b = _center_crop_z(vol_b, nz)
    mask_b = _center_crop_z(mask_b, nz)

    if apply_nudge_z:
        dz_px = int(round(state.nudge_z_mm / max(z_mm, 1e-6)))
        if dz_px != 0:
            vol_b, mask_b = _shift_stack_z(
                vol_b, mask_b, dz_px, cval=state.mobile_base.air
            )

    return vol_a, mask_a, vol_b, mask_b, z_mm


def ensure_posed(state: StitchState) -> PosedVolumes:
    """Cache posed stacks without nudge_z (applied when composing a Z)."""
    if state.posed is None:
        vol_a, mask_a, vol_b, mask_b, z_mm = prepare_posed_volumes(
            state, apply_nudge_z=False
        )
        state.posed = PosedVolumes(
            vol_a=vol_a,
            mask_a=mask_a,
            vol_b=vol_b,
            mask_b=mask_b,
            z_mm=z_mm,
        )
    return state.posed


def recompose_at_z(state: StitchState, z: int) -> None:
    """Compose one Z from the posed cache onto preview canvases."""
    posed = ensure_posed(state)
    z = int(np.clip(z, 0, posed.n_slices - 1))
    state.view_z = z

    vol_b = posed.vol_b
    mask_b = posed.mask_b
    dz_px = int(round(state.nudge_z_mm / max(posed.z_mm, 1e-6)))
    if dz_px != 0:
        vol_b, mask_b = _shift_stack_z(
            vol_b, mask_b, dz_px, cval=state.mobile_base.air
        )

    ca, cb, ma, mb = compose_canvases(
        posed.vol_a[z],
        posed.mask_a[z],
        vol_b[z],
        mask_b[z],
        state.total_lag_px,
        col_offset_px=state.total_col_offset_px,
        cval=state.fixed.air,
    )
    state.canvas_a = ca
    state.canvas_b = cb
    state.mask_a = ma
    state.mask_b = mb
    if state.sediment is not None:
        # center_z stays at Auto-detect (cylinder fixed while browsing Z)
        rebuild_sediment_mask_2d(state)


def recompose(state: StitchState) -> None:
    """Rebuild canvases from current pose + auto lag + manual nudge (no re-search)."""
    if state.view_z is not None:
        recompose_at_z(state, state.view_z)
        return
    img_b, mask_b = _mobile_posed(state)
    lag = state.total_lag_px
    col = state.total_col_offset_px
    ca, cb, ma, mb = compose_canvases(
        state.fixed.image,
        state.fixed.mask,
        img_b,
        mask_b,
        lag,
        col_offset_px=col,
        cval=state.fixed.air,
    )
    state.canvas_a = ca
    state.canvas_b = cb
    state.mask_a = ma
    state.mask_b = mb
    if state.sediment is not None:
        rebuild_sediment_mask_2d(state)


def recompute_overlap(state: StitchState, *, reset_nudge: bool = True) -> OverlapResult:
    img_b, mask_b = _mobile_posed(state)
    result = search_overlap(
        state.fixed.image,
        state.fixed.mask,
        img_b,
        mask_b,
        spacing_mm=state.spacing_mm,
    )
    state.overlap = result
    if reset_nudge:
        state.nudge_x_mm = 0.0
        state.nudge_y_mm = 0.0
        state.nudge_z_mm = 0.0
        state.refine = None
    invalidate_posed(state)
    # Keep view_z if set so slice browser stays on the same index after flip/roll
    recompose(state)
    return result


def set_flip(state: StitchState, flip: bool) -> None:
    state.flip = bool(flip)
    recompute_overlap(state, reset_nudge=True)


def set_roll(state: StitchState, roll_deg: float) -> None:
    state.roll_deg = float(roll_deg)
    recompute_overlap(state, reset_nudge=True)


def set_nudge(
    state: StitchState,
    *,
    x_mm: float | None = None,
    y_mm: float | None = None,
    z_mm: float | None = None,
) -> None:
    if x_mm is not None:
        state.nudge_x_mm = float(x_mm)
    if y_mm is not None:
        state.nudge_y_mm = float(y_mm)
    if z_mm is not None:
        state.nudge_z_mm = float(z_mm)
    recompose(state)


def set_view_z(state: StitchState, z: int) -> None:
    """Switch preview to a Z index from the posed Coron stack."""
    recompose_at_z(state, int(z))


def core_crop_box(
    mask_a: np.ndarray,
    mask_b: np.ndarray,
    *,
    pad_px: int = 8,
) -> tuple[int, int, int, int]:
    """Tight bbox (r0, r1, c0, c1) around mask union, with padding."""
    union = mask_a | mask_b
    rows = np.where(union.any(axis=1))[0]
    cols = np.where(union.any(axis=0))[0]
    if rows.size == 0 or cols.size == 0:
        h, w = mask_a.shape
        return 0, h, 0, w
    r0 = max(0, int(rows[0]) - pad_px)
    r1 = min(mask_a.shape[0], int(rows[-1]) + 1 + pad_px)
    c0 = max(0, int(cols[0]) - pad_px)
    c1 = min(mask_a.shape[1], int(cols[-1]) + 1 + pad_px)
    return r0, r1, c0, c1


def crop_to_box(arr: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    r0, r1, c0, c1 = box
    if arr.ndim == 2:
        return arr[r0:r1, c0:c1]
    if arr.ndim == 3 and arr.shape[-1] in (3, 4):
        return arr[r0:r1, c0:c1, ...]
    raise ValueError(f"Unsupported array shape for crop: {arr.shape}")


def masked_hu_values(state: StitchState) -> np.ndarray:
    """HU samples for histogram / auto-window (union of A/B masks on canvases)."""
    assert state.canvas_a is not None and state.canvas_b is not None
    assert state.mask_a is not None and state.mask_b is not None
    return np.concatenate(
        [
            state.canvas_a[state.mask_a],
            state.canvas_b[state.mask_b],
        ]
    )


def auto_window_limits(state: StitchState) -> tuple[float, float]:
    vals = masked_hu_values(state)
    if vals.size == 0:
        return 0.0, 1.0
    return float(np.percentile(vals, 1)), float(np.percentile(vals, 99))


def window_limits(state: StitchState) -> tuple[float, float]:
    if state.window_lo is not None and state.window_hi is not None:
        lo, hi = float(state.window_lo), float(state.window_hi)
        if hi <= lo:
            hi = lo + 1.0
        return lo, hi
    return auto_window_limits(state)


def set_auto_window(state: StitchState) -> tuple[float, float]:
    lo, hi = auto_window_limits(state)
    state.window_lo = lo
    state.window_hi = hi
    return lo, hi


def fused_hu(state: StitchState) -> np.ndarray:
    """Current-slice fused HU (full canvas, fuse_mode / match_hu applied)."""
    assert state.canvas_a is not None and state.canvas_b is not None
    assert state.mask_a is not None and state.mask_b is not None
    return fuse_preview(
        state.canvas_a,
        state.canvas_b,
        state.mask_a,
        state.mask_b,
        opacity_a=1.0,
        opacity_b=1.0,
        mode=state.fuse_mode,
        feather_mm=max(4.0, 12.0 * state.spacing_mm),
        spacing_mm=state.spacing_mm,
        match_hu=state.match_hu,
    )


def core_union_mask(state: StitchState) -> np.ndarray:
    assert state.mask_a is not None and state.mask_b is not None
    return state.mask_a | state.mask_b


def rebuild_sediment_mask_2d(state: StitchState) -> None:
    """Refresh sediment.mask_2d = cylinder chord at current view Z."""
    if state.sediment is None or state.mask_a is None:
        return
    sed = state.sediment
    z_mm = state.posed.z_mm if state.posed is not None else None
    view_z = float(state.view_z) if state.view_z is not None else None
    sed.mask_2d = build_sediment_mask_2d(
        core_union_mask(state),
        center_col_px=sed.center_col_px,
        radius_mm=sed.radius_mm,
        spacing_mm=state.spacing_mm,
        erode_mm=sed.erode_mm,
        nudge_x_mm=sed.nudge_x_mm,
        view_z=view_z,
        center_z=sed.center_z,
        spacing_z_mm=z_mm,
        nudge_z_mm=sed.nudge_z_mm,
    )


def detect_sediment_roi(
    state: StitchState,
    *,
    liner_od_mm: float | None = None,
    liner_wall_mm: float | None = None,
) -> SedimentRoi:
    """Outside-in liner/void peel on current fused slice; store on state."""
    assert state.canvas_a is not None
    img = fused_hu(state)
    mask = core_union_mask(state)
    prev = state.sediment
    od = liner_od_mm if liner_od_mm is not None else (prev.liner_od_mm if prev else None)
    wall = (
        liner_wall_mm
        if liner_wall_mm is not None
        else (prev.liner_wall_mm if prev else None)
    )
    cx, radius_mm, liner_hu, void_hu, note = detect_sediment_from_profile(
        img,
        mask,
        spacing_mm=state.spacing_mm,
        liner_od_mm=od,
        liner_wall_mm=wall,
    )
    cz = float(state.view_z if state.view_z is not None else 0)
    sed = SedimentRoi(
        center_col_px=cx,
        center_z=cz,
        radius_mm=radius_mm,
        erode_mm=prev.erode_mm if prev else 0.0,
        nudge_x_mm=prev.nudge_x_mm if prev else 0.0,
        nudge_z_mm=prev.nudge_z_mm if prev else 0.0,
        liner_od_mm=od,
        liner_wall_mm=wall,
        liner_hu=liner_hu,
        void_hu=void_hu,
        method="outside_in",
        note=note,
    )
    state.sediment = sed
    rebuild_sediment_mask_2d(state)
    return sed


def build_sediment_volume_mask(state: StitchState) -> np.ndarray:
    """
    Build (Z,H,W) sediment cylinder using posed stacks' content masks + ROI.

    Requires posed volumes; uses compose geometry via per-Z core masks on canvas
    size — builds from posed half masks composed like preview.
    """
    if state.sediment is None:
        raise ValueError("No sediment ROI — run detect first")
    posed = ensure_posed(state)
    sed = state.sediment
    # Compose core masks per Z onto canvas (same lag/col as preview)
    from core_stitch.overlap import compose_canvases

    lag = state.total_lag_px
    col = state.total_col_offset_px
    vol_b, mask_b = posed.vol_b, posed.mask_b
    dz_px = int(round(state.nudge_z_mm / max(posed.z_mm, 1e-6)))
    if dz_px != 0:
        vol_b, mask_b = _shift_stack_z(
            vol_b, mask_b, dz_px, cval=state.mobile_base.air
        )
    cores = []
    for z in range(posed.n_slices):
        _ca, _cb, ma, mb = compose_canvases(
            posed.vol_a[z],
            posed.mask_a[z],
            vol_b[z],
            mask_b[z],
            lag,
            col_offset_px=col,
            cval=state.fixed.air,
        )
        cores.append(ma | mb)
    core3 = np.stack(cores, axis=0)
    cz = float(sed.center_z)
    mask3 = build_sediment_mask_3d(
        core3,
        center_col_px=sed.center_col_px,
        center_z=cz,
        radius_mm=sed.radius_mm,
        spacing_xy_mm=state.spacing_mm,
        spacing_z_mm=posed.z_mm,
        erode_mm=sed.erode_mm,
        nudge_x_mm=sed.nudge_x_mm,
        nudge_z_mm=sed.nudge_z_mm,
    )
    sed.mask_3d = mask3
    return mask3


def _downcore_cache_key(state: StitchState) -> tuple | None:
    if state.sediment is None:
        return None
    try:
        posed = ensure_posed(state)
    except Exception:
        return None
    sed = state.sediment
    return (
        id(posed.vol_a),
        int(state.total_lag_px),
        int(state.total_col_offset_px),
        float(state.nudge_z_mm),
        str(state.fuse_mode),
        bool(state.match_hu),
        float(sed.center_col_px),
        float(sed.center_z),
        float(sed.radius_mm),
        float(sed.erode_mm),
        float(sed.nudge_x_mm),
        float(sed.nudge_z_mm),
        bool(sed.mask_3d is not None),
    )


def compute_downcore_mean_hu(state: StitchState) -> DowncoreProfile:
    """
    Mean HU along Y inside the sediment cylinder, averaged over Z (and X).

    Uses mask_3d when present; otherwise accumulates per-Z cylinder chords.
    """
    if state.sediment is None:
        raise ValueError("No sediment ROI — run Auto sediment first")
    sed = state.sediment
    posed, vol_b, mask_b = _posed_b_shifted(state)
    lag = state.total_lag_px
    col = state.total_col_offset_px
    feather = max(4.0, 12.0 * state.spacing_mm)

    # Establish canvas shape from mid Z
    z0 = posed.n_slices // 2
    ca0, cb0, ma0, mb0 = compose_canvases(
        posed.vol_a[z0],
        posed.mask_a[z0],
        vol_b[z0],
        mask_b[z0],
        lag,
        col_offset_px=col,
        cval=state.fixed.air,
    )
    h, w = ca0.shape
    sum_hu = np.zeros(h, dtype=np.float64)
    count = np.zeros(h, dtype=np.float64)

    use_m3 = (
        sed.mask_3d is not None
        and sed.mask_3d.shape[0] == posed.n_slices
        and sed.mask_3d.shape[1] == h
        and sed.mask_3d.shape[2] == w
    )

    for z in range(posed.n_slices):
        ca, cb, ma, mb = compose_canvases(
            posed.vol_a[z],
            posed.mask_a[z],
            vol_b[z],
            mask_b[z],
            lag,
            col_offset_px=col,
            cval=state.fixed.air,
        )
        fused = fuse_preview(
            ca,
            cb,
            ma,
            mb,
            opacity_a=1.0,
            opacity_b=1.0,
            mode=state.fuse_mode,
            feather_mm=feather,
            spacing_mm=state.spacing_mm,
            match_hu=state.match_hu,
        )
        if use_m3:
            sm = sed.mask_3d[z]  # type: ignore[index]
        else:
            sm = build_sediment_mask_2d(
                ma | mb,
                center_col_px=sed.center_col_px,
                radius_mm=sed.radius_mm,
                spacing_mm=state.spacing_mm,
                erode_mm=sed.erode_mm,
                nudge_x_mm=sed.nudge_x_mm,
                view_z=float(z),
                center_z=sed.center_z,
                spacing_z_mm=posed.z_mm,
                nudge_z_mm=sed.nudge_z_mm,
            )
        _mean, cnt = mean_hu_per_row(fused, sm)
        ok = cnt > 0
        sum_hu[ok] += _mean[ok] * cnt[ok]
        count[ok] += cnt[ok]

    mean = np.full(h, np.nan, dtype=np.float64)
    ok = count > 0
    mean[ok] = sum_hu[ok] / count[ok]
    return DowncoreProfile(
        mean_hu=mean,
        count=count.astype(np.int64),
        spacing_mm=state.spacing_mm,
        method="mask_3d_stack" if use_m3 else "cylinder_chord_stack",
    )


def ensure_downcore_profile(state: StitchState) -> DowncoreProfile:
    """Return cached downcore profile, recomputing when pose/ROI changes."""
    key = _downcore_cache_key(state)
    if key is None:
        raise ValueError("No sediment ROI — run Auto sediment first")
    if state._downcore_profile is not None and state._downcore_cache_key == key:
        return state._downcore_profile
    prof = compute_downcore_mean_hu(state)
    state._downcore_profile = prof
    state._downcore_cache_key = key
    return prof


def preview_arrays(
    state: StitchState,
    *,
    align_mode: bool = True,
    opacity_a: float = 0.7,
    opacity_b: float = 0.7,
    crop_core: bool = False,
    pad_mm: float = 3.0,
    overlay_sediment: bool | None = None,
    downcore_plot: bool | None = None,
    plot_width: int = 160,
) -> np.ndarray:
    """Return HxW uint8 or HxWx3 uint8 for display / JPEG / 8-bit TIFF."""
    assert state.canvas_a is not None
    assert state.mask_a is not None and state.mask_b is not None
    lo, hi = window_limits(state)
    show_sed = state.show_sediment if overlay_sediment is None else overlay_sediment
    show_plot = state.show_downcore_plot if downcore_plot is None else downcore_plot
    if align_mode:
        out = alignment_rgb(
            state.canvas_a,
            state.canvas_b,  # type: ignore[arg-type]
            state.mask_a,
            state.mask_b,
            lo=lo,
            hi=hi,
            opacity_a=opacity_a,
            opacity_b=opacity_b,
        )
    else:
        fused = fused_hu(state)
        if hi <= lo:
            hi = lo + 1.0
        u8 = np.clip((fused - lo) / (hi - lo) * 255.0, 0, 255).astype(np.uint8)
        union = state.mask_a | state.mask_b
        out = np.where(union, u8, 0).astype(np.uint8)
    if (
        show_sed
        and state.sediment is not None
        and state.sediment.mask_2d is not None
        and state.sediment.mask_2d.any()
    ):
        out = overlay_sediment_rgb(out, state.sediment.mask_2d)

    box = (0, out.shape[0], 0, out.shape[1])
    if crop_core:
        pad_px = max(2, int(round(pad_mm / max(state.spacing_mm, 1e-6))))
        box = core_crop_box(state.mask_a, state.mask_b, pad_px=pad_px)
        out = crop_to_box(out, box)

    # Plot only if already cached — never compute here (keeps GUI paint responsive)
    if show_plot and state.sediment is not None and state.canvas_a is not None:
        key = _downcore_cache_key(state)
        prof = state._downcore_profile
        if (
            prof is not None
            and key is not None
            and state._downcore_cache_key == key
            and prof.n_rows == state.canvas_a.shape[0]
        ):
            r0, r1, _c0, _c1 = box
            if out.ndim == 2:
                out = np.stack([out, out, out], axis=-1)
            strip = render_downcore_plot_rgb(
                prof, row0=r0, row1=r1, width=plot_width
            )
            if strip.shape[0] == out.shape[0]:
                sep = np.full((out.shape[0], 3, 3), 40, dtype=np.uint8)
                out = np.concatenate([out, sep, strip], axis=1)
    return out


def _posed_b_shifted(
    state: StitchState,
) -> tuple[PosedVolumes, np.ndarray, np.ndarray]:
    posed = ensure_posed(state)
    vol_b, mask_b = posed.vol_b, posed.mask_b
    dz_px = int(round(state.nudge_z_mm / max(posed.z_mm, 1e-6)))
    if dz_px != 0:
        vol_b, mask_b = _shift_stack_z(
            vol_b, mask_b, dz_px, cval=state.mobile_base.air
        )
    return posed, vol_b, mask_b


def default_xz_y_row(state: StitchState, frac: float = 0.5) -> int | None:
    """
    Y (row) along upright core for XZ cut; frac in [0,1] within core span.

    Uses a mid-Z posed compose (not the coronal canvas) so the cut matches the
    volume geometry used to build the XZ plane.
    """
    try:
        posed, vol_b, mask_b = _posed_b_shifted(state)
    except Exception:
        return None
    z = posed.n_slices // 2
    _ca, _cb, ma, mb = compose_canvases(
        posed.vol_a[z],
        posed.mask_a[z],
        vol_b[z],
        mask_b[z],
        state.total_lag_px,
        col_offset_px=state.total_col_offset_px,
        cval=state.fixed.air,
    )
    union = ma | mb
    rows = np.where(union.any(axis=1))[0]
    if rows.size == 0:
        return union.shape[0] // 2
    r0, r1 = int(rows[0]), int(rows[-1])
    t = float(np.clip(frac, 0.0, 1.0))
    return int(round(r0 + t * (r1 - r0)))


def _build_fused_xz_plane(
    state: StitchState, y_row: int
) -> tuple[np.ndarray, np.ndarray]:
    """Fuse every Z at fixed canvas row → HU + core mask shaped (Z, W)."""
    posed, vol_b, mask_b = _posed_b_shifted(state)
    lag = state.total_lag_px
    col = state.total_col_offset_px
    feather = max(4.0, 12.0 * state.spacing_mm)
    hu_rows: list[np.ndarray] = []
    m_rows: list[np.ndarray] = []
    for z in range(posed.n_slices):
        ca, cb, ma, mb = compose_canvases(
            posed.vol_a[z],
            posed.mask_a[z],
            vol_b[z],
            mask_b[z],
            lag,
            col_offset_px=col,
            cval=state.fixed.air,
        )
        y = int(np.clip(y_row, 0, ca.shape[0] - 1))
        # 1-row fuse (same modes as coronal)
        fused = fuse_preview(
            ca[y : y + 1],
            cb[y : y + 1],
            ma[y : y + 1],
            mb[y : y + 1],
            opacity_a=1.0,
            opacity_b=1.0,
            mode=state.fuse_mode,
            feather_mm=feather,
            spacing_mm=state.spacing_mm,
            match_hu=state.match_hu,
        )
        hu_rows.append(fused[0].astype(np.float32))
        m_rows.append((ma[y] | mb[y]))
    return np.stack(hu_rows, axis=0), np.stack(m_rows, axis=0)


def preview_xz_arrays(
    state: StitchState,
    *,
    y_frac: float = 0.5,
    overlay_sediment: bool | None = None,
    crop_core: bool = True,
    pad_mm: float = 3.0,
) -> np.ndarray | None:
    """
    Compact XZ QC image (rows=Z, cols=X), isotropic mm display.

    Returns RGB uint8, or None if posed stack not ready.
    Yellow line = current coronal Z; green ellipse = sediment cylinder.
    """
    try:
        posed = ensure_posed(state)
    except Exception:
        return None
    y_row = default_xz_y_row(state, y_frac)
    if y_row is None:
        return None

    key = (
        id(posed.vol_a),
        int(state.total_lag_px),
        int(state.total_col_offset_px),
        float(state.nudge_z_mm),
        str(state.fuse_mode),
        bool(state.match_hu),
        int(y_row),
    )
    if state._xz_cache_key != key or state._xz_hu is None or state._xz_mask is None:
        state._xz_hu, state._xz_mask = _build_fused_xz_plane(state, y_row)
        state._xz_cache_key = key

    hu = state._xz_hu
    msk = state._xz_mask
    lo, hi = window_limits(state)
    if hi <= lo:
        hi = lo + 1.0
    u8 = np.clip((hu - lo) / (hi - lo) * 255.0, 0, 255).astype(np.uint8)
    # Show something even outside mask (dim) so the panel never looks "empty"
    u8_out = np.where(msk, u8, (u8.astype(np.float32) * 0.15).astype(np.uint8))

    # Crop X to core columns (keep full Z)
    c0, c1 = 0, u8_out.shape[1]
    if crop_core and msk.any():
        cols = np.where(msk.any(axis=0))[0]
        if cols.size:
            pad_px = max(2, int(round(pad_mm / max(state.spacing_mm, 1e-6))))
            c0 = max(0, int(cols[0]) - pad_px)
            c1 = min(u8_out.shape[1], int(cols[-1]) + 1 + pad_px)
            u8_out = u8_out[:, c0:c1]
            msk = msk[:, c0:c1]

    show_sed = state.show_sediment if overlay_sediment is None else overlay_sediment
    view_z = float(state.view_z) if state.view_z is not None else float(posed.n_slices // 2)

    if show_sed and state.sediment is not None:
        sed = state.sediment
        r_mm = sed.effective_radius_mm
        cx = sed.center_col_px + sed.nudge_x_mm / max(state.spacing_mm, 1e-6) - c0
        cz = sed.center_z + sed.nudge_z_mm / max(posed.z_mm, 1e-6)
        out = overlay_sediment_circle_xz(
            u8_out,
            center_x_px=cx,
            center_z_px=cz,
            radius_x_px=r_mm / max(state.spacing_mm, 1e-6),
            radius_z_px=r_mm / max(posed.z_mm, 1e-6),
            view_z_px=view_z,
        )
    else:
        out = np.stack([u8_out, u8_out, u8_out], axis=-1)
        if 0 <= view_z < out.shape[0]:
            out[int(round(view_z)), :] = (255, 220, 60)

    # Isotropic display: scale Z so 1 px ≈ spacing_xy mm
    # (per-channel resize — avoids skimage channel_axis / multichannel API drift)
    z_scale = float(posed.z_mm) / max(state.spacing_mm, 1e-6)
    if abs(z_scale - 1.0) > 0.05 and out.shape[0] > 1:
        new_h = max(1, int(round(out.shape[0] * z_scale)))
        chans = [
            resize(
                out[:, :, i].astype(np.float32),
                (new_h, out.shape[1]),
                order=1,
                preserve_range=True,
                anti_aliasing=True,
            )
            for i in range(out.shape[2])
        ]
        out = np.clip(np.stack(chans, axis=-1), 0, 255).astype(np.uint8)
    return out


def fine_register_axis(
    state: StitchState,
    *,
    search_mm: float = 15.0,
) -> OverlapResult:
    """
    Bounded re-search of lag around current offset (±search_mm).
    """
    assert state.overlap is not None
    img_b, mask_b = _mobile_posed(state)
    spacing = state.spacing_mm
    center = state.overlap.offset_px
    half = max(1, int(round(search_mm / spacing)))

    pa_img, pa_mask = state.fixed.image, state.fixed.mask
    best = state.overlap
    best_score = state.overlap.peak_score
    for lag in range(center - half, center + half + 1):
        # score only this lag via compose+profile ncc through search_overlap range hack:
        # run a tiny local search by temporarily restricting — simpler: direct ncc
        from core_stitch.overlap import _ncc, masked_profile

        ha = pa_img.shape[0]
        hb = img_b.shape[0]
        # extract overlap profiles at this lag
        a0 = max(0, lag)
        b0 = max(0, -lag)
        ov = min(ha - a0, hb - b0)
        if ov < 8:
            continue
        # build temporary canvases is heavy; profile on raw with lag
        prof_a = masked_profile(pa_img, pa_mask)[a0 : a0 + ov]
        prof_b = masked_profile(img_b, mask_b)[b0 : b0 + ov]
        air_a = state.fixed.air
        air_b = state.mobile_base.air
        prof_a = np.where(np.isfinite(prof_a), prof_a, air_a)
        prof_b = np.where(np.isfinite(prof_b), prof_b, air_b)
        score = _ncc(prof_a, prof_b)
        if score > best_score:
            best_score = score
            # update full overlap result by setting offset and recomposing
            best = OverlapResult(
                offset_mm=float(lag * spacing),
                offset_px=int(lag),
                peak_score=float(score),
                second_score=state.overlap.second_score,
                peak_ratio=state.overlap.peak_ratio,
                low_confidence=state.overlap.low_confidence,
                profile_a=state.overlap.profile_a,
                profile_b=state.overlap.profile_b,
            )
    state.overlap = best
    state.nudge_x_mm = 0.0
    state.nudge_y_mm = 0.0
    state.nudge_z_mm = 0.0
    state.refine = None
    invalidate_posed(state)
    recompose(state)
    return best


def refine_join(
    state: StitchState,
    *,
    search_xy_mm: float = 3.0,
    search_z_mm: float = 4.0,
) -> RefineResult:
    """
    Bounded 3D translation refine of B in the implied overlap band.

    Starts from current auto lag + nudge; bakes residual into nudge_x/y/z.
    """
    posed = ensure_posed(state)
    vol_b, mask_b = posed.vol_b, posed.mask_b
    dz_px = int(round(state.nudge_z_mm / max(posed.z_mm, 1e-6)))
    if dz_px != 0:
        vol_b, mask_b = _shift_stack_z(
            vol_b, mask_b, dz_px, cval=state.mobile_base.air
        )
    result = refine_join_volumes(
        posed.vol_a,
        posed.mask_a,
        vol_b,
        mask_b,
        lag_px=state.total_lag_px,
        col_offset_px=state.total_col_offset_px,
        spacing_xy_mm=state.spacing_mm,
        spacing_z_mm=posed.z_mm,
        cval=state.fixed.air,
        search_xy_mm=search_xy_mm,
        search_z_mm=search_z_mm,
    )
    state.nudge_x_mm += result.dx_mm
    state.nudge_y_mm += result.dy_mm
    state.nudge_z_mm += result.dz_mm
    state.refine = result
    if state.view_z is None and state.posed is not None:
        state.view_z = state.posed.n_slices // 2
    recompose(state)
    return result


def _content_mask(image: np.ndarray, threshold: float, air: float) -> np.ndarray:
    """Threshold + largest CC; fallback to above-air if Otsu thr is unusable."""
    from core_stitch.segment import largest_component

    thr = threshold if np.isfinite(threshold) else air + 50.0
    raw = np.isfinite(image) & (image >= thr)
    if raw.mean() > 0.55:
        raw = np.isfinite(image) & (image < thr)
    result = largest_component(raw)
    if result.n_components == 0:
        return np.isfinite(image) & (image > air + 30.0)
    return result.mask


def _transform_half_volume(
    half: HalfPrepared,
    *,
    flip: bool = False,
    roll_deg: float = 0.0,
    relateral_to_col: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Apply the same in-plane prep as the mid-slice half to every Z frame.

    Returns (volume HU, mask) shaped (Z, H, W).
    """
    series = half.series
    _z, row_mm, col_mm = series.spacing_mm
    target = half.target_mm
    air = half.air
    imgs: list[np.ndarray] = []
    masks: list[np.ndarray] = []

    for z in range(series.volume.shape[0]):
        img = resample_to_mm(series.volume[z], (row_mm, col_mm), target)
        img = rotate_image(img, half.upright_deg, cval=air, order=1)
        if abs(half.lateral_dcol) > 1e-3:
            img = shift_image(img, (0.0, half.lateral_dcol), cval=air, order=1)
        mask = _content_mask(img, half.seg_threshold, air)
        imgs.append(img)
        masks.append(mask)

    # Pad/crop to a common in-plane shape (rotate reshape is stable per input size)
    h = max(im.shape[0] for im in imgs)
    w = max(im.shape[1] for im in imgs)
    vol = np.full((len(imgs), h, w), air, dtype=np.float32)
    mvol = np.zeros((len(imgs), h, w), dtype=bool)
    for i, (im, m) in enumerate(zip(imgs, masks)):
        r0 = (h - im.shape[0]) // 2
        c0 = (w - im.shape[1]) // 2
        vol[i, r0 : r0 + im.shape[0], c0 : c0 + im.shape[1]] = im
        mvol[i, r0 : r0 + m.shape[0], c0 : c0 + m.shape[1]] = m

    if flip:
        vol = np.flip(vol, axis=1).copy()
        mvol = np.flip(mvol, axis=1).copy()

    if abs(roll_deg) > 1e-6:
        rolled_i: list[np.ndarray] = []
        rolled_m: list[np.ndarray] = []
        for i in range(vol.shape[0]):
            ri, rm = apply_roll(vol[i], mvol[i], roll_deg, cval=air)
            if relateral_to_col is not None:
                g = core_geometry(rm, target)
                ri, rm = lateral_align_to(ri, rm, g, relateral_to_col)
            rolled_i.append(ri)
            rolled_m.append(rm)
        h = max(im.shape[0] for im in rolled_i)
        w = max(im.shape[1] for im in rolled_i)
        vol = np.full((len(rolled_i), h, w), air, dtype=np.float32)
        mvol = np.zeros((len(rolled_i), h, w), dtype=bool)
        for i, (im, m) in enumerate(zip(rolled_i, rolled_m)):
            r0 = (h - im.shape[0]) // 2
            c0 = (w - im.shape[1]) // 2
            vol[i, r0 : r0 + im.shape[0], c0 : c0 + im.shape[1]] = im
            mvol[i, r0 : r0 + m.shape[0], c0 : c0 + m.shape[1]] = m

    return vol, mvol


def _resample_volume_z(vol: np.ndarray, src_mm: float, dst_mm: float, cval: float) -> np.ndarray:
    if vol.shape[0] < 2 or abs(src_mm - dst_mm) < 1e-9:
        return np.asarray(vol, dtype=np.float32)
    extent = (vol.shape[0] - 1) * src_mm
    new_n = max(1, int(round(extent / dst_mm)) + 1)
    if new_n == vol.shape[0]:
        return np.asarray(vol, dtype=np.float32)
    out = resize(
        vol.astype(np.float32),
        (new_n, vol.shape[1], vol.shape[2]),
        order=1,
        mode="constant",
        cval=cval,
        anti_aliasing=False,
        preserve_range=True,
    )
    return np.asarray(out, dtype=np.float32)


def _resample_mask_z(mask: np.ndarray, src_mm: float, dst_mm: float) -> np.ndarray:
    if mask.shape[0] < 2 or abs(src_mm - dst_mm) < 1e-9:
        return mask.astype(bool)
    extent = (mask.shape[0] - 1) * src_mm
    new_n = max(1, int(round(extent / dst_mm)) + 1)
    if new_n == mask.shape[0]:
        return mask.astype(bool)
    out = resize(
        mask.astype(np.float32),
        (new_n, mask.shape[1], mask.shape[2]),
        order=0,
        mode="constant",
        cval=0.0,
        anti_aliasing=False,
        preserve_range=True,
    )
    return out >= 0.5


def _center_crop_z(vol: np.ndarray, n: int) -> np.ndarray:
    if vol.shape[0] == n:
        return vol
    z0 = max(0, (vol.shape[0] - n) // 2)
    return vol[z0 : z0 + n]


def build_stitched_mpr(state: StitchState) -> tuple[np.ndarray, float]:
    """
    Apply current pose (flip/roll/lag/nudge/refine) to full Coron stacks and fuse.

    Returns (fused HU volume Z×H×W, slice_spacing_mm).
    """
    vol_a, mask_a, vol_b, mask_b, z_mm = prepare_posed_volumes(state)
    lag = state.total_lag_px
    col = state.total_col_offset_px
    air = state.fixed.air
    feather_mm = max(4.0, 12.0 * state.spacing_mm)
    fused_slices: list[np.ndarray] = []
    for z in range(vol_a.shape[0]):
        ca, cb, ma, mb = compose_canvases(
            vol_a[z],
            mask_a[z],
            vol_b[z],
            mask_b[z],
            lag,
            col_offset_px=col,
            cval=air,
        )
        fused_slices.append(
            fuse_preview(
                ca,
                cb,
                ma,
                mb,
                opacity_a=1.0,
                opacity_b=1.0,
                mode=state.fuse_mode,
                feather_mm=feather_mm,
                spacing_mm=state.spacing_mm,
                match_hu=state.match_hu,
            )
        )
    fused = np.stack(fused_slices, axis=0).astype(np.float32)
    return fused, z_mm


def sidecar_dict(state: StitchState) -> dict:
    ov = state.overlap
    ref = state.refine
    return {
        "version": "0.1.0",
        "fixed": {
            "path": str(state.fixed.series.path),
            "series_uid": state.fixed.series.series_uid,
            "series_description": state.fixed.series.series_description,
        },
        "mobile": {
            "path": str(state.mobile_base.series.path),
            "series_uid": state.mobile_base.series.series_uid,
            "series_description": state.mobile_base.series.series_description,
        },
        "target_mm": state.spacing_mm,
        "flip": state.flip,
        "roll_deg": state.roll_deg,
        "nudge_x_mm": state.nudge_x_mm,
        "nudge_y_mm": state.nudge_y_mm,
        "nudge_z_mm": state.nudge_z_mm,
        "fuse_mode": state.fuse_mode,
        "match_hu": state.match_hu,
        "window_lo": state.window_lo,
        "window_hi": state.window_hi,
        "view_z": state.view_z,
        "n_slices": None if state.posed is None else state.posed.n_slices,
        "overlap_mm": None if ov is None else ov.offset_mm,
        "overlap_px": None if ov is None else ov.offset_px,
        "total_lag_px": state.total_lag_px,
        "total_col_offset_px": state.total_col_offset_px,
        "peak_score": None if ov is None else ov.peak_score,
        "second_score": None if ov is None else ov.second_score,
        "peak_ratio": None if ov is None else ov.peak_ratio,
        "low_confidence": None if ov is None else ov.low_confidence,
        "refine": None
        if ref is None
        else {
            "dx_mm": ref.dx_mm,
            "dy_mm": ref.dy_mm,
            "dz_mm": ref.dz_mm,
            "score": ref.score,
            "score_before": ref.score_before,
            "method": ref.method,
            "shift_px_zyx": list(ref.shift_px),
        },
        "sediment": None
        if state.sediment is None
        else {
            "center_col_px": state.sediment.center_col_px,
            "center_z": state.sediment.center_z,
            "radius_mm": state.sediment.radius_mm,
            "erode_mm": state.sediment.erode_mm,
            "effective_radius_mm": state.sediment.effective_radius_mm,
            "nudge_x_mm": state.sediment.nudge_x_mm,
            "nudge_z_mm": state.sediment.nudge_z_mm,
            "liner_od_mm": state.sediment.liner_od_mm,
            "liner_wall_mm": state.sediment.liner_wall_mm,
            "liner_hu": state.sediment.liner_hu,
            "void_hu": state.sediment.void_hu,
            "method": state.sediment.method,
            "note": state.sediment.note,
            "has_mask_2d": bool(
                state.sediment.mask_2d is not None and state.sediment.mask_2d.any()
            ),
            "has_mask_3d": bool(
                state.sediment.mask_3d is not None and state.sediment.mask_3d.any()
            ),
        },
        "warnings": list(state.warnings),
    }
