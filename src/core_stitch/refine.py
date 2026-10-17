"""Bounded 3D join refine on the implied A∩B overlap (phase correlation + NCC)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from skimage.registration import phase_cross_correlation

from core_stitch.overlap import _ncc, compose_canvases


@dataclass(frozen=True)
class RefineResult:
    dx_mm: float  # added to nudge_x (columns)
    dy_mm: float  # added to nudge_y (rows / along-core)
    dz_mm: float  # B shift along stack (+Z = higher index)
    score: float  # NCC on overlap after refine
    score_before: float
    method: str
    shift_px: tuple[float, float, float]  # (dz, dy, dx) applied to B


def _shift_stack_z(
    vol: np.ndarray,
    mask: np.ndarray,
    dz_px: int,
    *,
    cval: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Integer Z shift of B; +dz moves content toward higher Z."""
    if dz_px == 0:
        return vol, mask
    out = np.full_like(vol, cval)
    mout = np.zeros_like(mask)
    if dz_px > 0:
        out[dz_px:] = vol[:-dz_px]
        mout[dz_px:] = mask[:-dz_px]
    else:
        k = -int(dz_px)
        out[:-k] = vol[k:]
        mout[:-k] = mask[k:]
    return out, mout


def _compose_overlap_stacks(
    vol_a: np.ndarray,
    mask_a: np.ndarray,
    vol_b: np.ndarray,
    mask_b: np.ndarray,
    *,
    lag_px: int,
    col_offset_px: int,
    cval: float,
    z_indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compose selected Z slices onto a shared canvas; return Za,Zb,Ma,Mb stacks."""
    slices_a: list[np.ndarray] = []
    slices_b: list[np.ndarray] = []
    masks_a: list[np.ndarray] = []
    masks_b: list[np.ndarray] = []
    for z in z_indices:
        ca, cb, ma, mb = compose_canvases(
            vol_a[z],
            mask_a[z],
            vol_b[z],
            mask_b[z],
            lag_px,
            col_offset_px=col_offset_px,
            cval=cval,
        )
        slices_a.append(ca)
        slices_b.append(cb)
        masks_a.append(ma)
        masks_b.append(mb)
    return (
        np.stack(slices_a, axis=0),
        np.stack(slices_b, axis=0),
        np.stack(masks_a, axis=0),
        np.stack(masks_b, axis=0),
    )


def _overlap_bbox(both: np.ndarray, *, pad: int = 8) -> tuple[slice, slice, slice] | None:
    """Bounding box of 3D boolean mask as (z,y,x) slices."""
    coords = np.argwhere(both)
    if coords.size == 0:
        return None
    z0, y0, x0 = coords.min(axis=0)
    z1, y1, x1 = coords.max(axis=0) + 1
    z0 = max(0, z0 - 0)
    y0 = max(0, y0 - pad)
    x0 = max(0, x0 - pad)
    z1 = min(both.shape[0], z1)
    y1 = min(both.shape[1], y1 + pad)
    x1 = min(both.shape[2], x1 + pad)
    if (z1 - z0) < 2 or (y1 - y0) < 16 or (x1 - x0) < 16:
        return None
    return slice(z0, z1), slice(y0, y1), slice(x0, x1)


def _masked_ncc_3d(
    a: np.ndarray,
    b: np.ndarray,
    mask: np.ndarray,
) -> float:
    if mask.sum() < 64:
        return -1e9
    return _ncc(a[mask].astype(np.float64), b[mask].astype(np.float64))


def _downsample_xy(vol: np.ndarray, factor: int) -> np.ndarray:
    if factor <= 1:
        return vol
    return np.ascontiguousarray(vol[:, ::factor, ::factor])


def refine_join_volumes(
    vol_a: np.ndarray,
    mask_a: np.ndarray,
    vol_b: np.ndarray,
    mask_b: np.ndarray,
    *,
    lag_px: int,
    col_offset_px: int,
    spacing_xy_mm: float,
    spacing_z_mm: float,
    cval: float,
    search_xy_mm: float = 3.0,
    search_z_mm: float = 4.0,
    max_z_slices: int = 24,
) -> RefineResult:
    """
    Estimate residual (dz, dy, dx) to apply to B from current lag/col.

    Uses phase correlation on the implied overlap ROI, then a small NCC
    refine; rejects shifts outside the search bounds.
    """
    nz = vol_a.shape[0]
    if nz >= 3:
        z0 = max(0, nz // 2 - max_z_slices // 2)
        z1 = min(nz, z0 + max_z_slices)
        z_indices = np.arange(z0, z1)
    else:
        z_indices = np.arange(nz)

    ca, cb, ma, mb = _compose_overlap_stacks(
        vol_a,
        mask_a,
        vol_b,
        mask_b,
        lag_px=lag_px,
        col_offset_px=col_offset_px,
        cval=cval,
        z_indices=z_indices,
    )
    both = ma & mb
    score_before = _masked_ncc_3d(ca, cb, both)
    bbox = _overlap_bbox(both, pad=12)
    if bbox is None:
        return RefineResult(
            dx_mm=0.0,
            dy_mm=0.0,
            dz_mm=0.0,
            score=float(score_before),
            score_before=float(score_before),
            method="skipped_no_overlap",
            shift_px=(0.0, 0.0, 0.0),
        )

    zs, ys, xs = bbox
    ra = ca[zs, ys, xs].astype(np.float32)
    rb = cb[zs, ys, xs].astype(np.float32)
    rm = both[zs, ys, xs]
    # Suppress exterior for correlation
    fill = float(np.median(ra[rm])) if rm.any() else 0.0
    ra = np.where(rm, ra, fill)
    rb = np.where(rm, rb, fill)

    max_xy = max(1, int(np.ceil(search_xy_mm / max(spacing_xy_mm, 1e-6))))
    max_z = max(0, int(np.ceil(search_z_mm / max(spacing_z_mm, 1e-6))))

    # Coarse PCR on XY-downsampled stack (faster, still 3D)
    ds = 2 if min(ra.shape[1], ra.shape[2]) >= 64 else 1
    ra_ds = _downsample_xy(ra, ds)
    rb_ds = _downsample_xy(rb, ds)
    rm_ds = _downsample_xy(rm.astype(np.uint8), ds) > 0
    try:
        shift_ds, _, _ = phase_cross_correlation(
            ra_ds,
            rb_ds,
            upsample_factor=1,
            reference_mask=rm_ds,
            moving_mask=rm_ds,
        )
        # shift is how much to move rb_ds to align to ra_ds → (dz, dy/ds, dx/ds)
        dz0 = float(shift_ds[0])
        dy0 = float(shift_ds[1]) * ds
        dx0 = float(shift_ds[2]) * ds
        method = "phase_corr+ncc"
    except Exception:
        dz0 = dy0 = dx0 = 0.0
        method = "ncc_grid"

    # Clamp PCR seed into bounds, then local NCC grid (full-res ROI)
    dz0 = float(np.clip(round(dz0), -max_z, max_z))
    dy0 = float(np.clip(round(dy0), -max_xy, max_xy))
    dx0 = float(np.clip(round(dx0), -max_xy, max_xy))

    best = (dz0, dy0, dx0)
    best_score = -1e9
    # Neighborhood around PCR estimate
    z_radius = min(max_z, 2)
    xy_radius = min(max_xy, 3)
    for dz in range(int(dz0) - z_radius, int(dz0) + z_radius + 1):
        if abs(dz) > max_z:
            continue
        rb_z, rm_z = _shift_stack_z(rb, rm, int(dz), cval=fill)
        for dy in range(int(dy0) - xy_radius, int(dy0) + xy_radius + 1):
            if abs(dy) > max_xy:
                continue
            for dx in range(int(dx0) - xy_radius, int(dx0) + xy_radius + 1):
                if abs(dx) > max_xy:
                    continue
                # Shift B in-plane within ROI (np.roll + zero edges roughly via mask)
                b_shift = np.roll(rb_z, shift=(int(dy), int(dx)), axis=(1, 2))
                m_shift = np.roll(rm_z, shift=(int(dy), int(dx)), axis=(1, 2))
                # Invalidate wrapped margins
                if dy > 0:
                    m_shift[:, :dy, :] = False
                elif dy < 0:
                    m_shift[:, dy:, :] = False
                if dx > 0:
                    m_shift[:, :, :dx] = False
                elif dx < 0:
                    m_shift[:, :, dx:] = False
                score = _masked_ncc_3d(ra, b_shift, rm & m_shift)
                if score > best_score:
                    best_score = score
                    best = (float(dz), float(dy), float(dx))

    if best_score < score_before - 0.02:
        # Do not worsen the join
        return RefineResult(
            dx_mm=0.0,
            dy_mm=0.0,
            dz_mm=0.0,
            score=float(score_before),
            score_before=float(score_before),
            method="rejected_worse",
            shift_px=(0.0, 0.0, 0.0),
        )

    dz, dy, dx = best
    # Sub-pixel polish on best integer via upsampled PCR at that Z (2D mid of ROI)
    try:
        rb_z, rm_z = _shift_stack_z(rb, rm, int(dz), cval=fill)
        b_shift = np.roll(rb_z, shift=(int(dy), int(dx)), axis=(1, 2))
        mid = ra.shape[0] // 2
        sh, _, _ = phase_cross_correlation(
            ra[mid],
            b_shift[mid],
            upsample_factor=10,
            reference_mask=rm[mid],
            moving_mask=np.roll(rm_z[mid], shift=(int(dy), int(dx)), axis=(0, 1)),
        )
        # residual 2D shift of B toward A
        dy += float(np.clip(sh[0], -0.9, 0.9))
        dx += float(np.clip(sh[1], -0.9, 0.9))
        method = "phase_corr+ncc+subpixel"
    except Exception:
        pass

    return RefineResult(
        dx_mm=float(dx * spacing_xy_mm),
        dy_mm=float(dy * spacing_xy_mm),
        dz_mm=float(dz * spacing_z_mm),
        score=float(best_score),
        score_before=float(score_before),
        method=method,
        shift_px=(dz, dy, dx),
    )
