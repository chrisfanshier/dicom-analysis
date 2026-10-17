"""Whole-core sediment ROI: outside-in liner/void peel (+ optional liner OD/wall prior)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import binary_erosion, gaussian_filter1d

# Common whole-core liner presets (OD, wall) in mm — edit as needed
LINER_PRESETS: dict[str, tuple[float | None, float | None]] = {
    "Auto (no prior)": (None, None),
    "OD 66 mm · wall 3.0 mm": (66.0, 3.0),
    "OD 70 mm · wall 3.0 mm": (70.0, 3.0),
    "OD 98 mm · wall 3.0 mm": (98.0, 3.0),
    "OD 110 mm · wall 3.2 mm": (110.0, 3.2),
}


@dataclass
class SedimentRoi:
    """Sediment ROI parameters + masks (whole-core cylinder / vertical band)."""

    center_col_px: float
    center_z: float  # posed-stack index
    radius_mm: float  # auto inner radius before erode
    erode_mm: float = 0.0
    nudge_x_mm: float = 0.0
    nudge_z_mm: float = 0.0
    liner_od_mm: float | None = None
    liner_wall_mm: float | None = None
    liner_hu: float | None = None
    void_hu: float | None = None
    method: str = "outside_in"
    note: str = ""
    mask_2d: np.ndarray | None = None  # current canvas (H, W)
    mask_3d: np.ndarray | None = None  # (Z, H, W) if built

    @property
    def effective_radius_mm(self) -> float:
        return max(0.0, float(self.radius_mm) - float(self.erode_mm))


def _column_profile(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Mean HU per column over masked rows; NaN where no mask."""
    _h, w = image.shape
    prof = np.full(w, np.nan, dtype=np.float64)
    for c in range(w):
        m = mask[:, c]
        if m.any():
            prof[c] = float(image[m, c].mean())
    return prof


def _interp_nans(y: np.ndarray) -> np.ndarray:
    x = np.arange(y.size)
    good = np.isfinite(y)
    if good.sum() < 4:
        return np.where(np.isfinite(y), y, 0.0)
    return np.interp(x, x[good], y[good])


def _peel_from_outside(
    seg: np.ndarray,
    *,
    from_left: bool,
    spacing_mm: float,
    wall_mm: float | None,
) -> tuple[int, float, float]:
    """
    Walk from the outer end of the profile inward.

    1) Find liner peak in a short outer window
    2) Find void minimum in a short window just inward of that peak
       (does NOT search all the way to center — avoids sediment laminations)

    Returns (inner_edge_index_in_seg, liner_hu, void_hu).
    """
    n = seg.size
    if n < 8:
        return n // 2, float(seg.max()), float(seg.min())

    # Work left→right; flip if peeling from right
    y = seg if from_left else seg[::-1]

    sp = max(spacing_mm, 1e-6)
    if wall_mm is not None and wall_mm > 0:
        wall_px = max(2, int(round(wall_mm / sp)))
        liner_lo = max(0, int(round(0.25 * wall_px)))
        liner_hi = min(n - 1, int(round(2.0 * wall_px)))
        void_depth = max(2, int(round(1.5 * wall_px)))
    else:
        # ~outer 12% for liner, then ~10% further for void
        liner_lo = 0
        liner_hi = max(4, int(round(0.12 * n)))
        void_depth = max(3, int(round(0.10 * n)))

    liner_hi = max(liner_lo + 2, min(liner_hi, n - 3))
    peak_rel = int(np.argmax(y[liner_lo : liner_hi + 1]))
    peak = liner_lo + peak_rel
    liner_hu = float(y[peak])

    # Void: only immediately inward of liner peak (not into mid-core)
    v0 = peak + 1
    v1 = min(n - 1, peak + void_depth)
    if v1 <= v0:
        inner = peak
        void_hu = liner_hu
    else:
        void_rel = int(np.argmin(y[v0 : v1 + 1]))
        inner = v0 + void_rel
        void_hu = float(y[inner])

    # Map back if we flipped
    if not from_left:
        inner = n - 1 - inner
        # peak unused after return
    return int(inner), liner_hu, void_hu


def detect_sediment_from_profile(
    image: np.ndarray,
    mask: np.ndarray,
    *,
    spacing_mm: float,
    liner_od_mm: float | None = None,
    liner_wall_mm: float | None = None,
) -> tuple[float, float, float | None, float | None, str]:
    """
    Outside-in liner/void peel on both sides of a horizontal HU profile.

    Optional liner OD + wall (mm) tighten the outer search windows and
    can seed radius if peel is weak.

    Returns (center_col_px, radius_mm, liner_hu, void_hu, note).
    """
    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    if rows.size < 8 or cols.size < 16:
        cy, cx = np.array(mask.shape) / 2.0
        return float(cx), 10.0, None, None, "fallback: mask too small"

    r0 = int(rows[int(0.25 * (len(rows) - 1))])
    r1 = int(rows[int(0.75 * (len(rows) - 1))]) + 1
    band = np.zeros_like(mask)
    band[r0:r1, :] = mask[r0:r1, :]
    if band.sum() < 32:
        band = mask

    prof = _interp_nans(_column_profile(image, band))
    sm = gaussian_filter1d(prof, sigma=max(1.0, 1.5 / max(spacing_mm, 1e-3)))

    c0, c1 = int(cols[0]), int(cols[-1]) + 1
    seg = sm[c0:c1]
    if seg.size < 16:
        half = 0.35 * (c1 - c0) * spacing_mm
        return float(0.5 * (c0 + c1)), float(half), None, None, "fallback: short profile"

    wall = float(liner_wall_mm) if liner_wall_mm else None
    od = float(liner_od_mm) if liner_od_mm else None

    li, l_hu, l_void = _peel_from_outside(
        seg, from_left=True, spacing_mm=spacing_mm, wall_mm=wall
    )
    ri, r_hu, r_void = _peel_from_outside(
        seg, from_left=False, spacing_mm=spacing_mm, wall_mm=wall
    )

    # Ensure order
    if ri < li:
        li, ri = ri, li
        l_hu, r_hu = r_hu, l_hu
        l_void, r_void = r_void, l_void

    left_inner = float(c0 + li)
    right_inner = float(c0 + ri)
    liner_hu = float(0.5 * (l_hu + r_hu))
    void_hu = float(0.5 * (l_void + r_void))

    note = "outside-in liner->void"
    prior_r = None
    if od is not None and wall is not None and od > 2 * wall:
        prior_r = 0.5 * od - wall
        note += f" · prior ID~{2 * prior_r:.1f} mm"

    # If one side failed (inner edges too close / crossed), use prior or mask fraction
    min_width_px = max(6, int(round(8.0 / max(spacing_mm, 1e-6))))
    if right_inner - left_inner < min_width_px:
        mid = 0.5 * (c0 + c1 - 1)
        if prior_r is not None:
            r_px = prior_r / max(spacing_mm, 1e-6)
            left_inner = mid - r_px
            right_inner = mid + r_px
            note += " · weak peel→prior radius"
        else:
            left_inner = float(c0 + 0.18 * (c1 - c0))
            right_inner = float(c0 + 0.82 * (c1 - c0))
            note += " · weak peel→fractional width"

    center_col = 0.5 * (left_inner + right_inner)
    radius_mm = float(0.5 * (right_inner - left_inner) * spacing_mm)

    # Blend toward prior if available (keeps peel, soft-pulls scale)
    if prior_r is not None:
        radius_mm = float(0.65 * radius_mm + 0.35 * prior_r)
        # Keep center from peel; optionally nudge if OD vs mask width disagrees strongly
        mask_width_mm = (c1 - c0) * spacing_mm
        if abs(mask_width_mm - od) > 0.25 * od:
            note += " · mask width≠OD (check crop/align)"

    mask_half = 0.5 * (cols[-1] - cols[0]) * spacing_mm
    radius_mm = float(np.clip(radius_mm, 2.0 * spacing_mm, 0.95 * mask_half))

    return center_col, radius_mm, liner_hu, void_hu, note


def build_sediment_mask_2d(
    core_mask: np.ndarray,
    *,
    center_col_px: float,
    radius_mm: float,
    spacing_mm: float,
    erode_mm: float = 0.0,
    nudge_x_mm: float = 0.0,
    view_z: float | None = None,
    center_z: float | None = None,
    spacing_z_mm: float | None = None,
    nudge_z_mm: float = 0.0,
) -> np.ndarray:
    """
    Coronal sediment mask = cylinder chord at this Z.

    Without Z params: full-diameter vertical band |x-cx|<=R (legacy).
    With view_z / center_z / spacing_z: half-width shrinks as
    sqrt(R^2 - dz^2) so off-axis coronals do not spill into the liner.
    """
    _h, w = core_mask.shape
    r_mm = max(0.0, radius_mm - erode_mm)
    half_mm = r_mm
    if (
        view_z is not None
        and center_z is not None
        and spacing_z_mm is not None
        and spacing_z_mm > 0
        and r_mm > 0
    ):
        cz = float(center_z) + float(nudge_z_mm) / float(spacing_z_mm)
        dz_mm = (float(view_z) - cz) * float(spacing_z_mm)
        inside = r_mm * r_mm - dz_mm * dz_mm
        if inside <= 0.0:
            return np.zeros_like(core_mask, dtype=bool)
        half_mm = float(np.sqrt(inside))
    r_px = half_mm / max(spacing_mm, 1e-6)
    cx = center_col_px + nudge_x_mm / max(spacing_mm, 1e-6)
    xx = np.arange(w, dtype=np.float64)[np.newaxis, :]
    band = np.abs(xx - cx) <= r_px
    out = core_mask & band
    if erode_mm > 0.5 * spacing_mm and out.any():
        iters = max(1, int(round(0.25 * erode_mm / spacing_mm)))
        out = binary_erosion(out, iterations=iters)
        out &= core_mask
    return out


def build_sediment_mask_3d(
    core_masks: np.ndarray,
    *,
    center_col_px: float,
    center_z: float,
    radius_mm: float,
    spacing_xy_mm: float,
    spacing_z_mm: float,
    erode_mm: float = 0.0,
    nudge_x_mm: float = 0.0,
    nudge_z_mm: float = 0.0,
) -> np.ndarray:
    """
    Circular cylinder along Y: (x-cx)^2 + (z-cz)^2 <= R^2, AND per-slice core mask.

    core_masks: (Z, H, W) bool
    """
    nz, _h, w = core_masks.shape
    r_mm = max(0.0, radius_mm - erode_mm)
    cx = center_col_px + nudge_x_mm / max(spacing_xy_mm, 1e-6)
    cz = center_z + nudge_z_mm / max(spacing_z_mm, 1e-6)
    z = np.arange(nz, dtype=np.float64)[:, None, None]
    x = np.arange(w, dtype=np.float64)[None, None, :]
    dx = (x - cx) * spacing_xy_mm
    dz = (z - cz) * spacing_z_mm
    cyl = (dx * dx + dz * dz) <= (r_mm * r_mm)
    return core_masks & cyl


def overlay_sediment_rgb(
    base: np.ndarray,
    sediment_mask: np.ndarray,
    *,
    alpha: float = 0.35,
) -> np.ndarray:
    """Tint sediment ROI green on an L or RGB uint8 image; draw edge."""
    if base.ndim == 2:
        rgb = np.stack([base, base, base], axis=-1).astype(np.float32)
    else:
        rgb = base.astype(np.float32).copy()
    if sediment_mask.shape[:2] != rgb.shape[:2]:
        return np.clip(rgb, 0, 255).astype(np.uint8)
    edge = sediment_mask ^ binary_erosion(sediment_mask, iterations=1)
    tint = rgb.copy()
    tint[sediment_mask, 1] = np.clip(
        tint[sediment_mask, 1] * (1 - alpha) + 220 * alpha, 0, 255
    )
    tint[sediment_mask, 0] *= 1 - 0.5 * alpha
    tint[sediment_mask, 2] *= 1 - 0.5 * alpha
    tint[edge] = (40, 255, 80)
    return np.clip(tint, 0, 255).astype(np.uint8)


def overlay_sediment_circle_xz(
    base: np.ndarray,
    *,
    center_x_px: float,
    center_z_px: float,
    radius_x_px: float,
    radius_z_px: float,
    view_z_px: float | None = None,
    alpha: float = 0.30,
) -> np.ndarray:
    """
    Tint circular (or elliptical in px) sediment ROI on an XZ image.

    Image layout: rows = Z, cols = X. Optional horizontal line at view_z_px.
    """
    if base.ndim == 2:
        rgb = np.stack([base, base, base], axis=-1).astype(np.float32)
    else:
        rgb = base.astype(np.float32).copy()
    nz, w = rgb.shape[:2]
    if radius_x_px <= 0 or radius_z_px <= 0:
        out = np.clip(rgb, 0, 255).astype(np.uint8)
        if view_z_px is not None and 0 <= view_z_px < nz:
            zi = int(round(view_z_px))
            out[zi, :] = (255, 220, 60)
        return out
    zz = np.arange(nz, dtype=np.float64)[:, None]
    xx = np.arange(w, dtype=np.float64)[None, :]
    disk = ((xx - center_x_px) / radius_x_px) ** 2 + (
        (zz - center_z_px) / radius_z_px
    ) ** 2 <= 1.0
    ring = disk ^ binary_erosion(disk, iterations=1)
    tint = rgb.copy()
    tint[disk, 1] = np.clip(tint[disk, 1] * (1 - alpha) + 220 * alpha, 0, 255)
    tint[disk, 0] *= 1 - 0.5 * alpha
    tint[disk, 2] *= 1 - 0.5 * alpha
    tint[ring] = (40, 255, 80)
    out = np.clip(tint, 0, 255).astype(np.uint8)
    if view_z_px is not None and 0 <= view_z_px < nz:
        zi = int(np.clip(round(view_z_px), 0, nz - 1))
        out[zi, :] = (255, 220, 60)
    # Center crosshair
    cx_i = int(np.clip(round(center_x_px), 0, w - 1))
    cz_i = int(np.clip(round(center_z_px), 0, nz - 1))
    out[cz_i, max(0, cx_i - 3) : min(w, cx_i + 4)] = (40, 255, 80)
    out[max(0, cz_i - 3) : min(nz, cz_i + 4), cx_i] = (40, 255, 80)
    return out
