"""1-DoF overlap search along core axis; flip/roll; canvas compose."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from core_stitch.geometry import CoreGeometry, air_value, shift_image, shift_mask


@dataclass(frozen=True)
class OverlapResult:
    offset_mm: float  # slide of B along +row (after upright) relative to A
    offset_px: int
    peak_score: float
    second_score: float
    peak_ratio: float
    low_confidence: bool
    profile_a: np.ndarray
    profile_b: np.ndarray


def upright_angle(geom: CoreGeometry) -> float:
    """Rotation (CCW deg) to make long axis align with +row (vertical down)."""
    # axis angle is atan2(col, row); want axis → (1, 0)
    return -geom.angle_deg


def masked_profile(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Mean intensity along rows using masked columns; NaN where no mask."""
    h = image.shape[0]
    prof = np.full(h, np.nan, dtype=np.float64)
    for r in range(h):
        m = mask[r]
        if m.any():
            prof[r] = float(image[r, m].mean())
    return prof


def _ncc(a: np.ndarray, b: np.ndarray) -> float:
    """NCC on finite pairs only."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 8:
        return -1e9
    a = a[m]
    b = b[m]
    a = (a - a.mean()) / (a.std() + 1e-6)
    b = (b - b.mean()) / (b.std() + 1e-6)
    return float(np.dot(a, b) / len(a))


def _mask_row_span(mask: np.ndarray) -> tuple[int, int]:
    rows = np.where(mask.any(axis=1))[0]
    if rows.size == 0:
        return 0, max(0, mask.shape[0] - 1)
    return int(rows.min()), int(rows.max())


def search_overlap(
    image_a: np.ndarray,
    mask_a: np.ndarray,
    image_b: np.ndarray,
    mask_b: np.ndarray,
    *,
    spacing_mm: float,
    min_overlap_frac: float = 0.08,
    max_overlap_frac: float = 0.45,
    ambiguous_ratio: float = 0.92,
    seed_overlap_frac: float = 0.18,
    refine_frac: float = 0.25,
    require_below: bool = False,
) -> OverlapResult:
    """
    Search integer row shifts of B relative to A (B[0] at A[lag]).

    Images are assumed already upright (long axis ≈ vertical) and laterally aligned.
    Seeds from core-mask extents (A above / B below and swapped), then refines with NCC.
    """
    pa = masked_profile(image_a, mask_a)
    pb = masked_profile(image_b, mask_b)

    a0, a1 = _mask_row_span(mask_a)
    b0, b1 = _mask_row_span(mask_b)
    a_len = max(a1 - a0, 1)
    b_len = max(b1 - b0, 1)
    seed_ov = int(round(seed_overlap_frac * min(a_len, b_len)))
    seed_ov = max(8, seed_ov)

    # A=TOP above, B=BTM below: B content starts near end of A content
    seed_below = (a1 - seed_ov) - b0
    seed_above = (a0 + seed_ov) - b1

    min_ov = max(8, int(round(min_overlap_frac * min(len(pa), len(pb)))))
    max_ov = max(min_ov + 1, int(round(max_overlap_frac * min(len(pa), len(pb)))))
    refine = max(20, int(round(refine_frac * min(a_len, b_len))))

    candidate_lags: set[int] = set()
    for seed in (seed_below, seed_above):
        for lag in range(seed - refine, seed + refine + 1):
            candidate_lags.add(int(lag))
    # Also a coarse global sweep every N px so we don't miss odd cases
    step = max(4, min_ov // 4)
    for lag in range(-len(pb) + min_ov, len(pa) - min_ov + 1, step):
        candidate_lags.add(int(lag))

    scores: list[tuple[float, int]] = []
    for lag in sorted(candidate_lags):
        if require_below and lag <= 0:
            continue  # B must start below A's start: the order is fixed (used when the halves were swapped by hand)
        aa0 = max(0, lag)
        bb0 = max(0, -lag)
        ov = min(len(pa) - aa0, len(pb) - bb0)
        if ov < min_ov or ov > max_ov:
            continue
        # Prefer overlaps that involve mask content, not empty FOV
        a_rows = np.arange(aa0, aa0 + ov)
        b_rows = np.arange(bb0, bb0 + ov)
        content = mask_a[a_rows].any(axis=1) & mask_b[b_rows].any(axis=1)
        if content.sum() < min_ov // 2:
            continue
        score = _ncc(pa[aa0 : aa0 + ov], pb[bb0 : bb0 + ov])
        scores.append((score, lag))

    if not scores:
        # Fallback seed without NCC
        lag = int(seed_below)
        return OverlapResult(
            offset_mm=float(lag * spacing_mm),
            offset_px=lag,
            peak_score=-1e9,
            second_score=-1e9,
            peak_ratio=0.0,
            low_confidence=True,
            profile_a=pa,
            profile_b=pb,
        )

    scores.sort(key=lambda t: t[0], reverse=True)
    best_score, best_lag = scores[0]
    second = scores[1][0] if len(scores) > 1 else -1e9
    peak_ratio = (best_score + 1.05) / (second + 1.05) if second > -1e8 else 10.0
    low = bool(best_score < 0.15 or peak_ratio < (1.0 / ambiguous_ratio))

    return OverlapResult(
        offset_mm=float(best_lag * spacing_mm),
        offset_px=int(best_lag),
        peak_score=float(best_score),
        second_score=float(second),
        peak_ratio=float(peak_ratio),
        low_confidence=low,
        profile_a=pa,
        profile_b=pb,
    )


def apply_flip(image: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """End-for-end flip along vertical (row) axis."""
    return np.flipud(image).copy(), np.flipud(mask).copy()


def apply_roll(
    image: np.ndarray,
    mask: np.ndarray,
    roll_deg: float,
    *,
    cval: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Small in-plane rotation about image center (proxy roll for MPR slabs)."""
    if abs(roll_deg) < 1e-6:
        return image, mask
    from core_stitch.geometry import rotate_image, rotate_mask

    return (
        rotate_image(image, roll_deg, cval=cval, order=1),
        rotate_mask(mask, roll_deg),
    )


def lateral_align_to(
    image: np.ndarray,
    mask: np.ndarray,
    geom: CoreGeometry,
    target_col: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Shift so centroid column matches target_col (pixels)."""
    dcol = target_col - geom.centroid_rc[1]
    if abs(dcol) < 1e-3:
        return image, mask
    return (
        shift_image(image, (0.0, dcol), cval=air_value(image), order=1),
        shift_mask(mask, (0.0, dcol)),
    )


def compose_canvases(
    image_a: np.ndarray,
    mask_a: np.ndarray,
    image_b: np.ndarray,
    mask_b: np.ndarray,
    lag_px: int,
    *,
    col_offset_px: int = 0,
    cval: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Place A and B on a shared canvas; B origin at (lag_px, col_offset_px) on A.

    Returns canvas_a, canvas_b, mask_a_c, mask_b_c (float HU / bool).
    """
    ha, wa = image_a.shape
    hb, wb = image_b.shape
    r0 = min(0, lag_px)
    r1 = max(ha, lag_px + hb)
    c0 = min(0, col_offset_px)
    c1 = max(wa, col_offset_px + wb)
    h = r1 - r0
    w = c1 - c0

    ca = np.full((h, w), cval, dtype=np.float32)
    cb = np.full((h, w), cval, dtype=np.float32)
    ma = np.zeros((h, w), dtype=bool)
    mb = np.zeros((h, w), dtype=bool)

    ar, ac = -r0, -c0
    br, bc = lag_px - r0, col_offset_px - c0
    ca[ar : ar + ha, ac : ac + wa] = image_a
    cb[br : br + hb, bc : bc + wb] = image_b
    ma[ar : ar + ha, ac : ac + wa] = mask_a
    mb[br : br + hb, bc : bc + wb] = mask_b
    return ca, cb, ma, mb


def _match_hu_b_to_a(
    canvas_a: np.ndarray,
    canvas_b: np.ndarray,
    both: np.ndarray,
    mask_b: np.ndarray,
) -> np.ndarray:
    """Affine-match B intensities to A on overlap; apply on all of mask_b."""
    if both.sum() < 64:
        return canvas_b
    a = canvas_a[both].astype(np.float64)
    b = canvas_b[both].astype(np.float64)
    sb = float(b.std())
    sa = float(a.std())
    if sb < 1e-3:
        return canvas_b
    scale = sa / sb
    offset = float(a.mean()) - scale * float(b.mean())
    out = canvas_b.astype(np.float32, copy=True)
    out[mask_b] = (canvas_b[mask_b] * scale + offset).astype(np.float32)
    return out


def _feather_weight_b(
    mask_a: np.ndarray,
    mask_b: np.ndarray,
    *,
    feather_px: float,
) -> np.ndarray:
    """
    Weight for B in [0,1]: 0 on A-only, 1 on B-only, smooth A→B along +row in overlap.
    """
    both = mask_a & mask_b
    w = np.zeros(mask_a.shape, dtype=np.float32)
    w[mask_b & ~mask_a] = 1.0
    if not both.any():
        return w
    rows = np.where(both.any(axis=1))[0]
    r0 = int(rows[0])
    r1 = int(rows[-1])
    span = max(r1 - r0, 1)
    # Base linear ramp through overlap band
    for r in rows:
        t = (r - r0) / span
        w[r, both[r]] = float(t)
    # Soften ramp (smoothstep) through overlap band
    if feather_px > 1:
        fp = float(feather_px)
        for r in rows:
            t = (r - r0) / span
            te = t * t * (3.0 - 2.0 * t)
            alpha = min(1.0, span / max(fp, 1.0))
            w[r, both[r]] = float((1.0 - alpha) * t + alpha * te)
    return np.clip(w, 0.0, 1.0)


# Overlap handling (Dragonfly-style). Keys are stable for sidecar/GUI.
FUSE_MODES = (
    "max",  # brighter / higher HU wins — avoids half-value seam if one side missing
    "mean",  # equal average in overlap
    "blend",  # opacity-weighted average (uses Opacity A/B in GUI)
    "feather",  # A→B ramp along core through overlap band
    "min",  # darker / lower HU wins
    "a",  # A wins overlap
    "b",  # B wins overlap
)


def fuse_preview(
    canvas_a: np.ndarray,
    canvas_b: np.ndarray,
    mask_a: np.ndarray,
    mask_b: np.ndarray,
    *,
    opacity_a: float = 1.0,
    opacity_b: float = 1.0,
    mode: str = "max",
    feather_mm: float = 0.0,
    spacing_mm: float = 1.0,
    match_hu: bool = False,
) -> np.ndarray:
    """
    Grayscale fuse for QC / export.

    Overlap modes (see FUSE_MODES): max | mean | blend | feather | min | a | b.
    Optional HU match of B→A on overlap before combining.
    """
    mode = (mode or "max").lower().strip()
    if mode not in FUSE_MODES:
        mode = "max"
    oa = float(np.clip(opacity_a, 0.0, 1.0))
    ob = float(np.clip(opacity_b, 0.0, 1.0))
    both = mask_a & mask_b
    only_a = mask_a & ~mask_b
    only_b = mask_b & ~mask_a

    cb = canvas_b
    if match_hu and both.any():
        cb = _match_hu_b_to_a(canvas_a, canvas_b, both, mask_b)

    # Prefer air-like fill over zeros outside masks (zeros caused dark seam rows)
    air = float(
        np.median(canvas_a[mask_a])
        if mask_a.any()
        else (np.median(canvas_a) if canvas_a.size else 0.0)
    )
    out = np.full(canvas_a.shape, air, dtype=np.float32)
    out[only_a] = canvas_a[only_a]
    out[only_b] = cb[only_b]

    if not both.any():
        return out

    if mode == "max":
        out[both] = np.maximum(canvas_a[both], cb[both])
    elif mode == "min":
        out[both] = np.minimum(canvas_a[both], cb[both])
    elif mode == "a":
        out[both] = canvas_a[both]
    elif mode == "b":
        out[both] = cb[both]
    elif mode == "mean":
        out[both] = 0.5 * (canvas_a[both] + cb[both])
    elif mode == "feather":
        fm = float(feather_mm) if feather_mm > 0 else max(4.0, 12.0 * float(spacing_mm))
        feather_px = max(1.0, fm / max(float(spacing_mm), 1e-6))
        wb = _feather_weight_b(mask_a, mask_b, feather_px=feather_px)
        wa = 1.0 - wb
        wa_o = wa * oa
        wb_o = wb * ob
        denom = wa_o + wb_o
        fused = np.full(canvas_a.shape, air, dtype=np.float32)
        ok = denom > 1e-6
        fused[ok] = (wa_o[ok] * canvas_a[ok] + wb_o[ok] * cb[ok]) / denom[ok]
        out[both] = fused[both]
    else:  # blend — opacity-weighted (equal if both 1)
        denom = oa + ob
        if denom <= 1e-6:
            out[both] = air
        else:
            out[both] = (oa * canvas_a[both] + ob * cb[both]) / denom
    return out


def alignment_rgb(
    canvas_a: np.ndarray,
    canvas_b: np.ndarray,
    mask_a: np.ndarray,
    mask_b: np.ndarray,
    *,
    lo: float,
    hi: float,
    opacity_a: float = 0.7,
    opacity_b: float = 0.7,
) -> np.ndarray:
    """A=magenta, B=green windowed overlay with per-layer opacity."""
    if hi <= lo:
        hi = lo + 1.0
    oa = float(np.clip(opacity_a, 0.0, 1.0))
    ob = float(np.clip(opacity_b, 0.0, 1.0))
    a = np.clip((canvas_a - lo) / (hi - lo), 0, 1) * 255 * oa
    b = np.clip((canvas_b - lo) / (hi - lo), 0, 1) * 255 * ob
    a = np.where(mask_a, a, 0)
    b = np.where(mask_b, b, 0)
    rgb = np.zeros((*canvas_a.shape, 3), dtype=np.float32)
    rgb[..., 0] = a
    rgb[..., 2] = a
    rgb[..., 1] = b
    return np.clip(rgb, 0, 255).astype(np.uint8)
