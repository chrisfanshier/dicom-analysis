"""Single-series run-through for sections that need no stitching.

Same crop-to-core, greyscale window and file products as the stitched export
(windowed JPEG, 8-bit LZW TIFF, 16-bit HU TIFF, JSON sidecar), without a join.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image

from core_stitch.load import MprSeries, load_series
from core_stitch.scale import auto_bounds, core_scale_block, dpi_for
from core_stitch.pipeline import HalfPrepared, _prepare_half, core_crop_box, crop_to_box


@dataclass
class SingleState:
    series: MprSeries
    half: HalfPrepared
    frame: int
    window_lo: float | None = None
    window_hi: float | None = None
    warnings: list[str] = field(default_factory=list)

    @property
    def n_frames(self) -> int:
        return int(self.series.volume.shape[0])

    @property
    def spacing_mm(self) -> float:
        return self.half.target_mm


def prepare_single(
    path: Path | str, series_uid: str | None = None, frame: int | None = None
) -> SingleState:
    series = load_series(path, series_uid)
    nz = int(series.volume.shape[0])
    f = nz // 2 if frame is None else int(np.clip(frame, 0, nz - 1))
    target = float(min(series.spacing_mm[1], series.spacing_mm[2]))
    half, warns = _prepare_half(series, target, frame=f)
    st = SingleState(series=series, half=half, frame=f, warnings=warns)
    auto_window(st)
    return st


def set_frame(state: SingleState, frame: int) -> None:
    nz = state.n_frames
    f = int(np.clip(frame, 0, nz - 1))
    if f == state.frame:
        return
    half, warns = _prepare_half(state.series, state.spacing_mm, frame=f)
    state.half, state.frame, state.warnings = half, f, warns


def auto_window(state: SingleState) -> tuple[float, float]:
    vals = state.half.image[state.half.mask]
    if vals.size == 0:
        vals = state.half.image.ravel()
    lo, hi = float(np.percentile(vals, 1)), float(np.percentile(vals, 99))
    state.window_lo, state.window_hi = lo, hi
    return lo, hi


def window_limits(state: SingleState) -> tuple[float, float]:
    if state.window_lo is None or state.window_hi is None:
        return auto_window(state)
    lo, hi = float(state.window_lo), float(state.window_hi)
    return (lo, lo + 1.0) if hi <= lo else (lo, hi)


def crop_box(state: SingleState, pad_mm: float = 3.0) -> tuple[int, int, int, int]:
    pad_px = max(2, int(round(pad_mm / max(state.spacing_mm, 1e-6))))
    return core_crop_box(state.half.mask, state.half.mask, pad_px=pad_px)


def gray_preview(state: SingleState, pad_mm: float = 3.0, crop: bool = True) -> np.ndarray:
    """8-bit greyscale (cropped to the core) using the current HU window."""
    lo, hi = window_limits(state)
    img = state.half.image
    if crop:
        img = crop_to_box(img, crop_box(state, pad_mm))
    return np.clip((img - lo) / (hi - lo) * 255.0, 0, 255).astype(np.uint8)


def sidecar_single(state: SingleState) -> dict:
    s = state.series
    lo, hi = window_limits(state)
    return {
        "version": "0.1.0",
        "unstitched": True,
        "series": {
            "series_uid": s.series_uid,
            "series_description": s.series_description,
            "n_slices": state.n_frames,
            "spacing_mm_zrc": list(s.spacing_mm),
        },
        "target_mm": state.spacing_mm,
        "frame": state.frame,
        "upright_deg": state.half.upright_deg,
        "window_lo": lo,
        "window_hi": hi,
        "warnings": list(state.warnings),
    }


def export_single(
    state: SingleState,
    out_dir: Path | str,
    *,
    stem: str = "single",
    jpeg: bool = True,
    tiff: bool = True,
    json_sidecar: bool = True,
    pad_mm: float = 3.0,
    extra_meta: dict | None = None,
    core_bounds: tuple[float, float, str] | None = None,
    scale_out: dict | None = None,
) -> dict[str, Path]:
    if not any((jpeg, tiff, json_sidecar)):
        raise ValueError("Select at least one export type")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}

    meta = sidecar_single(state)
    if extra_meta:
        meta.update(extra_meta)
    meta["exported_at"] = datetime.now(timezone.utc).isoformat()
    meta["export_jpeg_pad_mm"] = pad_mm
    box = crop_box(state, pad_mm)
    meta["export_crop_box_r0r1c0c1"] = [int(x) for x in box]

    gray = gray_preview(state, pad_mm)
    meta["export_shape_px"] = [int(gray.shape[0]), int(gray.shape[1])]
    if core_bounds is None:
        t_cm, b_cm = auto_bounds(state.half.mask, box, state.spacing_mm)
        core_bounds = (t_cm, b_cm, "auto")
    meta["core_scale"] = core_scale_block(
        gray.shape[:2],
        state.spacing_mm,
        core_bounds[0],
        core_bounds[1],
        source=core_bounds[2],
        expected_length_cm=(extra_meta or {}).get("catalog_section_length_cm"),
    )
    if scale_out is not None:
        scale_out.update(meta['core_scale'])
    dpi = (dpi_for(state.spacing_mm),) * 2

    if jpeg:
        p = out_dir / f"{stem}.jpg"
        Image.fromarray(gray, mode="L").save(p, format="JPEG", quality=95, optimize=True, dpi=dpi)
        meta["export_jpeg"] = p.name
        paths["jpeg"] = p
    if tiff:
        p = out_dir / f"{stem}.tif"
        Image.fromarray(gray, mode="L").save(p, format="TIFF", compression="tiff_lzw", dpi=dpi)
        meta["export_tiff"] = p.name
        paths["tiff"] = p
        hu = crop_to_box(state.half.image, box)
        hu_min = float(np.min(hu))
        hu16 = np.clip(np.rint(hu - hu_min), 0, 65535).astype(np.uint16)
        p16 = out_dir / f"{stem}_hu.tif"
        Image.fromarray(hu16, mode="I;16").save(p16, format="TIFF", compression="tiff_lzw", dpi=dpi)
        meta["export_tiff_hu"] = p16.name
        meta["export_tiff_hu_offset"] = hu_min
        meta["export_tiff_hu_spacing_mm"] = state.spacing_mm
        paths["tiff_hu"] = p16
    if json_sidecar:
        p = out_dir / f"{stem}.json"
        p.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        paths["json"] = p
    return paths
