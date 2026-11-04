"""Load a DICOM series folder and produce a core-section display image."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pydicom

from core_stitch.load import MprSeries, load_series
from dicom_multi_view.detect import SeriesDetect, detect_from_dataset, kind_from_tags

try:
    import pylibjpeg  # noqa: F401
except ImportError:
    pass


@dataclass
class LoadedSection:
    """One loaded series ready for multi-view display."""

    path: Path
    label: str
    series: MprSeries
    detect: SeriesDetect
    # Display: rows = down-core depth, cols = across core
    image_hu: np.ndarray
    depth_mm_per_px: float
    across_mm_per_px: float
    depth_extent_mm: float
    view_mode: str  # "coronal_mpr" | "axial_reformat" | "other_mid"
    plane_index: int
    plane_count: int
    notes: list[str] = field(default_factory=list)
    core_name: str = ""
    section_name: str = ""
    # Depth labeling (always stored in cm for the UI)
    top_depth_cm: float = 0.0
    # If True, continuous sync will not overwrite top_depth_cm
    manual_top: bool = False
    # Core sediment bounds as insets from image ends (cm). Scale ticks and
    # continuous chaining use the span between these markers, not full image.
    core_top_inset_cm: float = 0.0
    core_bottom_inset_cm: float = 0.0
    # Greyscale display window (HU); None → auto 1–99% when composing
    win_lo: float | None = None
    win_hi: float | None = None

    @property
    def depth_extent_cm(self) -> float:
        return self.depth_extent_mm / 10.0

    @property
    def core_length_cm(self) -> float:
        """Sediment length between core-top and core-bottom markers (cm)."""
        return max(
            0.0,
            self.depth_extent_cm - self.core_top_inset_cm - self.core_bottom_inset_cm,
        )

    def clamp_core_insets(self) -> None:
        """Keep insets non-negative and within the image depth extent."""
        extent = max(self.depth_extent_cm, 0.0)
        top = max(0.0, float(self.core_top_inset_cm))
        bottom = max(0.0, float(self.core_bottom_inset_cm))
        if top + bottom > extent:
            # Preserve ratio when both overshoot; otherwise clip the mover.
            if extent <= 0:
                top = bottom = 0.0
            else:
                scale = extent / (top + bottom)
                top *= scale
                bottom *= scale
        self.core_top_inset_cm = top
        self.core_bottom_inset_cm = bottom

    @property
    def display_label(self) -> str:
        parts = [p for p in (self.core_name.strip(), self.section_name.strip()) if p]
        return " · ".join(parts) if parts else self.label

    @property
    def badge_compression(self) -> str:
        return self.detect.compression

    @property
    def badge_geometry(self) -> str:
        if self.detect.is_coronal_mpr or self.view_mode == "coronal_mpr":
            return "coronal MPR"
        if self.view_mode == "axial_reformat" or self.detect.is_axial or self.detect.plane == "axial":
            return "axial → coronal"
        return f"{self.detect.plane} ({self.detect.kind})"

    @property
    def slider_help(self) -> str:
        """Short mapping for the cut/frame slider."""
        if self.view_mode == "coronal_mpr":
            return (
                "Slider = MPR frame (parallel coronal cuts through the core diameter)."
            )
        if self.view_mode == "axial_reformat":
            return (
                "Slider = coronal cut through the axial brick (Y index); "
                "image rows are down-core (section top at top)."
            )
        return "Slider = stack frame index."

    def physical_size_mm(self) -> tuple[float, float]:
        """(height_mm, width_mm) of the current display image."""
        h, w = self.image_hu.shape
        return float(h * self.depth_mm_per_px), float(w * self.across_mm_per_px)


def suggest_labels(series_dir: Path | str) -> tuple[str, str]:
    """Auto-suggest (core_name, section_name) from path parts."""
    series_dir = Path(series_dir)
    section = series_dir.name
    core = ""
    for part in series_dir.parts:
        if re.search(r"(JC|TC|GC)-\d", part, re.I):
            core = part
            break
    if not core:
        parent = series_dir.parent
        if parent.name.upper().startswith("ST") and parent.parent.name:
            core = parent.parent.name
        else:
            core = parent.name if parent.name else series_dir.name
    return core, section


def _probe_detect(series_dir: Path) -> SeriesDetect:
    series_dir = Path(series_dir)
    for path in sorted(series_dir.iterdir()):
        if not path.is_file():
            continue
        try:
            ds = pydicom.dcmread(path, stop_before_pixels=True, force=True)
        except Exception:
            continue
        if not (
            hasattr(ds, "SOPClassUID")
            or "PixelData" in ds
            or getattr(ds, "Modality", None)
        ):
            continue
        return detect_from_dataset(ds, series_dir)
    return SeriesDetect(
        plane="unknown",
        kind="other",
        compression="unknown",
        compression_detail="no_header",
        transfer_syntax="",
        series_description="",
        image_type="",
        is_coronal_mpr=False,
        is_axial=False,
    )


def _detect_from_series(series: MprSeries, series_dir: Path) -> SeriesDetect:
    """Classify using a frame from the loaded primary group (not a stray scout)."""
    path = series.slice_paths[0] if series.slice_paths else None
    if path is not None:
        try:
            ds = pydicom.dcmread(path, stop_before_pixels=True, force=True)
            det = detect_from_dataset(ds, series_dir)
            desc = series.series_description or det.series_description
            if desc and desc != det.series_description:
                kind = kind_from_tags(desc, det.image_type, det.plane)
                is_coronal_mpr = kind in {"coron_mpr", "coron_top", "coron_btm"} or (
                    det.plane == "coronal"
                    and ("MPR" in det.image_type.upper() or "mpr" in desc.lower())
                )
                return SeriesDetect(
                    plane=det.plane,
                    kind=kind,
                    compression=det.compression,
                    compression_detail=det.compression_detail,
                    transfer_syntax=series.transfer_syntax or det.transfer_syntax,
                    series_description=desc,
                    image_type=det.image_type,
                    is_coronal_mpr=is_coronal_mpr,
                    is_axial=(det.plane == "axial" or kind in {"axial", "axial_mpr"})
                    and not is_coronal_mpr,
                )
            return SeriesDetect(
                plane=det.plane,
                kind=det.kind,
                compression=det.compression,
                compression_detail=det.compression_detail,
                transfer_syntax=series.transfer_syntax or det.transfer_syntax,
                series_description=desc,
                image_type=det.image_type,
                is_coronal_mpr=det.is_coronal_mpr,
                is_axial=det.is_axial,
            )
        except Exception:
            pass
    return _probe_detect(series_dir)


def _window_hu(img: np.ndarray, lo_pct: float = 1.0, hi_pct: float = 99.0) -> tuple[float, float]:
    finite = img[np.isfinite(img)]
    if finite.size == 0:
        return 0.0, 1.0
    return float(np.percentile(finite, lo_pct)), float(np.percentile(finite, hi_pct))


def hu_to_u8(img: np.ndarray, lo: float | None = None, hi: float | None = None) -> np.ndarray:
    if lo is None or hi is None:
        a, b = _window_hu(img)
        lo = a if lo is None else lo
        hi = b if hi is None else hi
    if hi <= lo:
        hi = lo + 1.0
    clipped = np.clip(img, lo, hi)
    return ((clipped - lo) / (hi - lo) * 255.0).astype(np.uint8)


def auto_window_hu(img: np.ndarray) -> tuple[float, float]:
    """Default 1–99% HU window for display."""
    return _window_hu(img, 1.0, 99.0)


def coronal_from_axial(
    volume: np.ndarray,
    y_index: int | None = None,
) -> tuple[np.ndarray, int]:
    """
    Axial brick (Z,Y,X) → coronal plane (Z,X); Y selects cut through diameter.

    Rows increase downcore with section top at image top (flip Z so depth
    increases downward on screen, matching the cm scale bar).
    """
    nz, ny, nx = volume.shape
    if y_index is None:
        y_index = ny // 2
    y_index = int(np.clip(y_index, 0, max(ny - 1, 0)))
    # IPP-sorted Z often has the section top at high index for Aquilion cores;
    # flip so top-of-section is at row 0 (matches typical core photos).
    img = np.asarray(volume[::-1, y_index, :], dtype=np.float32)
    return img, y_index


def mid_coronal_mpr(volume: np.ndarray, z_index: int | None = None) -> tuple[np.ndarray, int]:
    """Coronal MPR stack frame (Z,Y,X)[z] → (Y,X) as stored (already coronal)."""
    nz = volume.shape[0]
    if z_index is None:
        z_index = nz // 2
    z_index = int(np.clip(z_index, 0, max(nz - 1, 0)))
    return np.asarray(volume[z_index], dtype=np.float32), z_index


def _looks_like_coronal_mpr_slab(volume: np.ndarray) -> bool:
    """Thin stack + large in-plane matrix → Coron TOP/BTM-style MPR, not axial brick."""
    nz, ny, nx = volume.shape
    return nz <= 160 and min(ny, nx) >= 640


def crop_air_margins(
    img: np.ndarray,
    *,
    hu_thresh: float = -500.0,
    pad_px: int = 8,
) -> np.ndarray:
    """
    Crop empty FOV so physical width ≈ core diameter (not full recon matrix).

    Keeps a small pad around content. Falls back to the original if empty.
    """
    mask = np.isfinite(img) & (img > hu_thresh)
    if not mask.any():
        return img
    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    r0 = max(0, int(rows[0]) - pad_px)
    r1 = min(img.shape[0], int(rows[-1]) + 1 + pad_px)
    c0 = max(0, int(cols[0]) - pad_px)
    c1 = min(img.shape[1], int(cols[-1]) + 1 + pad_px)
    if r1 - r0 < 8 or c1 - c0 < 8:
        return img
    return np.asarray(img[r0:r1, c0:c1], dtype=np.float32)


def _upright_core_section(
    img: np.ndarray,
    row_mm: float,
    col_mm: float,
) -> tuple[np.ndarray, float, float]:
    """
    Ensure downcore is image rows. Returns (image, depth_mm_per_px, across_mm_per_px).
    """
    d_mm, a_mm = float(row_mm), float(col_mm)
    mask = np.isfinite(img) & (img > -500.0)
    if mask.any():
        rows = np.where(mask.any(axis=1))[0]
        cols = np.where(mask.any(axis=0))[0]
        row_span = int(rows[-1] - rows[0]) + 1
        col_span = int(cols[-1] - cols[0]) + 1
        # Prefer the longer content axis as depth (vertical).
        if col_span * a_mm > row_span * d_mm * 1.05:
            img = np.asarray(np.rot90(img, k=1), dtype=np.float32)
            d_mm, a_mm = a_mm, d_mm
    return img, d_mm, a_mm


def build_display(
    series: MprSeries,
    detect: SeriesDetect,
    plane_index: int | None = None,
) -> tuple[np.ndarray, float, float, str, int, int, list[str]]:
    """
    Vertical core-section image + spacing.

    Coronal MPR: one stack frame (parallel cut through diameter); depth = in-plane rows.
    Axial: coronal reformat at Y; depth = slice axis Z (section top at image top).
    """
    vol = series.volume
    z_mm, row_mm, col_mm = series.spacing_mm
    notes: list[str] = []

    if detect.compression == "compressed":
        notes.append(
            f"Compressed ({detect.compression_detail}); decoded on read via pydicom/pylibjpeg."
        )

    slab = _looks_like_coronal_mpr_slab(vol)
    use_coronal_frame = (
        detect.is_coronal_mpr
        or detect.plane == "coronal"
        or (
            slab
            and detect.kind
            in {"coron_mpr", "coron_top", "coron_btm", "coronal", "mpr_other"}
        )
        or (slab and not detect.is_axial)
    )

    if use_coronal_frame:
        # Never axial→coronal reformat an already-coronal MPR stack.
        img, idx = mid_coronal_mpr(vol, plane_index)
        img, d_mm, a_mm = _upright_core_section(img, row_mm, col_mm)
        img = crop_air_margins(img)
        notes.append(
            "Coronal MPR: slider browses parallel cuts (stack frames) through the core."
        )
        return img, d_mm, a_mm, "coronal_mpr", idx, vol.shape[0], notes

    if detect.is_axial or detect.plane == "axial":
        img, idx = coronal_from_axial(vol, plane_index)
        img = crop_air_margins(img)
        notes.append(
            "Axial brick: coronal reformat; slider moves the cut through the core diameter (Y)."
        )
        return img, float(z_mm), float(col_mm), "axial_reformat", idx, vol.shape[1], notes

    img, idx = mid_coronal_mpr(vol, plane_index)
    img, d_mm, a_mm = _upright_core_section(img, row_mm, col_mm)
    img = crop_air_margins(img)
    notes.append(f"Unrecognized plane ({detect.plane}/{detect.kind}); mid stack slice.")
    return img, d_mm, a_mm, "other_mid", idx, vol.shape[0], notes


def load_section(
    series_dir: Path | str,
    label: str | None = None,
    *,
    core_name: str | None = None,
    section_name: str | None = None,
    top_depth_cm: float = 0.0,
    manual_top: bool = False,
) -> LoadedSection:
    series_dir = Path(series_dir)
    sug_core, sug_sec = suggest_labels(series_dir)
    core_name = sug_core if core_name is None else core_name
    section_name = sug_sec if section_name is None else section_name
    series = load_series(series_dir)

    _UNCOMP = {
        "1.2.840.10008.1.2",
        "1.2.840.10008.1.2.1",
        "1.2.840.10008.1.2.2",
    }
    detect = _detect_from_series(series, series_dir)
    compression = detect.compression
    if compression == "unknown" and series.transfer_syntax in _UNCOMP:
        compression = "uncompressed"
        detect = SeriesDetect(
            plane=detect.plane,
            kind=detect.kind,
            compression=compression,
            compression_detail=detect.compression_detail or series.transfer_syntax,
            transfer_syntax=series.transfer_syntax or detect.transfer_syntax,
            series_description=series.series_description or detect.series_description,
            image_type=detect.image_type,
            is_coronal_mpr=detect.is_coronal_mpr,
            is_axial=detect.is_axial,
        )
    elif series.transfer_syntax and detect.transfer_syntax != series.transfer_syntax:
        detect = SeriesDetect(
            plane=detect.plane,
            kind=detect.kind,
            compression=detect.compression,
            compression_detail=detect.compression_detail,
            transfer_syntax=series.transfer_syntax,
            series_description=series.series_description or detect.series_description,
            image_type=detect.image_type,
            is_coronal_mpr=detect.is_coronal_mpr,
            is_axial=detect.is_axial,
        )

    img, d_mm, a_mm, mode, pidx, pcount, notes = build_display(series, detect)
    wlo, whi = auto_window_hu(img)
    name = label or " · ".join(p for p in (core_name, section_name) if p) or series_dir.name
    return LoadedSection(
        path=series_dir,
        label=name,
        series=series,
        detect=detect,
        image_hu=img,
        depth_mm_per_px=d_mm,
        across_mm_per_px=a_mm,
        depth_extent_mm=float(img.shape[0] * d_mm),
        view_mode=mode,
        plane_index=pidx,
        plane_count=pcount,
        notes=notes,
        core_name=core_name or "",
        section_name=section_name or "",
        top_depth_cm=float(top_depth_cm),
        manual_top=bool(manual_top),
        win_lo=wlo,
        win_hi=whi,
    )


def set_plane(section: LoadedSection, plane_index: int) -> LoadedSection:
    """Rebuild display for a new MPR frame or axial coronal-cut index."""
    img, d_mm, a_mm, mode, pidx, pcount, notes = build_display(
        section.series, section.detect, plane_index
    )
    wlo, whi = section.win_lo, section.win_hi
    if wlo is None or whi is None:
        wlo, whi = auto_window_hu(img)
    return LoadedSection(
        path=section.path,
        label=section.label,
        series=section.series,
        detect=section.detect,
        image_hu=img,
        depth_mm_per_px=d_mm,
        across_mm_per_px=a_mm,
        depth_extent_mm=float(img.shape[0] * d_mm),
        view_mode=mode,
        plane_index=pidx,
        plane_count=pcount,
        notes=notes or section.notes,
        core_name=section.core_name,
        section_name=section.section_name,
        top_depth_cm=section.top_depth_cm,
        manual_top=section.manual_top,
        core_top_inset_cm=section.core_top_inset_cm,
        core_bottom_inset_cm=section.core_bottom_inset_cm,
        win_lo=wlo,
        win_hi=whi,
    )


def sync_continuous_tops(sections: list[LoadedSection]) -> None:
    """
    Set non-manual tops so continuous depth chains by core length in list order.

    running_cm starts at 0; each auto section gets top = running; then
    running += core_length_cm (span between core-top / core-bottom markers).
    Manual tops are kept; running continues from manual_top + core_length.
    """
    running = 0.0
    for sec in sections:
        sec.clamp_core_insets()
        if not sec.manual_top:
            sec.top_depth_cm = running
        running = sec.top_depth_cm + sec.core_length_cm
