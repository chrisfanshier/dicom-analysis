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

    @property
    def depth_extent_cm(self) -> float:
        return self.depth_extent_mm / 10.0

    @property
    def display_label(self) -> str:
        parts = [p for p in (self.core_name.strip(), self.section_name.strip()) if p]
        return " · ".join(parts) if parts else self.label

    @property
    def badge_compression(self) -> str:
        return self.detect.compression

    @property
    def badge_geometry(self) -> str:
        if self.detect.is_coronal_mpr:
            return "coronal MPR"
        if self.detect.is_axial or self.detect.plane == "axial":
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
                "image rows are down-core Z."
            )
        return "Slider = stack frame index."


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


def coronal_from_axial(
    volume: np.ndarray,
    y_index: int | None = None,
) -> tuple[np.ndarray, int]:
    """Axial brick (Z,Y,X) → coronal plane (Z,X); Y selects cut through diameter."""
    nz, ny, nx = volume.shape
    if y_index is None:
        y_index = ny // 2
    y_index = int(np.clip(y_index, 0, max(ny - 1, 0)))
    return np.asarray(volume[:, y_index, :], dtype=np.float32), y_index


def mid_coronal_mpr(volume: np.ndarray, z_index: int | None = None) -> tuple[np.ndarray, int]:
    """Coronal MPR stack frame (Z,Y,X)[z] → (Y,X)."""
    nz = volume.shape[0]
    if z_index is None:
        z_index = nz // 2
    z_index = int(np.clip(z_index, 0, max(nz - 1, 0)))
    return np.asarray(volume[z_index], dtype=np.float32), z_index


def build_display(
    series: MprSeries,
    detect: SeriesDetect,
    plane_index: int | None = None,
) -> tuple[np.ndarray, float, float, str, int, int, list[str]]:
    """
    Vertical core-section image + spacing.

    Coronal MPR: one stack frame (parallel cut through diameter); depth = in-plane rows.
    Axial: coronal reformat at Y; depth = slice axis Z.
    """
    vol = series.volume
    z_mm, row_mm, col_mm = series.spacing_mm
    notes: list[str] = []

    if detect.compression == "compressed":
        notes.append(
            f"Compressed ({detect.compression_detail}); decoded on read via pydicom/pylibjpeg."
        )

    if detect.is_coronal_mpr or detect.plane == "coronal":
        img, idx = mid_coronal_mpr(vol, plane_index)
        notes.append(
            "Coronal MPR: slider browses parallel cuts (stack frames) through the core."
        )
        return img, float(row_mm), float(col_mm), "coronal_mpr", idx, vol.shape[0], notes

    if detect.is_axial or detect.plane == "axial":
        img, idx = coronal_from_axial(vol, plane_index)
        notes.append(
            "Axial brick: coronal reformat; slider moves the cut through the core diameter (Y)."
        )
        return img, float(z_mm), float(col_mm), "axial_reformat", idx, vol.shape[1], notes

    img, idx = mid_coronal_mpr(vol, plane_index)
    notes.append(f"Unrecognized plane ({detect.plane}/{detect.kind}); mid stack slice.")
    return img, float(row_mm), float(col_mm), "other_mid", idx, vol.shape[0], notes


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
    probe = _probe_detect(series_dir)
    series = load_series(series_dir)

    _UNCOMP = {
        "1.2.840.10008.1.2",
        "1.2.840.10008.1.2.1",
        "1.2.840.10008.1.2.2",
    }
    compression = probe.compression
    if compression == "unknown" and series.transfer_syntax in _UNCOMP:
        compression = "uncompressed"

    detect = SeriesDetect(
        plane=probe.plane,
        kind=probe.kind,
        compression=compression,
        compression_detail=probe.compression_detail or series.transfer_syntax,
        transfer_syntax=series.transfer_syntax or probe.transfer_syntax,
        series_description=series.series_description or probe.series_description,
        image_type=probe.image_type,
        is_coronal_mpr=probe.is_coronal_mpr,
        is_axial=probe.is_axial,
    )
    if series.series_description and not detect.series_description:
        kind = kind_from_tags(series.series_description, detect.image_type, detect.plane)
        detect = SeriesDetect(
            plane=detect.plane,
            kind=kind,
            compression=detect.compression,
            compression_detail=detect.compression_detail,
            transfer_syntax=detect.transfer_syntax,
            series_description=series.series_description,
            image_type=detect.image_type,
            is_coronal_mpr=kind in {"coron_mpr", "coron_top", "coron_btm"},
            is_axial=detect.is_axial,
        )

    img, d_mm, a_mm, mode, pidx, pcount, notes = build_display(series, detect)
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
    )


def set_plane(section: LoadedSection, plane_index: int) -> LoadedSection:
    """Rebuild display for a new MPR frame or axial coronal-cut index."""
    img, d_mm, a_mm, mode, pidx, pcount, notes = build_display(
        section.series, section.detect, plane_index
    )
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
    )


def sync_continuous_tops(sections: list[LoadedSection]) -> None:
    """
    Set non-manual tops so continuous depth = sum of image lengths in list order.

    running_cm starts at 0; each auto section gets top = running; then
    running += extent. Manual tops are kept; running continues from
    max(manual_top + extent, …) so following autos still chain after them.
    """
    running = 0.0
    for sec in sections:
        if not sec.manual_top:
            sec.top_depth_cm = running
        running = sec.top_depth_cm + sec.depth_extent_cm
