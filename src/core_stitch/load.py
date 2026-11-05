"""Load Coron MPR DICOM series into HU stacks."""

from __future__ import annotations

import warnings
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pydicom
from pydicom.dataset import Dataset
from pydicom.uid import ImplicitVRLittleEndian


@dataclass(frozen=True)
class MprSeries:
    volume: np.ndarray  # (Z, Y, X) float32 HU
    spacing_mm: tuple[float, float, float]  # slice, row, col
    path: Path
    slice_paths: tuple[Path, ...]  # sorted paths matching volume[z]
    series_uid: str
    series_description: str
    transfer_syntax: str
    frame_of_reference_uid: str


def _has_pixels(ds: Dataset) -> bool:
    return "PixelData" in ds or "FloatPixelData" in ds or "DoubleFloatPixelData" in ds


def _read(path: Path) -> Dataset | None:
    try:
        ds = pydicom.dcmread(path, force=True)
    except Exception:
        return None
    if not _has_pixels(ds):
        return None
    return ds


def discover(series_dir: Path) -> list[tuple[Path, Dataset]]:
    series_dir = Path(series_dir)
    if not series_dir.is_dir():
        raise NotADirectoryError(series_dir)
    found: list[tuple[Path, Dataset]] = []
    for path in sorted(series_dir.iterdir()):
        if path.is_file():
            ds = _read(path)
            if ds is not None:
                found.append((path, ds))
    if not found:
        raise FileNotFoundError(f"No DICOM with pixels in {series_dir}")
    return found


def select_primary_group(
    items: list[tuple[Path, Dataset]],
) -> tuple[str, list[tuple[Path, Dataset]]]:
    by_key: dict[tuple[str, int, int], list[tuple[Path, Dataset]]] = defaultdict(list)
    for path, ds in items:
        uid = str(getattr(ds, "SeriesInstanceUID", "UNKNOWN"))
        rows = int(getattr(ds, "Rows", 0) or 0)
        cols = int(getattr(ds, "Columns", 0) or 0)
        by_key[(uid, rows, cols)].append((path, ds))

    (uid, rows, cols), group = max(
        by_key.items(), key=lambda kv: (len(kv[1]), kv[0][1] * kv[0][2])
    )
    if len(by_key) > 1:
        summary = ", ".join(
            f"{r}x{c}x{len(g)}"
            for (_u, r, c), g in sorted(by_key.items(), key=lambda kv: -len(kv[1]))
        )
        warnings.warn(
            f"Multiple series/shape groups ({summary}); using {rows}x{cols} ({len(group)})",
            UserWarning,
            stacklevel=2,
        )
    return uid, group


def _slice_normal(ds: Dataset) -> np.ndarray | None:
    iop = getattr(ds, "ImageOrientationPatient", None)
    if iop is None or len(iop) != 6:
        return None
    row = np.asarray(iop[:3], dtype=np.float64)
    col = np.asarray(iop[3:], dtype=np.float64)
    n = np.cross(row, col)
    norm = np.linalg.norm(n)
    if norm == 0:
        return None
    return n / norm


def sort_slices(items: list[tuple[Path, Dataset]]) -> list[tuple[Path, Dataset]]:
    if not items:
        return items
    normal = _slice_normal(items[0][1])

    def ipp_key(item: tuple[Path, Dataset]) -> float | None:
        ipp = getattr(item[1], "ImagePositionPatient", None)
        if ipp is None or normal is None:
            return None
        return float(np.dot(np.asarray(ipp, dtype=np.float64), normal))

    if all(ipp_key(i) is not None for i in items):
        return sorted(items, key=lambda i: ipp_key(i))  # type: ignore[arg-type, return-value]

    def inst(item: tuple[Path, Dataset]) -> int:
        try:
            return int(getattr(item[1], "InstanceNumber", 0) or 0)
        except (TypeError, ValueError):
            return 0

    return sorted(items, key=inst)


def _hu(ds: Dataset) -> np.ndarray:
    if not hasattr(ds, "file_meta") or ds.file_meta is None:
        ds.file_meta = pydicom.dataset.FileMetaDataset()
    if not getattr(ds.file_meta, "TransferSyntaxUID", None):
        ds.file_meta.TransferSyntaxUID = ImplicitVRLittleEndian
    arr = ds.pixel_array.astype(np.float32, copy=False)
    slope = float(getattr(ds, "RescaleSlope", 1.0) or 1.0)
    intercept = float(getattr(ds, "RescaleIntercept", 0.0) or 0.0)
    if slope != 1.0 or intercept != 0.0:
        arr = arr * slope + intercept
    return arr


def _spacing(items: list[tuple[Path, Dataset]]) -> tuple[float, float, float]:
    sample = items[0][1]
    ps = getattr(sample, "PixelSpacing", None)
    if ps is not None and len(ps) >= 2:
        row_mm, col_mm = float(ps[0]), float(ps[1])
    else:
        row_mm = col_mm = 1.0
        warnings.warn("PixelSpacing missing; using 1 mm", UserWarning)

    # Slice spacing from IPP if possible
    normal = _slice_normal(sample)
    z_mm = 2.0
    if normal is not None and len(items) >= 2:
        pos = []
        for _, ds in items:
            ipp = getattr(ds, "ImagePositionPatient", None)
            if ipp is not None:
                pos.append(float(np.dot(np.asarray(ipp, float), normal)))
        if len(pos) >= 2:
            diffs = np.diff(sorted(pos))
            diffs = diffs[np.abs(diffs) > 1e-6]
            if diffs.size:
                z_mm = float(np.median(np.abs(diffs)))
    else:
        st = getattr(sample, "SliceThickness", None)
        if st is not None:
            z_mm = float(st)
    return (z_mm, row_mm, col_mm)


def select_uid_group(
    items: list[tuple[Path, Dataset]], series_uid: str
) -> tuple[str, list[tuple[Path, Dataset]]]:
    """Keep the files of one SeriesInstanceUID (largest shape group if it has several).

    The release catalog stores only the last 12 characters of each UID
    (e.g. ``33191.951896``), so a UID that ends with ``series_uid`` matches too.
    """
    def _hit(ds: Dataset) -> bool:
        uid = str(getattr(ds, "SeriesInstanceUID", ""))
        return uid == series_uid or (len(series_uid) >= 8 and uid.endswith(series_uid))

    mine = [(p, ds) for p, ds in items if _hit(ds)]
    if not mine:
        raise FileNotFoundError(f"Series {series_uid} not found among {len(items)} files")
    return select_primary_group(mine)


def load_series(series_dir: Path | str, series_uid: str | None = None) -> MprSeries:
    """Load one series from a folder. ``series_uid`` picks it when a folder holds several."""
    series_dir = Path(series_dir)
    items = discover(series_dir)
    if series_uid:
        uid, items = select_uid_group(items, series_uid)
    else:
        uid, items = select_primary_group(items)
    items = sort_slices(items)
    arrays = [_hu(ds) for _, ds in items]
    shapes = {a.shape for a in arrays}
    if len(shapes) != 1:
        raise ValueError(f"Inconsistent shapes after group select: {shapes}")
    volume = np.stack(arrays, axis=0)
    sample = items[0][1]
    ts = str(
        getattr(getattr(sample, "file_meta", None), "TransferSyntaxUID", "unknown")
    )
    return MprSeries(
        volume=volume,
        spacing_mm=_spacing(items),
        path=series_dir,
        slice_paths=tuple(p for p, _ in items),
        series_uid=uid,
        series_description=str(getattr(sample, "SeriesDescription", "")),
        transfer_syntax=ts,
        frame_of_reference_uid=str(getattr(sample, "FrameOfReferenceUID", "")),
    )
