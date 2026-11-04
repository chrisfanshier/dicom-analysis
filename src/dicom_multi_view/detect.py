"""Header-based plane / compression / series-kind detection.

Mirrors logic from ``scripts/inventory_dicom_types.py`` so multi-view badges
match inventory classifications without importing the script.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from pydicom.dataset import Dataset

_UNCOMPRESSED_TS = {
    "1.2.840.10008.1.2",
    "1.2.840.10008.1.2.1",
    "1.2.840.10008.1.2.2",
}
_COMPRESSED_TS_LABELS = {
    "1.2.840.10008.1.2.4.70": "jpeg_lossless",
    "1.2.840.10008.1.2.4.57": "jpeg_lossless",
    "1.2.840.10008.1.2.4.90": "jpeg2000_lossless",
    "1.2.840.10008.1.2.4.91": "jpeg2000",
    "1.2.840.10008.1.2.4.50": "jpeg_baseline",
    "1.2.840.10008.1.2.4.51": "jpeg_extended",
    "1.2.840.10008.1.2.5": "rle",
    "1.2.840.10008.1.2.4.80": "jpegls_lossless",
    "1.2.840.10008.1.2.4.81": "jpegls",
}


@dataclass(frozen=True)
class SeriesDetect:
    plane: str  # axial | coronal | sagittal | oblique | unknown
    kind: str  # coron_mpr | axial | coronal | …
    compression: str  # uncompressed | compressed | unknown
    compression_detail: str
    transfer_syntax: str
    series_description: str
    image_type: str
    is_coronal_mpr: bool
    is_axial: bool


def compression_from_ts(ts: str, folder: Path | None = None) -> tuple[str, str]:
    """Return (bucket, detail) for TransferSyntaxUID (+ optional path hint)."""
    ts = (ts or "").strip()
    path_hint = False
    if folder is not None:
        path_hint = any(part.lower() == "uncompressed" for part in Path(folder).parts)
    if ts in _UNCOMPRESSED_TS:
        detail = {
            "1.2.840.10008.1.2": "implicit_le",
            "1.2.840.10008.1.2.1": "explicit_le",
            "1.2.840.10008.1.2.2": "explicit_be",
        }.get(ts, "uncompressed")
        if path_hint:
            detail = f"{detail}+path_uncompressed"
        return "uncompressed", detail
    if ts in _COMPRESSED_TS_LABELS:
        return "compressed", _COMPRESSED_TS_LABELS[ts]
    if ts.startswith("1.2.840.10008.1.2.4"):
        return "compressed", f"jpeg_family:{ts}"
    if ts.startswith("1.2.840.10008.1.2"):
        return "uncompressed", ts or "uncompressed_other"
    if path_hint and not ts:
        return "uncompressed", "path_uncompressed"
    if not ts:
        return "unknown", "no_transfer_syntax"
    return "unknown", ts


def plane_from_iop(iop) -> str:
    """Classify axial / coronal / sagittal / oblique from ImageOrientationPatient."""
    if iop is None or len(iop) < 6:
        return "unknown"
    try:
        r = np.array([float(iop[0]), float(iop[1]), float(iop[2])], dtype=np.float64)
        c = np.array([float(iop[3]), float(iop[4]), float(iop[5])], dtype=np.float64)
    except (TypeError, ValueError):
        return "unknown"
    n = np.cross(r, c)
    n_abs = np.abs(n)
    axis = int(np.argmax(n_abs))
    if n_abs[axis] < 0.8:
        return "oblique"
    return ("sagittal", "coronal", "axial")[axis]


def kind_from_tags(desc: str, image_type: str, plane: str) -> str:
    d = (desc or "").lower()
    it = (image_type or "").upper()
    is_mpr = "MPR" in it or "mpr" in d

    if re.search(r"\b(corona|coron|cor)\b.*\btop\b|\btop\b.*\b(corona|coron|cor)\b", d) or (
        re.search(r"\btop\b", d) and re.search(r"cor", d)
    ):
        return "coron_top"
    if re.search(r"\b(corona|coron|cor)\b.*\bbtm\b|\bbtm\b.*\b(corona|coron|cor)\b", d) or (
        re.search(r"\bbtm\b|\bbottom\b", d) and re.search(r"cor", d)
    ):
        return "coron_btm"
    if is_mpr and plane == "coronal":
        return "coron_mpr"
    if is_mpr and plane == "sagittal":
        return "sagittal_mpr"
    if is_mpr and plane == "axial":
        return "axial_mpr"
    if plane == "axial" and ("ORIGINAL" in it or not is_mpr):
        return "axial"
    if plane == "sagittal":
        return "sagittal"
    if plane == "coronal":
        return "coronal"
    if plane == "oblique":
        return "oblique"
    if is_mpr:
        return "mpr_other"
    return "other"


def _image_type_str(ds: Dataset) -> str:
    itype = getattr(ds, "ImageType", None)
    if isinstance(itype, (list, tuple)):
        return "\\".join(str(x) for x in itype)
    return str(itype or "")


def detect_from_dataset(ds: Dataset, folder: Path | None = None) -> SeriesDetect:
    desc = str(getattr(ds, "SeriesDescription", "") or "").strip()
    itype = _image_type_str(ds)
    iop = getattr(ds, "ImageOrientationPatient", None)
    plane = plane_from_iop(iop)
    kind = kind_from_tags(desc, itype, plane)
    ts = ""
    try:
        ts = str(ds.file_meta.TransferSyntaxUID)  # type: ignore[attr-defined]
    except Exception:
        pass
    compression, detail = compression_from_ts(ts, folder)
    is_coronal_mpr = kind in {"coron_mpr", "coron_top", "coron_btm"} or (
        plane == "coronal" and ("MPR" in itype.upper() or "mpr" in desc.lower())
    )
    is_axial = plane == "axial" or kind in {"axial", "axial_mpr"}
    return SeriesDetect(
        plane=plane,
        kind=kind,
        compression=compression,
        compression_detail=detail,
        transfer_syntax=ts,
        series_description=desc,
        image_type=itype,
        is_coronal_mpr=is_coronal_mpr,
        is_axial=is_axial and not is_coronal_mpr,
    )
