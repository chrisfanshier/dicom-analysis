#!/usr/bin/env python3
"""
Sweep a folder tree for DICOM series and summarize how many of each type.

Typical layout under ct_snapshot:
  <cruise>/ctscan/<core>/ST1/SE1
  <cruise>/ctscan/.../PA*/ST*/SE*

Classifies each SeriesInstanceUID from header tags (no pixel decode):
  - modality, ImageType (MPR vs ORIGINAL), SeriesDescription
  - approximate plane from ImageOrientationPatient
  - coarse label: coron_top / coron_btm / coron_other / axial / sagittal / oblique / other
  - compression from TransferSyntaxUID (and path hint if under an "uncompressed" folder)

Usage:
  python scripts/inventory_dicom_types.py "D:\\temp\\ct_snapshots\\ct_snapshot"
  python scripts/inventory_dicom_types.py "D:\\temp\\ct_snapshots\\ct_snapshot" -o inventory.csv
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pydicom
from pydicom.errors import InvalidDicomError

# Skip obvious non-image junk
SKIP_SUFFIXES = {".txt", ".csv", ".json", ".xml", ".jpg", ".jpeg", ".png", ".tif", ".tiff", ".zip"}
SKIP_NAMES = {"dicomdir", "version", "readme"}


# Common DICOM transfer syntax UIDs → coarse compression bucket
_UNCOMPRESSED_TS = {
    "1.2.840.10008.1.2",  # Implicit VR Little Endian
    "1.2.840.10008.1.2.1",  # Explicit VR Little Endian
    "1.2.840.10008.1.2.2",  # Explicit VR Big Endian
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


def _compression_from_ts(ts: str, folder: Path) -> tuple[str, str]:
    """
    Returns (bucket, detail).
    bucket: uncompressed | compressed | unknown
    detail: e.g. jpeg2000_lossless, explicit_le, path_uncompressed
    """
    ts = (ts or "").strip()
    path_hint = any(part.lower() == "uncompressed" for part in folder.parts)
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
        # other little-endian family — treat as uncompressed-ish
        return "uncompressed", ts or "uncompressed_other"
    if path_hint and not ts:
        return "uncompressed", "path_uncompressed"
    if not ts:
        return "unknown", "no_transfer_syntax"
    return "unknown", ts


@dataclass
class SeriesInfo:
    series_uid: str
    folder: str
    n_files: int
    modality: str
    series_description: str
    series_number: str
    image_type: str
    plane: str
    kind: str  # coron_top | coron_btm | coron_mpr | axial | sagittal | oblique | other
    rows: int
    columns: int
    slice_thickness_mm: str
    pixel_spacing_mm: str
    transfer_syntax: str
    compression: str  # uncompressed | compressed | unknown
    compression_detail: str
    cruise: str
    core_hint: str


def _is_candidate_file(path: Path) -> bool:
    name = path.name.lower()
    if name in SKIP_NAMES or name.startswith("."):
        return False
    if path.suffix.lower() in SKIP_SUFFIXES:
        return False
    return True


def _looks_like_dicom(path: Path) -> bool:
    """Cheap preamble check before full dcmread."""
    try:
        with path.open("rb") as f:
            head = f.read(132)
    except OSError:
        return False
    if len(head) >= 132 and head[128:132] == b"DICM":
        return True
    # Many Aquilion exports are Implicit VR without preamble — try dcmread later
    # Heuristic: IM*, no extension, or .dcm
    n = path.name.upper()
    if n.startswith("IM") or path.suffix.lower() in {".dcm", ".dicom", ""}:
        return True
    return False


def _plane_from_iop(iop) -> str:
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
    # Normal along patient axis → plane name
    return ("sagittal", "coronal", "axial")[axis]


def _kind_from_tags(desc: str, image_type: str, plane: str) -> str:
    d = (desc or "").lower()
    it = (image_type or "").upper()
    is_mpr = "MPR" in it or "mpr" in d

    # TOP / BTM (spellings vary: Corona, Cor, Coron)
    if re.search(r"\b(corona|coron|cor)\b.*\btop\b|\btop\b.*\b(corona|coron|cor)\b", d) or re.search(
        r"\btop\b", d
    ) and re.search(r"cor", d):
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


def _cruise_and_core(root: Path, folder: Path) -> tuple[str, str]:
    try:
        rel = folder.relative_to(root)
    except ValueError:
        return "", folder.name
    parts = rel.parts
    cruise = parts[0] if parts else ""
    # Prefer folder that looks like a core id (contains JC/TC or long name under ctscan)
    core = ""
    for p in parts:
        if re.search(r"(JC|TC|GC)-\d|ctscan$", p, re.I):
            if not p.lower().endswith("ctscan"):
                core = p
                break
    if not core and len(parts) >= 3:
        core = parts[2] if parts[1].lower() == "ctscan" else parts[-3]
    return cruise, core


def read_series_probe(folder: Path, files: list[Path]) -> SeriesInfo | None:
    """Read first readable DICOM in folder as series representative."""
    ds = None
    chosen = None
    for f in files[:40]:
        if not _looks_like_dicom(f):
            continue
        try:
            cand = pydicom.dcmread(f, stop_before_pixels=True, force=True)
        except (InvalidDicomError, OSError, Exception):
            continue
        if not (
            hasattr(cand, "SOPClassUID")
            or "PixelData" in cand
            or "FloatPixelData" in cand
            or getattr(cand, "Modality", None)
        ):
            continue
        ds = cand
        chosen = f
        break
    if ds is None or chosen is None:
        return None

    uid = str(getattr(ds, "SeriesInstanceUID", "") or f"NOUID:{folder}")
    desc = str(getattr(ds, "SeriesDescription", "") or "")
    itype = getattr(ds, "ImageType", None)
    if isinstance(itype, (list, tuple)):
        itype_s = "\\".join(str(x) for x in itype)
    else:
        itype_s = str(itype or "")
    iop = getattr(ds, "ImageOrientationPatient", None)
    plane = _plane_from_iop(iop)
    kind = _kind_from_tags(desc, itype_s, plane)
    ps = getattr(ds, "PixelSpacing", None)
    if ps is not None and len(ps) >= 2:
        ps_s = f"{float(ps[0]):.6g}\\{float(ps[1]):.6g}"
    else:
        ps_s = ""
    st = getattr(ds, "SliceThickness", None)
    st_s = f"{float(st):.4g}" if st is not None else ""
    ts = ""
    try:
        ts = str(ds.file_meta.TransferSyntaxUID)  # type: ignore[attr-defined]
    except Exception:
        pass
    compression, compression_detail = _compression_from_ts(ts, folder)

    return SeriesInfo(
        series_uid=uid,
        folder=str(folder),
        n_files=len(files),
        modality=str(getattr(ds, "Modality", "") or ""),
        series_description=desc.strip(),
        series_number=str(getattr(ds, "SeriesNumber", "") or ""),
        image_type=itype_s,
        plane=plane,
        kind=kind,
        rows=int(getattr(ds, "Rows", 0) or 0),
        columns=int(getattr(ds, "Columns", 0) or 0),
        slice_thickness_mm=st_s,
        pixel_spacing_mm=ps_s,
        transfer_syntax=ts,
        compression=compression,
        compression_detail=compression_detail,
        cruise="",
        core_hint="",
    )


def discover_series_folders(root: Path) -> list[Path]:
    """
    Directories that contain candidate DICOM files (non-recursive contents).
    Avoids treating parent folders that only hold subfolders as series.
    """
    import os

    found: list[Path] = []
    for dirpath, _dirnames, filenames in os.walk(root):
        p = Path(dirpath)
        files = [p / n for n in filenames if _is_candidate_file(p / n)]
        if not files:
            continue
        if any(_looks_like_dicom(f) for f in files[:20]):
            found.append(p)
    return found


def inventory(root: Path) -> list[SeriesInfo]:
    root = root.resolve()
    folders = discover_series_folders(root)
    # Key by UID + compression so compressed and uncompressed copies both count
    by_key: dict[str, SeriesInfo] = {}
    for folder in folders:
        files = sorted(
            [f for f in folder.iterdir() if f.is_file() and _is_candidate_file(f)],
            key=lambda x: x.name,
        )
        if not files:
            continue
        info = read_series_probe(folder, files)
        if info is None:
            continue
        cruise, core = _cruise_and_core(root, folder)
        info.cruise = cruise
        info.core_hint = core
        key = f"{info.series_uid}|{info.compression}|{info.compression_detail}"
        prev = by_key.get(key)
        if prev is None or info.n_files > prev.n_files:
            by_key[key] = info
    return sorted(
        by_key.values(),
        key=lambda s: (s.cruise, s.core_hint, s.kind, s.compression, s.folder),
    )


def print_summary(rows: list[SeriesInfo]) -> None:
    kind_c = Counter(r.kind for r in rows)
    plane_c = Counter(r.plane for r in rows)
    mod_c = Counter(r.modality or "?" for r in rows)
    comp_c = Counter(r.compression for r in rows)
    comp_detail_c = Counter(r.compression_detail for r in rows)
    mpr_c = Counter(
        "MPR" if "MPR" in (r.image_type or "").upper() else "non-MPR" for r in rows
    )
    # Cross-tab: kind × compression
    kind_comp = Counter(f"{r.kind} × {r.compression}" for r in rows)
    desc_c = Counter(
        _normalize_desc(r.series_description) for r in rows if r.series_description
    )

    print(f"\nSeries found: {len(rows)}")
    print(f"Files (sum of series folder counts): {sum(r.n_files for r in rows)}")
    print("\n=== By compression ===")
    for k, n in comp_c.most_common():
        print(f"  {k:16s}  {n:5d}")
    print("\n=== By compression detail (Transfer Syntax) ===")
    for k, n in comp_detail_c.most_common(20):
        print(f"  {k:28s}  {n:5d}")
    print("\n=== By kind ===")
    for k, n in kind_c.most_common():
        print(f"  {k:16s}  {n:5d}")
    print("\n=== Kind × compression ===")
    for k, n in kind_comp.most_common():
        print(f"  {k:36s}  {n:5d}")
    print("\n=== By plane (IOP) ===")
    for k, n in plane_c.most_common():
        print(f"  {k:16s}  {n:5d}")
    print("\n=== MPR vs other (ImageType) ===")
    for k, n in mpr_c.most_common():
        print(f"  {k:16s}  {n:5d}")
    print("\n=== Modality ===")
    for k, n in mod_c.most_common():
        print(f"  {k:16s}  {n:5d}")
    print("\n=== Top SeriesDescription patterns ===")
    for k, n in desc_c.most_common(25):
        print(f"  {n:5d}  {k}")


def _normalize_desc(desc: str) -> str:
    """Collapse TOP.1 / TOP.2 numbering noise for tallying."""
    d = re.sub(r"\s+", " ", desc.strip())
    d = re.sub(r"\.(?:\d+)\s*$", "", d)  # trailing .1 .2
    d = re.sub(r"(TOP|BTM|BOTTOM)\.\d+", r"\1", d, flags=re.I)
    return d


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Inventory DICOM series types under a tree")
    p.add_argument(
        "root",
        nargs="?",
        default=r"D:\temp\ct_snapshots\ct_snapshot",
        help="Root folder to sweep",
    )
    p.add_argument(
        "-o",
        "--output",
        type=str,
        default="",
        help="Write per-series CSV (default: <root>/dicom_inventory.csv)",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Optional max series folders to probe (debug)",
    )
    args = p.parse_args(argv)
    root = Path(args.root)
    if not root.is_dir():
        print(f"Not a directory: {root}", file=sys.stderr)
        return 2

    print(f"Scanning {root} …")
    rows = inventory(root)
    if args.limit and len(rows) > args.limit:
        rows = rows[: args.limit]

    print_summary(rows)

    out = Path(args.output) if args.output else root / "dicom_inventory.csv"
    fieldnames = list(asdict(rows[0]).keys()) if rows else [
        f.name for f in SeriesInfo.__dataclass_fields__.values()  # type: ignore[attr-defined]
    ]
    # dataclass fields order
    fieldnames = [
        "cruise",
        "core_hint",
        "kind",
        "plane",
        "modality",
        "series_description",
        "series_number",
        "image_type",
        "compression",
        "compression_detail",
        "n_files",
        "rows",
        "columns",
        "slice_thickness_mm",
        "pixel_spacing_mm",
        "transfer_syntax",
        "series_uid",
        "folder",
    ]
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(asdict(r))
    print(f"\nWrote {out} ({len(rows)} series)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
