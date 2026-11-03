#!/usr/bin/env python3
"""
Decompress DICOM files under a source tree → Explicit VR Little Endian.

Preserves relative folder structure under --out.

Usage:
  python scripts/decompress_dicom_tree.py ^
    "D:\\temp\\ct_snapshots\\ct_snapshot\\AR2307-moratorium\\ctscan" ^
    -o "D:\\temp\\uncompressed"
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pydicom
from pydicom.dataset import Dataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, generate_uid

# Ensure JPEG / JPEG2000 handlers are registered when available
try:
    import pylibjpeg  # noqa: F401
except ImportError:
    pass

_UNCOMPRESSED = {
    "1.2.840.10008.1.2",
    "1.2.840.10008.1.2.1",
    "1.2.840.10008.1.2.2",
}

SKIP_SUFFIXES = {
    ".txt",
    ".csv",
    ".json",
    ".xml",
    ".jpg",
    ".jpeg",
    ".png",
    ".tif",
    ".tiff",
    ".zip",
    ".pdf",
    ".avi",
    ".mp4",
}


def _is_dicom_candidate(path: Path) -> bool:
    if path.suffix.lower() in SKIP_SUFFIXES:
        return False
    name = path.name.lower()
    if name in {"dicomdir", "version", "readme"} or name.startswith("."):
        return False
    return True


def _looks_like_dicom(path: Path) -> bool:
    try:
        with path.open("rb") as f:
            head = f.read(132)
    except OSError:
        return False
    if len(head) >= 132 and head[128:132] == b"DICM":
        return True
    n = path.name.upper()
    return n.startswith("IM") or path.suffix.lower() in {".dcm", ".dicom", ""}


def decompress_file(src: Path, dst: Path) -> str:
    """
    Read src, write uncompressed Explicit VR LE to dst.
    Returns status: decompressed | copied_uncompressed | skipped | error:...
    """
    try:
        ds = pydicom.dcmread(src, force=True)
    except Exception as exc:
        return f"error:read:{exc}"

    if not (
        "PixelData" in ds
        or "FloatPixelData" in ds
        or "DoubleFloatPixelData" in ds
        or hasattr(ds, "SOPClassUID")
    ):
        return "skipped:not_image"

    try:
        ts = str(ds.file_meta.TransferSyntaxUID)
    except Exception:
        ts = ""

    dst.parent.mkdir(parents=True, exist_ok=True)

    if ts in _UNCOMPRESSED:
        # Already uncompressed — copy bytes as-is (faster, bit-identical)
        dst.write_bytes(src.read_bytes())
        return "copied_uncompressed"

    # Force pixel decode then re-encode uncompressed
    try:
        _ = ds.pixel_array  # decode compressed pixels into dataset
        ds.decompress()
    except Exception as exc:
        return f"error:decompress:{exc}"

    # Ensure file meta for Explicit VR LE
    if not hasattr(ds, "file_meta") or ds.file_meta is None:
        ds.file_meta = FileMetaDataset()
    ds.file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    if not getattr(ds.file_meta, "MediaStorageSOPClassUID", None):
        ds.file_meta.MediaStorageSOPClassUID = getattr(ds, "SOPClassUID", generate_uid())
    if not getattr(ds.file_meta, "MediaStorageSOPInstanceUID", None):
        ds.file_meta.MediaStorageSOPInstanceUID = getattr(
            ds, "SOPInstanceUID", generate_uid()
        )
    ds.is_little_endian = True
    ds.is_implicit_VR = False

    try:
        ds.save_as(dst, enforce_file_format=True)
    except TypeError:
        # older pydicom
        ds.save_as(dst, write_like_original=False)
    except Exception as exc:
        return f"error:write:{exc}"
    return "decompressed"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Decompress DICOM tree to Explicit VR LE")
    p.add_argument("source", type=str, help="Source parent folder")
    p.add_argument(
        "-o",
        "--out",
        type=str,
        required=True,
        help="Output parent folder (relative paths preserved)",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="List actions only",
    )
    p.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip files that already exist in the output tree",
    )
    args = p.parse_args(argv)

    src_root = Path(args.source).resolve()
    out_root = Path(args.out).resolve()
    if not src_root.is_dir():
        print(f"Not a directory: {src_root}", file=sys.stderr)
        return 2

    out_root.mkdir(parents=True, exist_ok=True)

    counts = {
        "decompressed": 0,
        "copied_uncompressed": 0,
        "skipped": 0,
        "skipped_existing": 0,
        "error": 0,
        "non_dicom": 0,
    }
    errors: list[str] = []

    files = [f for f in src_root.rglob("*") if f.is_file() and _is_dicom_candidate(f)]
    print(f"Found {len(files)} candidate files under {src_root}")
    print(f"Output -> {out_root}")

    for i, src in enumerate(files, 1):
        rel = src.relative_to(src_root)
        dst = out_root / rel
        if not _looks_like_dicom(src):
            counts["non_dicom"] += 1
            continue
        if args.dry_run:
            print(f"  would process {rel}")
            continue
        if args.skip_existing and dst.is_file() and dst.stat().st_size > 0:
            counts["skipped_existing"] += 1
            if i % 200 == 0 or i == len(files):
                print(
                    f"  ... {i}/{len(files)}  "
                    f"decomp={counts['decompressed']} "
                    f"copy={counts['copied_uncompressed']} "
                    f"skip_ex={counts['skipped_existing']} "
                    f"err={counts['error']}"
                )
            continue
        status = decompress_file(src, dst)
        if status.startswith("error"):
            counts["error"] += 1
            errors.append(f"{rel}: {status}")
            if counts["error"] <= 10:
                print(f"  FAIL {rel}: {status}")
        elif status.startswith("skipped"):
            counts["skipped"] += 1
        else:
            counts[status] = counts.get(status, 0) + 1
        if i % 200 == 0 or i == len(files):
            print(
                f"  ... {i}/{len(files)}  "
                f"decomp={counts['decompressed']} "
                f"copy={counts['copied_uncompressed']} "
                f"skip_ex={counts['skipped_existing']} "
                f"err={counts['error']}"
            )

    print("\nDone.")
    for k, v in counts.items():
        print(f"  {k}: {v}")
    if errors:
        log = out_root / "_decompress_errors.txt"
        log.write_text("\n".join(errors), encoding="utf-8")
        print(f"  errors logged -> {log}")
    return 0 if counts["error"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
