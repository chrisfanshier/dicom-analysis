"""Accept/export stitched JPEG / TIFF / DICOM MPR / NumPy + JSON sidecar."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pydicom
from PIL import Image
from pydicom.dataset import Dataset, FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, generate_uid

from core_stitch.hu_tiff import write_hu_tiff
from core_stitch.overlap import fuse_preview
from core_stitch.scale import auto_bounds, core_scale_block, dpi_for
from core_stitch.pipeline import (
    StitchState,
    build_stitched_mpr,
    core_crop_box,
    crop_to_box,
    fused_hu,
    preview_arrays,
    sidecar_dict,
    window_limits,
)


def _template_ds(state: StitchState) -> Dataset:
    paths = state.fixed.series.slice_paths
    if not paths:
        raise FileNotFoundError("No template DICOM paths on fixed series")
    mid = paths[len(paths) // 2]
    return pydicom.dcmread(mid, force=True)


def _slice_normal(ds: Dataset) -> np.ndarray:
    iop = getattr(ds, "ImageOrientationPatient", None)
    if iop is not None and len(iop) == 6:
        row = np.asarray(iop[:3], dtype=np.float64)
        col = np.asarray(iop[3:], dtype=np.float64)
        n = np.cross(row, col)
        norm = np.linalg.norm(n)
        if norm > 0:
            return n / norm
    return np.array([0.0, 0.0, 1.0], dtype=np.float64)


def write_mpr_dicom(
    state: StitchState,
    volume: np.ndarray,
    *,
    z_spacing_mm: float,
    out_dir: Path | str,
    series_description: str = "Coron STITCH",
) -> tuple[Path, str]:
    """
    Write fused HU volume as a new CT MPR series (one file per Z).

    Returns (series_dir, series_uid).
    """
    out_dir = Path(out_dir)
    series_dir = out_dir / "dicom_mpr"
    series_dir.mkdir(parents=True, exist_ok=True)

    template = _template_ds(state)
    series_uid = generate_uid()
    study_uid = str(
        getattr(template, "StudyInstanceUID", None) or generate_uid()
    )
    frame_uid = generate_uid()
    spacing = float(state.spacing_mm)
    normal = _slice_normal(template)
    iop = getattr(template, "ImageOrientationPatient", [1, 0, 0, 0, 1, 0])
    base_ipp = getattr(template, "ImagePositionPatient", None)
    if base_ipp is None:
        origin = np.zeros(3, dtype=np.float64)
    else:
        origin = np.asarray(base_ipp, dtype=np.float64)

    nz, rows, cols = volume.shape
    mid_z = (nz - 1) / 2.0
    lo, hi = window_limits(state)
    wc = 0.5 * (lo + hi)
    ww = max(hi - lo, 1.0)

    for z in range(nz):
        hu = volume[z]
        pixels = np.clip(np.rint(hu), -32768, 32767).astype(np.int16)

        file_meta = FileMetaDataset()
        file_meta.MediaStorageSOPClassUID = getattr(
            template, "SOPClassUID", "1.2.840.10008.5.1.4.1.1.2"
        )
        sop_uid = generate_uid()
        file_meta.MediaStorageSOPInstanceUID = sop_uid
        file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
        file_meta.ImplementationClassUID = generate_uid()

        ds = FileDataset(
            str(series_dir / f"IM{z:04d}.dcm"),
            {},
            file_meta=file_meta,
            preamble=b"\0" * 128,
        )

        for tag in (
            "PatientName",
            "PatientID",
            "PatientBirthDate",
            "PatientSex",
            "StudyDate",
            "StudyTime",
            "AccessionNumber",
            "StudyID",
            "ReferringPhysicianName",
        ):
            if hasattr(template, tag):
                setattr(ds, tag, getattr(template, tag))

        ds.StudyInstanceUID = study_uid
        ds.SeriesInstanceUID = series_uid
        ds.SOPInstanceUID = sop_uid
        ds.SOPClassUID = file_meta.MediaStorageSOPClassUID
        ds.Modality = "CT"
        ds.SeriesNumber = int(getattr(template, "SeriesNumber", 9000) or 9000) + 1
        ds.InstanceNumber = z + 1
        ds.SeriesDescription = series_description
        ds.ImageType = ["DERIVED", "SECONDARY", "MPR", "STITCH"]

        ds.SamplesPerPixel = 1
        ds.PhotometricInterpretation = "MONOCHROME2"
        ds.Rows = int(rows)
        ds.Columns = int(cols)
        ds.BitsAllocated = 16
        ds.BitsStored = 16
        ds.HighBit = 15
        ds.PixelRepresentation = 1
        ds.RescaleIntercept = 0
        ds.RescaleSlope = 1
        ds.PixelSpacing = [spacing, spacing]
        ds.SliceThickness = float(z_spacing_mm)
        ds.SpacingBetweenSlices = float(z_spacing_mm)
        ds.ImageOrientationPatient = [float(x) for x in iop]
        ipp = origin + (z - mid_z) * float(z_spacing_mm) * normal
        ds.ImagePositionPatient = [float(x) for x in ipp]
        ds.FrameOfReferenceUID = frame_uid
        ds.PositionReferenceIndicator = getattr(
            template, "PositionReferenceIndicator", ""
        )
        ds.WindowCenter = float(wc)
        ds.WindowWidth = float(ww)

        ds.PixelData = pixels.tobytes()
        ds.is_little_endian = True
        ds.is_implicit_VR = False
        ds.save_as(series_dir / f"IM{z:04d}.dcm", write_like_original=False)

    return series_dir, series_uid


def _crop_box(state: StitchState, pad_mm: float = 3.0) -> tuple[int, int, int, int]:
    assert state.mask_a is not None and state.mask_b is not None
    pad_px = max(2, int(round(pad_mm / max(state.spacing_mm, 1e-6))))
    return core_crop_box(state.mask_a, state.mask_b, pad_px=pad_px)


def export_accept(
    state: StitchState,
    out_dir: Path | str,
    *,
    stem: str = "stitch",
    jpeg: bool = True,
    tiff: bool = True,
    dicom: bool = True,
    npy: bool = True,
    json_sidecar: bool = True,
    pad_mm: float = 3.0,
    extra_meta: dict | None = None,
    core_bounds: tuple[float, float, str] | None = None,
    scale_out: dict | None = None,
) -> dict[str, Path | str]:
    """
    Export selected products. Greyscale JPEG/TIFF use the current HU window.

    TIFF is 8-bit LZW (lossless twin of the windowed JPEG) plus a 16-bit HU
    TIFF (`*_hu.tif`) when TIFF is enabled.
    """
    if not any(
        (
            jpeg,
            tiff,
            dicom,
            npy,
            json_sidecar,
        )
    ):
        raise ValueError("Select at least one export type")

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path | str] = {}

    meta = sidecar_dict(state)
    if extra_meta:
        meta.update(extra_meta)
    meta["exported_at"] = datetime.now(timezone.utc).isoformat()
    meta["export_view_z"] = state.view_z
    meta["export_jpeg_pad_mm"] = pad_mm
    lo, hi = window_limits(state)
    meta["export_window_lo"] = lo
    meta["export_window_hi"] = hi

    box = _crop_box(state, pad_mm=pad_mm)
    meta["export_crop_box_r0r1c0c1"] = [int(x) for x in box]

    need_gray = jpeg or tiff
    gray: np.ndarray | None = None
    if need_gray:
        gray = preview_arrays(
            state,
            align_mode=False,
            opacity_a=1.0,
            opacity_b=1.0,
            crop_core=True,
            pad_mm=pad_mm,
        )
        meta["export_shape_px"] = [int(gray.shape[0]), int(gray.shape[1])]
        meta["export_jpeg_cropped"] = True
        if core_bounds is None:
            t_cm, b_cm = auto_bounds(state.mask_a | state.mask_b, box, state.spacing_mm)  # type: ignore[operator]
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

    if jpeg and gray is not None:
        jpg_path = out_dir / f"{stem}.jpg"
        Image.fromarray(gray).save(
            jpg_path, format="JPEG", quality=95, optimize=True,
            dpi=(dpi_for(state.spacing_mm),) * 2,
        )
        meta["export_jpeg"] = jpg_path.name
        paths["jpeg"] = jpg_path

    if tiff and gray is not None:
        tif_path = out_dir / f"{stem}.tif"
        Image.fromarray(gray).save(
            tif_path, format="TIFF", compression="tiff_lzw",
            dpi=(dpi_for(state.spacing_mm),) * 2,
        )
        meta["export_tiff"] = tif_path.name
        paths["tiff"] = tif_path

        # signed 16-bit TIFF: the pixel value is the HU (no offset)
        hu = crop_to_box(fused_hu(state), box)
        tif_hu = out_dir / f"{stem}_hu.tif"
        write_hu_tiff(tif_hu, hu, dpi_for(state.spacing_mm))
        meta["export_tiff_hu"] = tif_hu.name
        meta["export_tiff_hu_offset"] = 0.0
        meta["export_tiff_hu_spacing_mm"] = state.spacing_mm
        paths["tiff_hu"] = tif_hu

    if npy and state.canvas_a is not None and state.canvas_b is not None:
        fused2d = fuse_preview(
            state.canvas_a,
            state.canvas_b,
            state.mask_a,  # type: ignore[arg-type]
            state.mask_b,  # type: ignore[arg-type]
            opacity_a=1.0,
            opacity_b=1.0,
            mode=state.fuse_mode,
            feather_mm=max(4.0, 12.0 * state.spacing_mm),
            spacing_mm=state.spacing_mm,
            match_hu=state.match_hu,
        )
        fused2d = crop_to_box(fused2d, box)
        npy_path = out_dir / f"{stem}_fused.npy"
        np.save(npy_path, fused2d.astype(np.float32))
        meta["export_npy"] = npy_path.name
        paths["npy"] = npy_path

    volume = None
    z_mm = None
    if dicom or npy:
        volume, z_mm = build_stitched_mpr(state)

    if dicom and volume is not None and z_mm is not None:
        dicom_dir, series_uid = write_mpr_dicom(
            state,
            volume,
            z_spacing_mm=z_mm,
            out_dir=out_dir,
        )
        meta["export_dicom_dir"] = dicom_dir.name
        meta["export_dicom_series_uid"] = series_uid
        meta["export_mpr_shape_zyx"] = [int(x) for x in volume.shape]
        meta["export_mpr_z_spacing_mm"] = float(z_mm)
        paths["dicom_dir"] = dicom_dir

    if npy and volume is not None and z_mm is not None:
        vol_npy = out_dir / f"{stem}_mpr.npy"
        np.save(vol_npy, volume.astype(np.float32))
        meta["export_mpr_npy"] = vol_npy.name
        meta["export_mpr_shape_zyx"] = [int(x) for x in volume.shape]
        meta["export_mpr_z_spacing_mm"] = float(z_mm)
        paths["mpr_npy"] = vol_npy

    if json_sidecar:
        json_path = out_dir / f"{stem}.json"
        json_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        paths["json"] = json_path

    return paths
