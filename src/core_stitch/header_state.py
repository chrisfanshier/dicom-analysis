"""GUI state and export for the header-geometry (SimpleITK) stitch.

``HeaderState`` answers the same questions the GUI asks of a single scan (frame, window,
crop box, greyscale preview, core mask), so the preview, window, guide and slice controls work
unchanged.  It renders one slice at a time for the live preview and the whole volume only when
you export.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import dataclasses

import numpy as np
import pydicom
from scipy.ndimage import binary_erosion
import SimpleITK as sitk
from PIL import Image
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, generate_uid

from core_stitch.geometry import core_geometry
from core_stitch.header_check import header_check
from core_stitch.hu_tiff import write_hu_tiff
from core_stitch.overlap import alignment_rgb
from core_stitch.itk_stitch import (
    Half,
    ItkStitch,
    RefineOutcome,
    image_based_move_mm,
    join_quality as _join_quality,
    load_half,
    plan_stitch,
    refine_translation,
    render_plan,
)
from core_stitch.pipeline import core_crop_box, crop_to_box
from core_stitch.scale import auto_bounds, core_scale_block, dpi_for

CROP_PAD_MM = 10.0  # keep this much beyond the core's width in the working volume
TRIM_ROWS = 3
SIDECAR_VERSION = "0.2.0"


class HeaderState:
    kind = "header"

    def __init__(self, top: Half, bottom: Half, *, placement: str = "whole_pixel") -> None:
        self.top, self.bottom = top, bottom
        self.placement = placement
        self.move_mm = [0.0, 0.0, 0.0]  # bottom half, from its header position: along, across, depth
        self.moved_by = "scanner header"  # or "image alignment" after Re-align
        self.refine_info: dict | None = None  # set by refine(); cleared when the shift is changed by hand
        self._undo: tuple | None = None  # (move_mm, moved_by, refine_info) from before the last refine
        self.last_quality: dict | None = None
        self.window_lo: float | None = None
        self.window_hi: float | None = None
        self.warnings: list[str] = []
        self.swapped = False  # True when the catalog's TOP/BOTTOM labels were exchanged on load
        self.top_turned = False  # the half in the TOP role was turned 180 degrees in place (scanned upside down)
        self.bottom_turned = False
        self.straighten = False  # rotate the finished stitch so the core is exactly vertical (at export)
        self.upside_down = False  # turn the whole picture 180 degrees (the core's top is at the bottom); lossless
        self.check = header_check(top, bottom)
        self._replan()
        self.frame = self.plan.n_slices // 2
        self._render()
        self.auto_window()

    # ----- construction -----
    @classmethod
    def load(cls, top_dir, top_uid, bottom_dir, bottom_uid, placement: str = "whole_pixel", *, swapped: bool = False,
             top_turned: bool = False, bottom_turned: bool = False) -> "HeaderState":
        """``swapped=True``: the two series are labelled backwards, so the one given as TOP is treated as the bottom half.

        ``top_turned`` / ``bottom_turned`` (for the series given as TOP / BOTTOM): that series was scanned upside down
        relative to the other, so its slices are turned 180 degrees in place (lossless).  The header position then no
        longer describes its content: place it with ``realign_from_image``.
        """
        specs = [(top_dir, top_uid, bool(top_turned)), (bottom_dir, bottom_uid, bool(bottom_turned))]
        if swapped:
            specs.reverse()
        (td, tu, tt), (bd, bu, bt) = specs
        st = cls(load_half("top", td, tu, turn180=tt), load_half("bottom", bd, bu, turn180=bt), placement=placement)
        st.swapped = bool(swapped)
        st.top_turned, st.bottom_turned = tt, bt
        return st

    # ----- plan / render -----
    def _replan(self) -> None:
        self.plan = plan_stitch(self.top, self.bottom, move_mm=tuple(self.move_mm), placement=self.placement, crop_pad_mm=CROP_PAD_MM)

    def _render(self) -> None:
        res = render_plan(self.plan, trim_rows=TRIM_ROWS, z_keep=(self.frame, self.frame))
        self.res: ItkStitch = _rotate180_full(res) if self.upside_down else res

    def set_upside_down(self, on: bool) -> None:
        """Turn the whole picture 180 degrees (rows and columns reversed; no pixel value changes)."""
        on = bool(on)
        if on != self.upside_down:
            self.upside_down = on
            self._render()

    def set_move(self, along: float, across: float, depth: float, *, source: str | None = None, keep_refine: bool = False) -> None:
        self.move_mm = [float(along), float(across), float(depth)]
        if source:
            self.moved_by = source
        if not keep_refine:
            self.refine_info = None
            self._undo = None  # a hand shift makes the old refine undo meaningless
        self._replan()
        self.frame = int(np.clip(self.frame, 0, self.plan.n_slices - 1))
        self._render()

    def set_placement(self, placement: str) -> None:
        if placement != self.placement:
            self.placement = placement
            self.set_move(*self.move_mm)

    def join_correlation(self) -> float:
        """Correlation of the two halves, as placed, over the overlap core of the current slice."""
        r = self.res
        m = r.ref_core[0] & r.mov_core[0] & r.ref_footprint[0] & r.mov_footprint[0]
        m = binary_erosion(m, np.ones((9, 1), bool))  # stay off the halves' edge rows
        if m.sum() < 200:
            return float("nan")
        return float(np.corrcoef(r.ref_vol[0][m].astype(np.float64), r.mov_vol[0][m].astype(np.float64))[0, 1])

    def refine(self) -> RefineOutcome:
        """Fine-tune the position by SimpleITK registration (translation along and across the core).

        The registration finds an exact (sub-pixel) optimum.  It is applied from the position the
        halves are *really* at: any whole-pixel rounding already in the placement is folded into the
        shift first, so the optimum is rounded once, not twice.
        """
        out = refine_translation(self.plan)
        if not out.applied:
            return out
        before_move, before_by, before_info = list(self.move_mm), self.moved_by, self.refine_info
        before_corr = self.join_correlation()
        p = self.plan
        sign = 1.0 if p.mov is p.bottom else -1.0
        v = p.snap_vec
        fold = [sign * float(np.dot(v, p.ref.col_dir)), sign * float(np.dot(v, p.ref.row_dir)), sign * float(np.dot(v, p.ref.normal))]
        a, b, d = (self.move_mm[i] + fold[i] for i in range(3))
        placed_before = p.move.copy()
        self.set_move(a + out.along_mm, b + out.across_mm, d, source=None, keep_refine=True)
        if self.plan.snapped and np.allclose(self.plan.move, placed_before, atol=1e-6):
            # the refined optimum rounds to the pixel the halves are already on
            self.set_move(*before_move, source=before_by, keep_refine=True)
            self.refine_info = before_info
            return RefineOutcome(0.0, 0.0, before_corr, before_corr, False,
                                 f"Refine's best position ({out.along_mm:+.2f} mm along, {out.across_mm:+.2f} mm across) rounds to the pixel the "
                                 "halves are already on, so nothing changed. Untick whole-pixel placement to use the sub-pixel position.",
                                 out.iterations)
        after_corr = self.join_correlation()
        if np.isfinite(before_corr) and np.isfinite(after_corr) and after_corr < before_corr - 0.002:
            # rounding to a whole pixel cannot honour this optimum: keep what was there
            self.set_move(*before_move, source=before_by, keep_refine=True)
            self.refine_info = before_info
            return RefineOutcome(0.0, 0.0, before_corr, after_corr, False,
                                 "Refine's best position is a fraction of a pixel and whole-pixel placement can't hold it "
                                 f"(it would lower the join correlation from {before_corr:.4f} to {after_corr:.4f}), so nothing "
                                 "was changed. Untick whole-pixel placement to use the sub-pixel position.", out.iterations)
        self._undo = (before_move, before_by, before_info)
        if not self.moved_by.endswith("+ refine"):
            self.moved_by += " + refine"
        self.refine_info = {
            "method": "SimpleITK translation registration, correlation metric, linear interpolation, 2 levels",
            "shift_mm": {"along_core": out.along_mm, "across_core": out.across_mm},
            "optimum_correlation_before": None if not np.isfinite(out.score_before) else round(out.score_before, 4),
            "optimum_correlation_after": round(out.score_after, 4),
            "join_correlation_as_placed_before": None if not np.isfinite(before_corr) else round(before_corr, 4),
            "join_correlation_as_placed_after": None if not np.isfinite(after_corr) else round(after_corr, 4),
            "iterations": out.iterations,
        }
        return RefineOutcome(out.along_mm, out.across_mm, before_corr, after_corr, True, "", out.iterations)

    def join_quality(self) -> dict:
        """Correlation over the overlap at the current placement, plus a scan of nearby positions.

        Renders just the overlap band (all slices), so it takes a second or two.
        """
        ov = self.res.info["overlap_rows"]
        if ov is None:
            q = {"evaluated": False, "reason": "the halves do not overlap"}
        else:
            a, b = max(ov[0] - 16, 0), min(ov[1] + 16, self.plan.size[1] - 1)
            band = render_plan(self.plan, trim_rows=TRIM_ROWS, y_keep=(a, b))
            q = _join_quality(band)
        self.last_quality = q
        return q

    def cross_section(self, frac: float = 0.5) -> tuple[np.ndarray, dict]:
        """Alignment colours across the core (depth x width) at a row inside the overlap.

        If the join is right the two halves' cross-sections coincide; a depth error shows them displaced
        and a tilt shows one lopsided.  Returns (RGB, info) with the share of the two outlines that overlap.
        """
        ov = self.res.info["overlap_rows"]
        if ov is None:
            raise ValueError("the halves do not overlap, so there is no cross-section to compare")
        row = int(round(ov[0] + float(np.clip(frac, 0.0, 1.0)) * (ov[1] - ov[0])))
        r = render_plan(self.plan, trim_rows=0, y_keep=(row, row), threshold_masks=True)
        top_is_ref = r.reference.name == "top"
        tv, tm = (r.ref_vol[:, 0, :], r.ref_core[:, 0, :]) if top_is_ref else (r.mov_vol[:, 0, :], r.mov_core[:, 0, :])
        bv, bm = (r.mov_vol[:, 0, :], r.mov_core[:, 0, :]) if top_is_ref else (r.ref_vol[:, 0, :], r.ref_core[:, 0, :])
        lo, hi = self.window_limits()
        rgb = alignment_rgb(tv, bv, tm, bm, lo=lo, hi=hi, opacity_a=1.0, opacity_b=1.0)
        cols = np.where((tm | bm).any(axis=0))[0]
        if cols.size:
            rgb = rgb[:, max(int(cols.min()) - 15, 0) : int(cols.max()) + 16]
        union = int((tm | bm).sum())
        spacing = r.grid.GetSpacing()
        info = {
            "row": row,
            "outline_overlap": float((tm & bm).sum() / union) if union else float("nan"),
            "slice_step_over_pixel": float(spacing[2] / spacing[0]),
            "pixel_mm": float(spacing[0]),
        }
        return rgb, info

    def can_undo_refine(self) -> bool:
        return self._undo is not None

    def undo_refine(self) -> bool:
        """Put the position back to what it was before the last Refine join."""
        if self._undo is None:
            return False
        move, by, info = self._undo
        self.set_move(*move, source=by, keep_refine=True)
        self.refine_info = info
        self._undo = None
        return True

    def refine_note(self) -> str:
        """What a refined shift will cost in resampling, for this pair, in plain words."""
        same = self.plan.same_lattice
        if not same:
            mov = self.plan.mov.name
            return (f"The {mov} half is already resampled once to the finer pixel size. A refined shift is folded into "
                    "that same step, so it adds no extra resampling or smoothing.")
        if self.placement == "whole_pixel":
            return ("Both halves share a pixel size and whole-pixel placement is on: a refined shift is rounded to a "
                    "whole pixel, so no voxels are resampled (placement good to half a pixel).")
        return ("Both halves share a pixel size and whole-pixel placement is off: a sub-pixel refined shift resamples "
                "the bottom half once (linear), which smooths it slightly.")

    def realign_from_image(self, require_below: bool = False) -> tuple[float, float, float]:
        """Fallback: ask the mask + profile alignment where the bottom half goes (optionally forcing it below the top half)."""
        mv = image_based_move_mm(
            self.top.series.path, self.top.series.series_uid, self.bottom.series.path, self.bottom.series.series_uid,
            require_below=require_below, turn_top=self.top_turned, turn_bottom=self.bottom_turned,
        )
        self.set_move(*mv, source="image alignment (mask + profile)")
        return mv

    # ----- the protocol the GUI uses for single-image modes -----
    @property
    def n_frames(self) -> int:
        return self.plan.n_slices

    @property
    def spacing_mm(self) -> float:
        return float(self.plan.ref.image.GetSpacing()[0])

    def set_frame(self, frame: int) -> None:
        f = int(np.clip(frame, 0, self.n_frames - 1))
        if f != self.frame:
            self.frame = f
            self._render()

    def core_mask(self) -> np.ndarray:
        return self.res.ref_core[0] | self.res.mov_core[0]

    def hu_values(self) -> np.ndarray:
        m = self.core_mask()
        v = self.res.volume[0][m]
        return v if v.size else self.res.volume[0].ravel()

    def auto_window(self) -> tuple[float, float]:
        v = self.hu_values()
        lo, hi = float(np.percentile(v, 1)), float(np.percentile(v, 99))
        self.window_lo, self.window_hi = lo, hi
        return lo, hi

    def window_limits(self) -> tuple[float, float]:
        if self.window_lo is None or self.window_hi is None:
            return self.auto_window()
        lo, hi = float(self.window_lo), float(self.window_hi)
        return (lo, lo + 1.0) if hi <= lo else (lo, hi)

    def crop_box(self, pad_mm: float = 3.0) -> tuple[int, int, int, int]:
        pad_px = max(2, int(round(pad_mm / max(self.spacing_mm, 1e-6))))
        m = self.core_mask()
        return core_crop_box(m, m, pad_px=pad_px)

    def gray_preview(self, pad_mm: float = 3.0, crop: bool = True) -> np.ndarray:
        lo, hi = self.window_limits()
        img = self.res.volume[0]
        if crop:
            img = crop_to_box(img, self.crop_box(pad_mm))
        return np.clip((img - lo) / (hi - lo) * 255.0, 0, 255).astype(np.uint8)

    def align_preview(self, opacity_a: float = 0.75, opacity_b: float = 0.75, pad_mm: float = 3.0, crop: bool = True) -> np.ndarray:
        """Alignment colours: TOP magenta, BOTTOM green, each as resampled (not fused), on the shared grid.

        Where the halves overlap and agree the colours add to grey/white; a misplaced half shows
        as magenta/green fringes or doubled features.
        """
        r = self.res
        top_is_ref = r.reference.name == "top"
        top_vol, top_core = (r.ref_vol[0], r.ref_core[0]) if top_is_ref else (r.mov_vol[0], r.mov_core[0])
        btm_vol, btm_core = (r.mov_vol[0], r.mov_core[0]) if top_is_ref else (r.ref_vol[0], r.ref_core[0])
        lo, hi = self.window_limits()
        rgb = alignment_rgb(top_vol, btm_vol, top_core, btm_core, lo=lo, hi=hi, opacity_a=opacity_a, opacity_b=opacity_b)
        if crop:
            r0, r1, c0, c1 = self.crop_box(pad_mm)
            rgb = rgb[r0:r1, c0:c1]
        return rgb

    # ----- text for the GUI -----
    def measured_tilt_deg(self) -> float:
        """Lean of the core's long axis from straight down the rows on the current slice, in degrees."""
        return _measured_tilt(self.res, 0, self.spacing_mm)

    def summary(self) -> str:
        i = self.res.info
        up = i["upsample_factor_moving"]
        lines = [
            ("SWAPPED. " if self.swapped else "")
            + f"TOP: {self.top.series.series_description.strip() or self.top.series.series_uid[-12:]}"
            + (" (turned 180 deg)" if self.top_turned else "") + "  |  "
            + f"BOTTOM: {self.bottom.series.series_description.strip() or self.bottom.series.series_uid[-12:]}"
            + (" (turned 180 deg)" if self.bottom_turned else ""),
            f"Reference: {i['reference']} half ({i['ref_pixel_mm']:.3f} mm/px), kept as scanned.",
            (f"{i['moving'].capitalize()} half resampled once from {i['moving_pixel_mm']:.3f} to {i['ref_pixel_mm']:.3f} mm/px (x{up:.2f})."
             if abs(up - 1.0) > 1e-3 else
             f"Both halves are {i['ref_pixel_mm']:.3f} mm/px; "
             + ("the shift is snapped to a whole pixel (all voxels original)." if i["snapped_to_pixel"] else "placed at sub-pixel precision.")),
            f"Overlap {i['overlap_mm']:.0f} mm. Position: {self.moved_by}"
            + ("" if not any(abs(x) > 1e-9 for x in self.move_mm) else
               f" + shift along {self.move_mm[0]:+.1f} / across {self.move_mm[1]:+.1f} / depth {self.move_mm[2]:+.1f} mm"),
        ]
        c = self.join_correlation()
        if c == c:  # not NaN
            lines.append(f"Join correlation on this slice: {c:.4f} (higher = the halves agree better in the overlap).")
        try:
            t = self.measured_tilt_deg()
            lines.append(
                f"Core tilt {t:+.2f} deg: " + ("will be straightened when you export." if self.straighten
                                              else "not straightened (tick 'Straighten the core' to rotate it upright on export).")
            )
        except Exception:
            pass
        return "\n".join(lines)

    def check_text(self) -> tuple[str, bool]:
        c = self.check
        if c["verdict"] == "headers_good":
            ncc = c["ncc_at_header"]
            return ("Scanner placement looks consistent" + (f" (profile correlation {ncc:.2f})." if ncc is not None else ".")
                    + "".join(" " + n for n in c.get("notes", [])), True)
        return ("Scanner placement is doubtful: " + "; ".join(c["reasons"]) + ". Adjust the shift or use Re-align from image."
                + "".join(" " + n for n in c.get("notes", [])), False)


# ---------------------------------------------------------------------------- export


def _air_hu(full: ItkStitch) -> float:
    return float(full.reference.air)


def write_header_dicom(full: ItkStitch, state: HeaderState, out_dir: Path, *, series_description: str = "Coron STITCH") -> tuple[Path, str]:
    """Write the fused volume as a CT MPR stack in the scanner's own patient coordinates."""
    series_dir = Path(out_dir) / "dicom_mpr"
    series_dir.mkdir(parents=True, exist_ok=True)
    ref = full.reference
    tmpl = pydicom.dcmread(ref.series.slice_paths[len(ref.series.slice_paths) // 2], stop_before_pixels=True, force=True)
    grid = full.grid
    spacing = grid.GetSpacing()
    origin = np.asarray(grid.GetOrigin())
    dm = np.asarray(grid.GetDirection(), float).reshape(3, 3)  # columns: row direction, column direction, normal
    row_dir, col_dir, normal = dm[:, 0], dm[:, 1], dm[:, 2]  # (rotated from the scan's if the stitch was straightened)
    series_uid = generate_uid()
    study_uid = str(getattr(tmpl, "StudyInstanceUID", None) or generate_uid())
    frame_uid = str(getattr(tmpl, "FrameOfReferenceUID", None) or generate_uid())  # same space as the scans
    lo, hi = state.window_limits()
    vol = np.clip(np.rint(full.volume), -32768, 32767).astype(np.int16)
    nz, rows, cols = vol.shape
    iop = [float(x) for x in np.r_[row_dir, col_dir]]
    for k in range(nz):
        meta = FileMetaDataset()
        meta.MediaStorageSOPClassUID = getattr(tmpl, "SOPClassUID", "1.2.840.10008.5.1.4.1.1.2")
        sop = generate_uid()
        meta.MediaStorageSOPInstanceUID = sop
        meta.TransferSyntaxUID = ExplicitVRLittleEndian
        meta.ImplementationClassUID = generate_uid()
        path = series_dir / f"IM{k:04d}.dcm"
        ds = FileDataset(str(path), {}, file_meta=meta, preamble=b"\0" * 128)
        for tag in ("PatientName", "PatientID", "PatientBirthDate", "PatientSex", "StudyDate", "StudyTime",
                    "AccessionNumber", "StudyID", "ReferringPhysicianName", "PatientPosition", "PatientOrientation"):
            if hasattr(tmpl, tag):
                setattr(ds, tag, getattr(tmpl, tag))
        ds.StudyInstanceUID, ds.SeriesInstanceUID, ds.SOPInstanceUID = study_uid, series_uid, sop
        ds.SOPClassUID = meta.MediaStorageSOPClassUID
        ds.Modality = "CT"
        ds.SeriesNumber = int(getattr(tmpl, "SeriesNumber", 9000) or 9000) + 1
        ds.InstanceNumber = k + 1
        ds.SeriesDescription = series_description
        ds.ImageType = ["DERIVED", "SECONDARY", "MPR", "STITCH"]
        ds.SamplesPerPixel = 1
        ds.PhotometricInterpretation = "MONOCHROME2"
        ds.Rows, ds.Columns = int(rows), int(cols)
        ds.BitsAllocated = ds.BitsStored = 16
        ds.HighBit = 15
        ds.PixelRepresentation = 1
        ds.RescaleIntercept, ds.RescaleSlope = 0, 1
        ds.PixelSpacing = [float(spacing[1]), float(spacing[0])]  # row pitch, column pitch
        ds.SliceThickness = ds.SpacingBetweenSlices = float(spacing[2])
        ds.ImageOrientationPatient = iop
        ds.ImagePositionPatient = [float(x) for x in origin + k * float(spacing[2]) * normal]
        ds.FrameOfReferenceUID = frame_uid
        ds.PositionReferenceIndicator = getattr(tmpl, "PositionReferenceIndicator", "")
        ds.WindowCenter, ds.WindowWidth = float(0.5 * (lo + hi)), float(max(hi - lo, 1.0))
        ds.PixelData = vol[k].tobytes()
        ds.is_little_endian, ds.is_implicit_VR = True, False
        ds.save_as(path, write_like_original=False)
    return series_dir, series_uid


def _measured_tilt(full: ItkStitch, frame: int, mm: float) -> float:
    """Lean of the stitched core's long axis from the row direction, in degrees (+ = leans to larger columns going down)."""
    mask = full.ref_core[frame] | full.mov_core[frame]
    return float(core_geometry(mask, mm).angle_deg)


def _straighten_full(full: ItkStitch, tilt_deg: float) -> ItkStitch:
    """Rotate the stitched volume in its slice plane so the core runs exactly down the rows.

    The output grid is the same grid turned about the through-stack axis, so the DICOM geometry stays exact
    (only the in-plane direction vectors change).  Volume: linear interpolation; masks: nearest neighbour.
    This is a second resampling of the moving half (the first put it on the reference grid).
    """
    g = full.grid
    d = np.asarray(g.GetDirection(), float).reshape(3, 3)
    e_i, e_j, n = d[:, 0], d[:, 1], d[:, 2]  # column-index direction, row-index direction (down the core), normal
    sp = np.asarray(g.GetSpacing(), float)
    th = np.radians(tilt_deg)
    axis = np.cos(th) * sp[1] * e_j + np.sin(th) * sp[0] * e_i  # the core axis in mm
    ej2 = axis / np.linalg.norm(axis)
    ei2 = np.cross(ej2, n)
    d2 = np.column_stack([ei2, ej2, n])
    size = np.asarray(g.GetSize(), int)
    corners = np.asarray([g.TransformIndexToPhysicalPoint((int(i), int(j), int(k)))
                          for i in (0, size[0] - 1) for j in (0, size[1] - 1) for k in (0, size[2] - 1)])
    o = np.asarray(g.GetOrigin(), float)
    u = (corners - o) @ d2  # each corner along the new axes, mm
    lo, hi = u.min(axis=0), u.max(axis=0)
    new_size = [int(np.ceil((hi[a] - lo[a]) / sp[a] - 1e-6)) + 1 for a in range(3)]
    ng = sitk.Image(new_size, sitk.sitkFloat32)
    ng.SetSpacing(tuple(float(x) for x in sp))
    ng.SetDirection(tuple(float(x) for x in d2.flatten()))
    ng.SetOrigin(tuple(float(x) for x in o + d2 @ lo))
    air = float(full.reference.air)

    def put(arr: np.ndarray, *, linear: bool) -> np.ndarray:
        img = sitk.GetImageFromArray(np.ascontiguousarray(arr, dtype=np.float32 if linear else np.uint8))
        img.CopyInformation(g)
        out = sitk.Resample(img, ng, sitk.Transform(), sitk.sitkLinear if linear else sitk.sitkNearestNeighbor,
                            air if linear else 0, sitk.sitkFloat32 if linear else sitk.sitkUInt8)
        return sitk.GetArrayFromImage(out)

    empty = np.empty((0, 0, 0), np.float32)  # the per-half renderings are not rotated; do not leave stale copies
    info = dict(full.info, straightened_by_deg=float(tilt_deg))
    return dataclasses.replace(
        full, volume=put(full.volume, linear=True), grid=ng, ref_vol=empty, mov_vol=empty,
        ref_footprint=put(full.ref_footprint, linear=False).astype(bool), mov_footprint=put(full.mov_footprint, linear=False).astype(bool),
        ref_core=put(full.ref_core, linear=False).astype(bool), mov_core=put(full.mov_core, linear=False).astype(bool), info=info,
    )


def _rotate180_full(full: ItkStitch) -> ItkStitch:
    """Turn every slice 180 degrees in its own plane: rows and columns reversed.  Lossless.

    The grid's row and column directions are negated and its origin moves to the opposite corner, so every
    voxel keeps its true position in space (a DICOM stack written from it describes the same volume).
    """
    g = full.grid
    d = np.asarray(g.GetDirection(), float).reshape(3, 3)
    sp = np.asarray(g.GetSpacing(), float)
    size = g.GetSize()  # (columns, rows, slices)
    origin = np.asarray(g.GetOrigin(), float) + (size[0] - 1) * sp[0] * d[:, 0] + (size[1] - 1) * sp[1] * d[:, 1]
    d2 = d.copy()
    d2[:, 0] *= -1.0
    d2[:, 1] *= -1.0
    ng = sitk.Image(list(size), sitk.sitkFloat32)
    ng.SetSpacing(tuple(float(x) for x in sp))
    ng.SetDirection(tuple(float(x) for x in d2.flatten()))
    ng.SetOrigin(tuple(float(x) for x in origin))

    def rot(a: np.ndarray) -> np.ndarray:
        return np.ascontiguousarray(a[:, ::-1, ::-1])

    return dataclasses.replace(
        full, volume=rot(full.volume), grid=ng, ref_vol=rot(full.ref_vol), mov_vol=rot(full.mov_vol),
        ref_footprint=rot(full.ref_footprint), mov_footprint=rot(full.mov_footprint),
        ref_core=rot(full.ref_core), mov_core=rot(full.mov_core), info=dict(full.info, rotated_180=True),
    )


def _rows_where(mask2d: np.ndarray) -> list[int] | None:
    r = np.where(mask2d.any(axis=1))[0]
    return None if r.size == 0 else [int(r[0]), int(r[-1])]


def sidecar_header(state: HeaderState, full: ItkStitch, frame: int, box) -> dict:
    """The stitch description in plain terms: where each half came from and where the join is."""
    i = full.info
    ref, mov = full.reference, full.moving
    r0 = int(box[0])
    fp_top = (full.ref_footprint if ref.name == "top" else full.mov_footprint)[frame]
    fp_btm = (full.ref_footprint if ref.name == "bottom" else full.mov_footprint)[frame]
    px = state.spacing_mm
    height = int(box[1]) - int(box[0])

    def shift(rng):
        """Rows in the exported image: clipped to it, None if the half is not in the picture."""
        if rng is None:
            return None
        a, b = max(rng[0] - r0, 0), min(rng[1] - r0, height - 1)
        return None if b < a else [int(a), int(b)]

    ov = shift(_rows_where(fp_top & fp_btm))
    c = state.check
    return {
        "version": SIDECAR_VERSION,
        "engine": "simpleitk_header_geometry",
        "software": {"simpleitk": i["sitk_version"]},
        "stitched": True,
        "top_bottom_swapped": bool(state.swapped),
        "top": {"series_uid": state.top.series.series_uid, "series_description": state.top.series.series_description,
                "native_mm_per_px": float(state.top.spacing_xyz[0]), "slice_step_mm": float(state.top.spacing_xyz[2]),
                "turned_180": bool(state.top_turned)},
        "bottom": {"series_uid": state.bottom.series.series_uid, "series_description": state.bottom.series.series_description,
                   "native_mm_per_px": float(state.bottom.spacing_xyz[0]), "slice_step_mm": float(state.bottom.spacing_xyz[2]),
                   "turned_180": bool(state.bottom_turned)},
        "resolution": {
            "output_mm_per_px": px,
            "reference_half": i["reference"],
            "interpolated_half": None if abs(i["upsample_factor_moving"] - 1.0) < 1e-3 else i["moving"],
            "upsample_factor_of_interpolated_half": None if abs(i["upsample_factor_moving"] - 1.0) < 1e-3 else float(i["upsample_factor_moving"]),
            "interpolation": i["interpolation"],
            "note": "The reference half keeps its scanned voxels. Interpolated pixels add no detail.",
        },
        "placement": {
            "source": state.moved_by,
            "bottom_shift_mm": {"along_core": state.move_mm[0], "across_core": state.move_mm[1], "through_stack": state.move_mm[2]},
            "pixel_placement": state.placement,
            "snapped_to_whole_pixel": bool(i["snapped_to_pixel"]),
            "snap_residual_mm": float(i["snap_residual_mm"]),
            "refine": state.refine_info,
        },
        "header_check": {
            "verdict": c["verdict"], "reasons": list(c["reasons"]),
            "across_diff_mm": c["across_diff_mm"], "depth_diff_mm": c["depth_diff_mm"],
            "profile_correlation_at_header": c["ncc_at_header"],
            "orientation_difference_deg": c.get("orientation_diff_deg") or 0.0,
            "notes": list(c.get("notes", [])),
        },
        "join": {
            "overlap_rows_in_image": ov,
            "overlap_cm": None if ov is None else round((ov[1] - ov[0] + 1) * px / 10.0, 2),
            "top_rows_in_image": shift(_rows_where(fp_top)),
            "bottom_rows_in_image": shift(_rows_where(fp_btm)),
            "fuse": "mean in the overlap; within %d rows of either half's image edge only the other half is used" % i["trim_rows"],
            "outside_core_fill_hu": _air_hu(full),
        },
        "warnings": list(state.warnings),
    }


def export_header(
    state: HeaderState,
    out_dir: Path | str,
    *,
    stem: str = "stitch",
    jpeg: bool = True,
    tiff: bool = True,
    dicom: bool = True,
    json_sidecar: bool = True,
    pad_mm: float = 3.0,
    extra_meta: dict | None = None,
    core_bounds: tuple[float, float, str] | None = None,
    scale_out: dict | None = None,
) -> dict[str, Path]:
    """Render the whole volume once and write the selected products."""
    if not any((jpeg, tiff, dicom, json_sidecar)):
        raise ValueError("Select at least one export type")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    full = render_plan(state.plan, trim_rows=TRIM_ROWS)  # every slice
    frame = int(np.clip(state.frame, 0, full.volume.shape[0] - 1))
    quality = _join_quality(full)  # of the stitch as rendered, before any rotation
    tilt = _measured_tilt(full, frame, state.spacing_mm)
    rotated = bool(state.straighten) and abs(tilt) > 1e-6
    if rotated:
        full = _straighten_full(full, tilt)
    if state.upside_down:
        full = _rotate180_full(full)
    mask = full.ref_core[frame] | full.mov_core[frame]
    pad_px = max(2, int(round(pad_mm / max(state.spacing_mm, 1e-6))))
    box = core_crop_box(mask, mask, pad_px=pad_px)
    lo, hi = state.window_limits()
    img = crop_to_box(full.volume[frame], box)
    gray = np.clip((img - lo) / (hi - lo) * 255.0, 0, 255).astype(np.uint8)
    if core_bounds is None:
        t_cm, b_cm = auto_bounds(mask, box, state.spacing_mm)
        core_bounds = (t_cm, b_cm, "auto")

    meta = sidecar_header(state, full, frame, box)
    if extra_meta:
        meta.update(extra_meta)
    meta["exported_at"] = datetime.now(timezone.utc).isoformat()
    meta["view_slice"] = frame
    meta["n_slices"] = int(full.volume.shape[0])
    meta["window_lo"], meta["window_hi"] = float(lo), float(hi)
    meta["join_quality"] = quality
    meta["rotate_180"] = {
        "applied": bool(state.upside_down),
        "note": "The whole picture was turned 180 degrees (rows and columns reversed; no pixel value changed) because the "
                "core was upside down. The DICOM orientation vectors follow, so every voxel keeps its true position."
                if state.upside_down else None,
    }
    meta["straighten"] = {
        "applied": bool(rotated),
        "measured_tilt_deg": round(tilt, 4),
        "rotation_applied_deg": round(-tilt, 4) if rotated else 0.0,
        "interpolation": ("linear: the whole stitched volume was resampled once more, so the reference half's voxels "
                          "are no longer exactly as scanned") if rotated else None,
    }
    meta["export_jpeg_pad_mm"] = pad_mm
    meta["export_crop_box_r0r1c0c1"] = [int(x) for x in box]
    meta["export_shape_px"] = [int(gray.shape[0]), int(gray.shape[1])]
    meta["core_scale"] = core_scale_block(
        gray.shape[:2], state.spacing_mm, core_bounds[0], core_bounds[1], source=core_bounds[2],
        expected_length_cm=(extra_meta or {}).get("catalog_section_length_cm"),
    )
    if scale_out is not None:
        scale_out.update(meta["core_scale"])
    dpi = (dpi_for(state.spacing_mm),) * 2

    paths: dict[str, Path] = {}
    if jpeg:
        p = out_dir / f"{stem}.jpg"
        Image.fromarray(gray).save(p, format="JPEG", quality=95, optimize=True, dpi=dpi)
        meta["export_jpeg"] = p.name
        paths["jpeg"] = p
    if tiff:
        p = out_dir / f"{stem}.tif"
        Image.fromarray(gray).save(p, format="TIFF", compression="tiff_lzw", dpi=dpi)
        meta["export_tiff"] = p.name
        paths["tiff"] = p
        p16 = out_dir / f"{stem}_hu.tif"
        write_hu_tiff(p16, img, dpi[0])
        meta["export_tiff_hu"] = p16.name
        meta["export_tiff_hu_offset"] = 0.0  # signed 16-bit: the pixel value is the HU
        meta["export_tiff_hu_spacing_mm"] = state.spacing_mm
        paths["tiff_hu"] = p16
    if dicom:
        d, uid = write_header_dicom(full, state, out_dir)
        meta["export_dicom_dir"] = d.name
        meta["export_dicom_series_uid"] = uid
        paths["dicom"] = d
    if json_sidecar:
        p = out_dir / f"{stem}.json"
        p.write_text(json.dumps(meta, indent=2, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)), encoding="utf-8")
        paths["json"] = p
    return paths
