"""Swap TOP/BOTTOM labels, and the optional straightening rotation (single scans and stitched pairs)."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pydicom

import core_stitch.header_state as hs
from core_stitch.header_state import HeaderState, export_header
from core_stitch.load import MprSeries
from core_stitch.pipeline import _prepare_half
from core_stitch.single import SingleState, set_straighten, set_upside_down, sidecar_single
from test_itk_stitch import Rig, phantom as truth_phantom


def tilted_bar_series(tilt_deg: float = 2.0) -> MprSeries:
    """A stack whose core (a bright bar) leans ``tilt_deg`` from vertical: larger columns as it goes down."""
    n, h, w = 3, 220, 140
    rr, cc = np.mgrid[0:h, 0:w]
    centre = 50.0 + (rr - 10.0) * np.tan(np.radians(tilt_deg))
    bar = (np.abs(cc - centre) < 12) & (rr > 10) & (rr < 210)
    sl = np.where(bar, 1100.0, -1000.0).astype(np.float32)
    sl = sl + np.random.default_rng(1).normal(0, 4, sl.shape).astype(np.float32)
    vol = np.stack([sl] * n)
    return MprSeries(volume=vol, spacing_mm=(2.0, 0.5, 0.5), path=Path("."), slice_paths=(), series_uid="9.9",
                     series_description="t", transfer_syntax="x", frame_of_reference_uid="f")


class SingleStraightenTests(unittest.TestCase):
    def test_default_leaves_every_pixel_as_scanned_and_records_the_tilt(self) -> None:
        s = tilted_bar_series(2.0)
        half, _ = _prepare_half(s, 0.5, frame=1, straighten=False)
        self.assertTrue(np.array_equal(half.image, s.volume[1]))  # not one value changed
        self.assertEqual(half.upright_deg, 0.0)  # no rotation applied
        self.assertAlmostEqual(abs(half.tilt_deg), 2.0, delta=0.4)  # but the lean is measured

    def test_straightening_rotates_so_the_core_is_vertical(self) -> None:
        s = tilted_bar_series(2.0)
        half, _ = _prepare_half(s, 0.5, frame=1, straighten=True)
        self.assertFalse(np.array_equal(half.image[: s.volume.shape[1]], s.volume[1]))
        self.assertAlmostEqual(half.upright_deg, -half.tilt_deg, places=6)
        self.assertLess(abs(half.geom.angle_deg), 0.4)  # now vertical

    def test_toggle_and_sidecar_say_what_was_done(self) -> None:
        s = tilted_bar_series(2.0)
        half, w = _prepare_half(s, 0.5, frame=1, straighten=False)
        st = SingleState(series=s, half=half, frame=1, warnings=w)
        meta = sidecar_single(st)
        self.assertFalse(meta["straighten"]["applied"])
        self.assertIsNone(meta["straighten"]["interpolation"])
        self.assertAlmostEqual(abs(meta["straighten"]["measured_tilt_deg"]), 2.0, delta=0.4)
        self.assertNotIn("series", meta)  # the worklist's own "series" block must not be overwritten
        self.assertEqual(meta["scan"]["series_uid"], "9.9")
        set_straighten(st, True)
        meta = sidecar_single(st)
        self.assertTrue(meta["straighten"]["applied"])
        self.assertIn("linear", meta["straighten"]["interpolation"])
        self.assertLess(abs(st.half.geom.angle_deg), 0.4)
        set_straighten(st, False)
        self.assertTrue(np.array_equal(st.half.image, s.volume[1]))


class HeaderSwapAndStraightenTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.rig = Rig(self.dir)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_swapped_labels_exchange_the_roles_and_are_recorded(self) -> None:
        st = HeaderState.load(self.rig.top_dir, "1.2.3.4", self.rig.btm_dir, "1.2.3.5", swapped=True)
        self.assertTrue(st.swapped)
        self.assertEqual(st.top.series.series_uid, "1.2.3.5")
        self.assertEqual(st.bottom.series.series_uid, "1.2.3.4")
        out = self.dir / "swapped"
        paths = export_header(st, out, stem="s", dicom=False)
        meta = json.loads(paths["json"].read_text(encoding="utf-8"))
        self.assertTrue(meta["top_bottom_swapped"])
        self.assertEqual(meta["top"]["series_uid"], "1.2.3.5")
        plain = HeaderState.load(self.rig.top_dir, "1.2.3.4", self.rig.btm_dir, "1.2.3.5")
        self.assertFalse(plain.swapped)
        meta2 = json.loads(export_header(plain, self.dir / "plain", stem="s", dicom=False)["json"].read_text(encoding="utf-8"))
        self.assertFalse(meta2["top_bottom_swapped"])

    def test_straightening_a_stitch_keeps_the_geometry_exact_and_is_recorded(self) -> None:
        st = HeaderState.load(self.rig.top_dir, "1.2.3.4", self.rig.btm_dir, "1.2.3.5")
        off = export_header(st, self.dir / "off", stem="s")
        meta_off = json.loads(off["json"].read_text(encoding="utf-8"))
        self.assertFalse(meta_off["straighten"]["applied"])
        st.straighten = True
        with mock.patch.object(hs, "_measured_tilt", return_value=3.0):  # the phantom is straight; pretend it leans 3 degrees
            on = export_header(st, self.dir / "on", stem="s")
        meta = json.loads(on["json"].read_text(encoding="utf-8"))
        self.assertTrue(meta["straighten"]["applied"])
        self.assertEqual(meta["straighten"]["measured_tilt_deg"], 3.0)
        self.assertIn("linear", meta["straighten"]["interpolation"])
        # the join quality describes the stitch as rendered, so it is the same with or without the rotation
        self.assertEqual(meta["join_quality"]["correlation"], meta_off["join_quality"]["correlation"])
        f_off = sorted(off["dicom"].glob("*.dcm"))
        f_on = sorted(on["dicom"].glob("*.dcm"))
        self.assertEqual(len(f_on), len(f_off))
        a = pydicom.dcmread(f_off[len(f_off) // 2])
        b = pydicom.dcmread(f_on[len(f_on) // 2])
        ra, rb = np.asarray(a.ImageOrientationPatient[:3], float), np.asarray(b.ImageOrientationPatient[:3], float)
        ca, cb = np.asarray(a.ImageOrientationPatient[3:], float), np.asarray(b.ImageOrientationPatient[3:], float)
        self.assertAlmostEqual(float(np.dot(rb, cb)), 0.0, places=6)  # still orthogonal
        self.assertAlmostEqual(float(np.degrees(np.arccos(np.clip(np.dot(ra, rb), -1, 1)))), 3.0, places=3)
        self.assertAlmostEqual(float(np.degrees(np.arccos(np.clip(np.dot(ca, cb), -1, 1)))), 3.0, places=3)
        # every pixel of the rotated DICOM still matches the phantom at its own physical position
        arr = b.pixel_array.astype(float)
        ipp = np.asarray(b.ImagePositionPatient, float)
        rr, cc = np.meshgrid(np.arange(arr.shape[0]), np.arange(arr.shape[1]), indexing="ij")
        pts = ipp + cc[..., None] * float(b.PixelSpacing[1]) * rb + rr[..., None] * float(b.PixelSpacing[0]) * cb
        truth = truth_phantom(pts[..., 0], pts[..., 1], pts[..., 2])
        sel = (truth > 600) & (arr > -1000)
        self.assertGreater(int(sel.sum()), 5000)
        self.assertLess(float(np.abs(arr - truth)[sel].mean()), 10.0)


class UpsideDownTests(unittest.TestCase):
    def test_single_scan_turned_180_is_exact_and_reversible(self) -> None:
        s = tilted_bar_series(2.0)
        half, w = _prepare_half(s, 0.5, frame=1, straighten=False)
        st = SingleState(series=s, half=half, frame=1, warnings=w)
        set_upside_down(st, True)
        self.assertTrue(np.array_equal(st.half.image, np.rot90(s.volume[1], 2)))  # reversed rows and columns, nothing else
        self.assertTrue(np.array_equal(st.half.mask, np.rot90(half.mask, 2)))
        meta = sidecar_single(st)
        self.assertTrue(meta["rotate_180"]["applied"])
        set_upside_down(st, False)
        self.assertTrue(np.array_equal(st.half.image, s.volume[1]))
        self.assertFalse(sidecar_single(st)["rotate_180"]["applied"])

    def test_stitched_pair_turned_180_keeps_every_voxel_where_it_was(self) -> None:
        with tempfile.TemporaryDirectory() as t:
            rig = Rig(Path(t))
            st = HeaderState.load(rig.top_dir, "1.2.3.4", rig.btm_dir, "1.2.3.5")
            plain = st.res.volume[0].copy()
            off = export_header(st, Path(t) / "off", stem="s")
            meta_off = json.loads(off["json"].read_text(encoding="utf-8"))
            st.set_upside_down(True)
            self.assertTrue(np.array_equal(st.res.volume[0], plain[::-1, ::-1]))  # the preview is the turned picture
            on = export_header(st, Path(t) / "on", stem="s")
            meta = json.loads(on["json"].read_text(encoding="utf-8"))
            self.assertTrue(meta["rotate_180"]["applied"])
            self.assertFalse(meta_off["rotate_180"]["applied"])
            # the join rows are mirrored in the turned picture
            h = meta["export_shape_px"][0]
            a, b = meta_off["join"]["top_rows_in_image"]
            self.assertEqual(meta["join"]["top_rows_in_image"], [h - 1 - b, h - 1 - a])
            # the DICOM stack describes the same volume: orientation vectors negated, and every pixel still matches the phantom
            f_off, f_on = sorted(off["dicom"].glob("*.dcm")), sorted(on["dicom"].glob("*.dcm"))
            d0, d1 = pydicom.dcmread(f_off[len(f_off) // 2]), pydicom.dcmread(f_on[len(f_on) // 2])
            self.assertTrue(np.allclose(np.asarray(d1.ImageOrientationPatient, float), -np.asarray(d0.ImageOrientationPatient, float), atol=1e-6))
            self.assertTrue(np.array_equal(d1.pixel_array, d0.pixel_array[::-1, ::-1]))  # exactly the same values, turned around
            arr = d1.pixel_array.astype(float)
            ipp = np.asarray(d1.ImagePositionPatient, float)
            row, col = np.asarray(d1.ImageOrientationPatient[:3], float), np.asarray(d1.ImageOrientationPatient[3:], float)
            rr, cc = np.meshgrid(np.arange(arr.shape[0]), np.arange(arr.shape[1]), indexing="ij")
            pts = ipp + cc[..., None] * float(d1.PixelSpacing[1]) * row + rr[..., None] * float(d1.PixelSpacing[0]) * col
            truth = truth_phantom(pts[..., 0], pts[..., 1], pts[..., 2])
            sel = (truth > 600) & (arr > -1000)
            self.assertGreater(int(sel.sum()), 5000)
            self.assertLess(float(np.abs(arr - truth)[sel].mean()), 3.0)


class SwapReorderTests(unittest.TestCase):
    def test_swap_puts_the_other_half_below_and_joins_by_the_images_when_headers_cannot(self) -> None:
        with tempfile.TemporaryDirectory() as t:
            # the lower half's header is 250 mm out, so the headers cannot place the pair; and the catalog labels are
            # backwards: "TOP" is really the lower half
            rig = Rig(Path(t), btm_header_error_z=250.0)
            st = HeaderState.load(rig.btm_dir, "1.2.3.5", rig.top_dir, "1.2.3.4", swapped=True)
            self.assertEqual(st.top.series.series_uid, "1.2.3.4")  # after the swap the upper half is the TOP role
            st.realign_from_image(require_below=True)
            self.assertGreater(st.join_correlation(), 0.95)  # the images agree in the re-ordered picture
            r = st.res
            top_is_ref = r.reference.name == "top"
            fp_top = (r.ref_footprint if top_is_ref else r.mov_footprint)[0]
            fp_btm = (r.mov_footprint if top_is_ref else r.ref_footprint)[0]
            first_row = lambda fp: int(np.where(fp.any(axis=1))[0][0])
            self.assertLess(first_row(fp_top), first_row(fp_btm))  # TOP role above BOTTOM role, each upright
            meta = json.loads(export_header(st, Path(t) / "out", stem="s", dicom=False)["json"].read_text(encoding="utf-8"))
            self.assertTrue(meta["top_bottom_swapped"])
            self.assertIn("image alignment", meta["placement"]["source"])


class TurnOneHalfTests(unittest.TestCase):
    def test_turning_the_upside_down_half_back_restores_the_exact_stitch(self) -> None:
        with tempfile.TemporaryDirectory() as t:
            ok = Rig(Path(t) / "ok")
            upside = Rig(Path(t) / "up", btm_upside_down=True)
            ref = HeaderState.load(ok.top_dir, "1.2.3.4", ok.btm_dir, "1.2.3.5")
            plain = HeaderState.load(upside.top_dir, "1.2.3.4", upside.btm_dir, "1.2.3.5")
            fixed = HeaderState.load(upside.top_dir, "1.2.3.4", upside.btm_dir, "1.2.3.5", bottom_turned=True)
            self.assertTrue(fixed.bottom_turned and not fixed.top_turned)
            self.assertFalse(np.array_equal(plain.res.volume, ref.res.volume))  # as scanned it is wrong
            self.assertTrue(np.array_equal(fixed.res.volume, ref.res.volume))  # turned back: not one value differs
            meta = json.loads(export_header(fixed, Path(t) / "out", stem="s", dicom=False)["json"].read_text(encoding="utf-8"))
            self.assertTrue(meta["bottom"]["turned_180"])
            self.assertFalse(meta["top"]["turned_180"])

    def test_turned_half_is_placed_by_the_images_when_the_headers_cannot(self) -> None:
        with tempfile.TemporaryDirectory() as t:
            rig = Rig(Path(t), btm_upside_down=True, btm_header_error_z=40.0)
            st = HeaderState.load(rig.top_dir, "1.2.3.4", rig.btm_dir, "1.2.3.5", bottom_turned=True)
            st.realign_from_image()  # uses the turned slices, not the files as stored
            self.assertGreater(st.join_correlation(), 0.95)


if __name__ == "__main__":
    unittest.main()
