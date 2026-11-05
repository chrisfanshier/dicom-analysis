"""Scanner-geometry (SimpleITK) mode: state, export and the main window."""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pydicom

from core_stitch.header_state import HeaderState, export_header
from core_stitch.itk_stitch import render_plan
from test_itk_stitch import Rig, phantom
from test_worklist import make_catalog


class HeaderStateTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.rig = Rig(self.dir)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def _state(self, rig=None, **kw) -> HeaderState:
        rig = rig or self.rig
        return HeaderState.load(rig.top_dir, "1.2.3.4", rig.btm_dir, "1.2.3.5", **kw)

    def test_one_slice_render_equals_the_same_slice_of_the_full_volume(self) -> None:
        st = self._state()
        st.set_frame(3)
        full = render_plan(st.plan, trim_rows=3)
        self.assertTrue(np.array_equal(st.res.volume[0], full.volume[3]))
        self.assertEqual(st.n_frames, full.volume.shape[0])

    def test_shift_moves_the_bottom_half_and_reset_restores_it(self) -> None:
        st = self._state()
        base = st.res.volume[0].copy()
        st.set_move(4.0, 0.0, 0.0)
        self.assertFalse(np.array_equal(base, st.res.volume[0]))
        self.assertEqual(st.move_mm, [4.0, 0.0, 0.0])
        st.set_move(0.0, 0.0, 0.0, source="scanner header")
        self.assertTrue(np.array_equal(base, st.res.volume[0]))

    def test_header_check_accepts_good_headers_and_flags_a_wrong_one(self) -> None:
        self.assertEqual(self._state().check["verdict"], "headers_good")
        with tempfile.TemporaryDirectory() as t:
            bad = Rig(Path(t), btm_header_error_z=40.0)  # header puts the bottom half 40 mm out
            chk = self._state(bad).check
            self.assertEqual(chk["verdict"], "needs_image_stitch")
            self.assertTrue(chk["reasons"])

    def test_refine_recovers_a_small_header_error_along_and_across(self) -> None:
        with tempfile.TemporaryDirectory() as t:
            rig = Rig(Path(t), btm_header_error_z=2.0, btm_header_error_x=1.5)  # header puts the bottom half 2 mm high, 1.5 mm right
            st = self._state(rig)
            out = st.refine()
            self.assertTrue(out.applied, out.note)
            self.assertAlmostEqual(out.along_mm, 2.0, delta=0.3)  # down 2 mm fixes a half that sits 2 mm high
            self.assertAlmostEqual(out.across_mm, -1.5, delta=0.3)
            self.assertGreater(out.score_after, out.score_before)
            self.assertGreater(out.score_after, 0.99)
            self.assertIn("refine", st.moved_by)
            self.assertEqual(st.refine_info["shift_mm"]["along_core"], out.along_mm)
            self.assertEqual(st.move_mm[2], 0.0)  # depth is never touched by refine
            st.set_move(*st.move_mm)  # a manual change afterwards drops the refine record
            self.assertIsNone(st.refine_info)

    def test_undo_refine_restores_the_earlier_position_and_the_live_correlation_is_shown(self) -> None:
        with tempfile.TemporaryDirectory() as t:
            rig = Rig(Path(t), btm_header_error_z=2.0)
            st = self._state(rig)
            self.assertIn("Join correlation on this slice", st.summary())
            before = st.join_correlation()
            self.assertTrue(st.refine().applied)
            self.assertTrue(st.can_undo_refine())
            self.assertGreater(st.join_correlation(), before)
            self.assertTrue(st.undo_refine())
            self.assertEqual(st.move_mm, [0.0, 0.0, 0.0])
            self.assertIsNone(st.refine_info)
            self.assertAlmostEqual(st.join_correlation(), before, places=6)
            self.assertFalse(st.undo_refine())  # nothing left to undo
            # a manual shift after a refine drops the undo
            st.refine()
            st.set_move(0.0, 0.0, 0.0)
            self.assertFalse(st.can_undo_refine())

    def test_refine_leaves_a_correct_placement_alone(self) -> None:
        st = self._state()
        out = st.refine()
        self.assertFalse(out.applied)
        self.assertEqual(st.move_mm, [0.0, 0.0, 0.0])

    def test_refine_with_whole_pixel_placement_is_never_worse_and_says_when_nothing_changes(self) -> None:
        with tempfile.TemporaryDirectory() as t:
            # equal pixel sizes, bottom half 0.3 px off the lattice and 0.5 mm off along the core
            rig = Rig(Path(t), px_top=0.5, px_btm=0.5, btm_offset=(0.15, 0.0, 0.0), btm_header_error_z=0.5)
            st = self._state(rig)
            before = st.join_correlation()
            out = st.refine()
            after = st.join_correlation()
            self.assertGreaterEqual(after, before - 0.002)  # rounding the optimum may not lose ground
            if not out.applied:
                self.assertTrue(out.note)
            st.set_placement("subpixel")
            out2 = st.refine()
            self.assertGreater(st.join_correlation(), 0.99)

    def test_refine_note_says_which_resampling_applies(self) -> None:
        self.assertIn("already resampled once", self._state().refine_note())  # 0.5 vs 0.4 mm
        with tempfile.TemporaryDirectory() as t:
            same = Rig(Path(t), px_top=0.5, px_btm=0.5)
            st = self._state(same)
            self.assertIn("rounded to a whole pixel", st.refine_note())
            st.set_placement("subpixel")
            self.assertIn("resamples the bottom half once", st.refine_note())

    def test_alignment_colours_show_each_half_in_its_own_colour(self) -> None:
        st = self._state()
        rgb = st.align_preview(1.0, 1.0)
        self.assertEqual(rgb.ndim, 3)
        self.assertEqual(rgb.shape[2], 3)
        self.assertEqual(rgb.dtype, np.uint8)
        top_fp = (st.res.ref_footprint if st.res.reference.name == "top" else st.res.mov_footprint)[0]
        btm_fp = (st.res.ref_footprint if st.res.reference.name == "bottom" else st.res.mov_footprint)[0]
        r0, r1, c0, c1 = st.crop_box(3.0)
        tf, bf = top_fp[r0:r1, c0:c1], btm_fp[r0:r1, c0:c1]
        core = rgb.max(axis=2) > 0
        top_only = core & tf & ~bf
        btm_only = core & bf & ~tf
        both = core & tf & bf
        self.assertGreater(int(top_only.sum()), 100)
        self.assertGreater(int(btm_only.sum()), 100)
        self.assertGreater(int(both.sum()), 100)
        self.assertTrue(np.all(rgb[top_only][:, 1] == 0))  # top only: magenta (no green)
        self.assertTrue(np.all(rgb[top_only][:, 0] == rgb[top_only][:, 2]))
        self.assertTrue(np.all(rgb[btm_only][:, [0, 2]] == 0))  # bottom only: green
        # overlap: most pixels carry both colours (pixels below the display window are black in a channel)
        self.assertGreater(float((rgb[both][:, 1] > 0).mean()), 0.8)
        self.assertGreater(float((rgb[both][:, 0] > 0).mean()), 0.8)

    def test_export_products_geometry_and_plain_sidecar(self) -> None:
        st = self._state()
        out = self.dir / "out"
        scale: dict = {}
        paths = export_header(st, out, stem="s", scale_out=scale, extra_meta={"catalog_section_length_cm": 20.0})
        for key in ("jpeg", "tiff", "tiff_hu", "dicom", "json"):
            self.assertIn(key, paths)
            self.assertTrue(paths[key].exists())
        meta = json.loads(paths["json"].read_text(encoding="utf-8"))
        for block in ("resolution", "placement", "header_check", "join", "core_scale", "top", "bottom"):
            self.assertIn(block, meta)
        self.assertEqual(meta["resolution"]["reference_half"], "bottom")  # 0.4 mm is the finer pixel
        self.assertEqual(meta["resolution"]["interpolated_half"], "top")
        self.assertAlmostEqual(meta["resolution"]["upsample_factor_of_interpolated_half"], 1.25, places=2)
        self.assertIsNotNone(meta["join"]["overlap_rows_in_image"])
        self.assertAlmostEqual(meta["join"]["overlap_cm"], 3.0, delta=0.2)  # 30 mm of overlap
        self.assertEqual(meta["core_scale"]["length_check"], "compared")
        text = json.dumps(meta)  # nothing that looks like a folder path
        self.assertNotIn(str(self.dir), text)
        self.assertNotIn("\\\\", text)
        self.assertNotIn("series_dir", text)
        # files are named, not located
        self.assertEqual(meta["export_jpeg"], "s.jpg")

        # DICOM: geometry is the scanner's, and pixels match the phantom's true values
        files = sorted((paths["dicom"]).glob("*.dcm"))
        self.assertEqual(len(files), st.n_frames)
        ds = pydicom.dcmread(files[len(files) // 2])
        ref_iop = [float(x) for x in pydicom.dcmread(sorted(self.rig.btm_dir.glob("*.dcm"))[0]).ImageOrientationPatient]
        self.assertEqual([round(float(x), 6) for x in ds.ImageOrientationPatient], ref_iop)
        self.assertAlmostEqual(float(ds.PixelSpacing[0]), 0.4, places=6)
        arr = ds.pixel_array.astype(float)
        ipp = np.asarray(ds.ImagePositionPatient, float)
        row = np.asarray(ds.ImageOrientationPatient[:3], float)
        col = np.asarray(ds.ImageOrientationPatient[3:], float)
        rr, cc = np.meshgrid(np.arange(arr.shape[0]), np.arange(arr.shape[1]), indexing="ij")
        pts = ipp + cc[..., None] * float(ds.PixelSpacing[1]) * row + rr[..., None] * float(ds.PixelSpacing[0]) * col
        truth = phantom(pts[..., 0], pts[..., 1], pts[..., 2])
        sel = (truth > 600) & (arr > -1000)
        self.assertGreater(int(sel.sum()), 5000)
        self.assertLess(float(np.abs(arr - truth)[sel].mean()), 3.0)
        # background is real air, not the median core value
        self.assertLess(float(np.median(arr[truth < -900])), -900)


class HeaderGuiTests(unittest.TestCase):
    def setUp(self) -> None:
        import tkinter as tk

        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.release = self.dir / "release"
        make_catalog(self.release)
        self.rig = Rig(self.dir / "rig")
        self._env = {k: os.environ.get(k) for k in ("CORE_STITCH_RELEASE", "CORE_STITCH_YTOOLS_CONFIG")}
        os.environ["CORE_STITCH_RELEASE"] = str(self.release)
        os.environ["CORE_STITCH_YTOOLS_CONFIG"] = str(self.dir / "cfg")
        (self.dir / "cfg").mkdir()
        from core_stitch.gui import StitchGui

        try:
            self.app = StitchGui()
        except tk.TclError:
            self.skipTest("no display")
        self.app.geometry("1200x800")
        self.app.update()

    def tearDown(self) -> None:
        try:
            self.app.destroy()
        except Exception:
            pass
        for k, v in self._env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        self.tmp.cleanup()

    def test_load_shift_and_export_in_the_main_window(self) -> None:
        app = self.app
        app.path_a, app.path_b = self.rig.top_dir, self.rig.btm_dir
        app.uid_a, app.uid_b = "1.2.3.4", "1.2.3.5"
        app.load_header_pair()
        app.update()
        app.refresh()
        app.update()
        st = app.single
        self.assertIsInstance(st, HeaderState)
        self.assertIsNone(app.state)
        self.assertTrue(app._hdr_active())
        self.assertIn("consistent", app.hdr_check.get())
        self.assertIn("Reference: bottom", app.hdr_info.get())
        self.assertIsNotNone(app._full_im)
        self.assertGreaterEqual(len(app.canvas.find_withtag("coreguide")), 4)
        self.assertEqual([c for c in app.winfo_children() if c.winfo_class() == "Toplevel"], [])

        # the alignment-colours checkbox switches the picture between colour overlay and grey
        app.align_var.set(True)
        app.refresh()
        self.assertEqual(app._full_im.mode, "RGB")
        app.align_var.set(False)
        app.refresh()
        self.assertEqual(app._full_im.mode, "L")

        # arrow keys shift the bottom half (Down = along the core); the controls follow
        app._nudge_keys(0, 5)
        app._hdr_apply()
        app.update()
        self.assertAlmostEqual(st.move_mm[0], 5.0)
        self.assertIn("shift along +5.0", app.hdr_info.get())
        app._hdr_reset()
        self.assertEqual(st.move_mm, [0.0, 0.0, 0.0])
        self.assertEqual(app.hdr_along.get(), 0.0)

        # slice browsing renders the requested slice
        app.z_var.set(2)
        app._apply_z()
        self.assertEqual(st.frame, 2)

        # export with the window's own controls (no worklist job): choose a folder
        out = self.dir / "export"
        with mock.patch("tkinter.filedialog.askdirectory", return_value=str(out)), \
             mock.patch("tkinter.messagebox.showinfo"), mock.patch("tkinter.messagebox.showwarning"):
            app.accept_export()
        meta = json.loads(next(out.glob("*.json")).read_text(encoding="utf-8"))
        self.assertEqual(meta["engine"], "simpleitk_header_geometry")
        self.assertTrue(list(out.glob("dicom_mpr/*.dcm")))
        self.assertTrue(list(out.glob("*.jpg")))


if __name__ == "__main__":
    unittest.main()
