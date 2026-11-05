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
