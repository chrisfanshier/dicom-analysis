"""The profile viewer: load a coronal MPR, scroll, window, axial cut, core-end guides (offscreen Qt)."""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

import numpy as np

from test_core_profile_io import write_mpr_cylinder

try:
    import PySide6  # noqa: F401
    import pyqtgraph  # noqa: F401

    HAVE_QT = True
except ImportError:
    HAVE_QT = False


class PureTests(unittest.TestCase):
    @unittest.skipUnless(HAVE_QT, "PySide6 / pyqtgraph not installed")
    def test_axial_image_is_square_pixels(self) -> None:
        from core_profile.app import axial_image

        vol = np.random.default_rng(0).normal(size=(10, 50, 40)).astype(np.float32)
        a = axial_image(vol, 7, dz_mm=2.0, dc_mm=0.5)
        self.assertEqual(a.shape, (40, 40))  # 10 slices x 2 mm = 20 mm = 40 px of 0.5 mm
        b = axial_image(vol, 7, dz_mm=0.5, dc_mm=0.5)
        np.testing.assert_array_equal(b, vol[:, 7, :])

    @unittest.skipUnless(HAVE_QT, "PySide6 / pyqtgraph not installed")
    def test_histogram_range_pads_percentiles(self) -> None:
        from core_profile.app import histogram_range

        lo, hi = histogram_range(np.linspace(0, 1000, 10001))
        self.assertLess(lo, 5.0)
        self.assertGreater(hi, 995.0)
        self.assertGreater(hi, lo)


@unittest.skipUnless(HAVE_QT, "PySide6 / pyqtgraph not installed")
class ViewerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from PySide6 import QtWidgets

        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        cls.tmp = tempfile.TemporaryDirectory()
        cls.dir = Path(cls.tmp.name) / "mpr"
        write_mpr_cylinder(cls.dir, length_mm=120.0)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.tmp.cleanup()

    def setUp(self) -> None:
        from core_profile.app import Viewer
        from core_stitch.load import load_series

        self.v = Viewer()
        self.v.load_series_obj(load_series(self.dir), sync=True)

    def tearDown(self) -> None:
        self.v.close()
        self.v.deleteLater()

    def test_loads_at_middle_slice_with_auto_window_and_ends(self) -> None:
        v = self.v
        self.assertEqual(v.frame, v.n_frames // 2)
        lo, hi = v.win
        self.assertLess(lo, hi)
        # the core fills the image length except for the unfilled margins: ends are near the image ends
        self.assertAlmostEqual(v.extent_cm, 12.0, delta=0.05)
        self.assertLess(v.top_cm + v.bottom_cm, 1.0)
        self.assertEqual(v.bounds_source, "auto")

    def test_scrolling_changes_slice_and_clamps(self) -> None:
        v = self.v
        k = v.frame
        v.set_frame(k + 1)
        self.assertEqual(v.frame, k + 1)
        self.assertEqual(v.z_spin.value(), k + 1)
        v.set_frame(10**6)
        self.assertEqual(v.frame, v.n_frames - 1)
        v.set_frame(-5)
        self.assertEqual(v.frame, 0)
        self.assertIn("Coronal slice: 0", v.z_label.text())

    def test_window_boxes_and_auto(self) -> None:
        v = self.v
        a0 = v.win
        lo, hi = v.hist_range
        b, w = lo + 0.3 * (hi - lo), lo + 0.8 * (hi - lo)  # the handles live inside the histogram range
        v.black.spin.setValue(b)
        v.white.spin.setValue(w)
        self.assertAlmostEqual(v.win[0], b, delta=0.1)  # the boxes show one decimal
        self.assertAlmostEqual(v.win[1], w, delta=0.1)
        self.assertAlmostEqual(v.region.getRegion()[0], b, delta=0.5)
        self.assertAlmostEqual(v.cor_img.levels[1], w, delta=0.5)
        v.auto_window()
        self.assertAlmostEqual(v.win[0], a0[0], delta=1e-6)
        self.assertAlmostEqual(v.win[1], a0[1], delta=1e-6)
        self.assertAlmostEqual(v.black.value(), a0[0], delta=0.1)

    def test_histogram_handles_move_window(self) -> None:
        v = self.v
        lo, hi = v.win
        v.region.setRegion((lo + 10, hi - 10))
        self.assertAlmostEqual(v.win[0], lo + 10, delta=1e-6)
        self.assertAlmostEqual(v.white.value(), hi - 10, delta=0.2)

    def test_axial_depth_follows_line_and_box(self) -> None:
        v = self.v
        v.depth_spin.setValue(4.0)
        self.assertAlmostEqual(v.depth_mm, 40.0, delta=0.01)
        self.assertAlmostEqual(v.depth_line.value(), 40.0, delta=0.01)
        v.depth_line.setValue(55.0)
        self.assertAlmostEqual(v.depth_mm, 55.0, delta=0.01)
        self.assertIn("5.50 cm", v.ax_title.text())

    def test_dragging_axial_frame_line_changes_coronal_slice(self) -> None:
        v = self.v
        v.frame_line.setValue(5.5 * v.dz + 0.1)
        self.assertEqual(v.frame, 5)

    def test_guides_and_boxes_stay_in_step(self) -> None:
        v = self.v
        v.top_edit.setValue(1.5)
        self.assertEqual(v.bounds_source, "manual")
        self.assertAlmostEqual(v.top_line.value(), 15.0, delta=0.01)
        v.bot_line.setValue((v.extent_cm - 2.0) * 10)  # drag the bottom guide to a 2 cm inset
        self.assertAlmostEqual(v.bot_edit.value(), 2.0, delta=0.01)
        self.assertAlmostEqual(float(v.len_label.text().split()[2]), v.extent_cm - 3.5, delta=0.02)
        v.top_edit.setValue(10**4)  # cannot cross the bottom guide
        self.assertLessEqual(v.top_cm + v.bottom_cm, v.extent_cm)
        v.reset_ends()
        self.assertEqual(v.bounds_source, "auto")
        self.assertAlmostEqual(v.top_cm, v.auto_ends[0], delta=1e-6)

    def test_goto_end_moves_axial_cut(self) -> None:
        v = self.v
        v.top_edit.setValue(2.0)
        v.goto_end("top")
        self.assertAlmostEqual(v.depth_mm, 20.0, delta=v.dr)

    def test_mask_and_profile_after_load(self) -> None:
        v = self.v
        self.assertIsNotNone(v.mask3d)
        self.assertEqual(v.mask3d.shape, v.series.volume.shape)
        mean, sd, cnt = v.profile
        n = len(mean)
        self.assertAlmostEqual(float(np.nanmedian(mean[n // 10 : 2 * n // 5])), 800.0, delta=10)
        self.assertAlmostEqual(float(np.nanmedian(mean[3 * n // 5 : 9 * n // 10])), 848.0, delta=10)
        self.assertEqual(len(v.prof_curve.getData()[0]), n)
        # between liner and sediment
        self.assertTrue(300 < v.t_sed < 800, v.t_sed)
        self.assertAlmostEqual(v.t_line.value(), v.t_sed)
        self.assertAlmostEqual(v.t_spin.value(), v.t_sed, delta=1.0)

    def test_threshold_marker_and_box_stay_in_step(self) -> None:
        v = self.v
        v.t_line.setValue(600.0)
        self.assertAlmostEqual(v.t_sed, 600.0)
        self.assertAlmostEqual(v.t_spin.value(), 600.0)
        self.assertIsNone(v.mask3d)  # stale until the pause is over
        v.recompute_mask(sync=True)
        self.assertIsNotNone(v.mask3d)
        v.t_spin.setValue(700.0)
        self.assertAlmostEqual(v.t_line.value(), 700.0)
        v.auto_threshold()
        self.assertTrue(300 < v.t_sed < 800)

    def test_margin_and_threshold_change_the_mask(self) -> None:
        v = self.v
        a = int(v.mask3d.sum())
        v.margin_spin.setValue(4.0)
        v.recompute_mask(sync=True)
        self.assertLess(int(v.mask3d.sum()), a)
        v.t_spin.setValue(5000.0)  # above everything: nothing is sediment
        v.recompute_mask(sync=True)
        self.assertEqual(int(v.mask3d.sum()), 0)
        self.assertTrue(np.isnan(v.profile[0]).all())

    def test_overlay_visibility(self) -> None:
        v = self.v
        self.assertTrue(v.cor_mask.isVisible() or v.show_mask.isChecked())
        v.show_mask.setChecked(False)
        self.assertFalse(v.cor_mask.isVisible())
        self.assertFalse(v.ax_mask.isVisible())
        v.show_mask.setChecked(True)
        self.assertTrue(v.show_mask.isChecked())

    def test_threaded_recompute_finishes(self) -> None:
        import time

        v = self.v
        v.margin_spin.setValue(2.0)  # starts the pause timer
        v.recompute_mask()  # worker thread
        t0 = time.time()
        while v.mask3d is None and time.time() - t0 < 30:
            self.app.processEvents()
            time.sleep(0.02)
        self.assertIsNotNone(v.mask3d)

    def test_cylinder_is_off_until_asked(self) -> None:
        v = self.v
        self.assertFalse(v.cyl_box.isChecked())
        self.assertIsNone(v._circle())
        self.assertFalse(v.roi.isVisible())
        self.assertAlmostEqual(v.liner_spin.value(), 6.35)

    def test_auto_fit_draws_the_circle_inside_the_liner(self) -> None:
        v = self.v
        t0 = v.t_sed
        v._autofit_clicked()
        self.assertTrue(v.cyl_box.isChecked())
        cx, cy, r = v._circle()
        # the synthetic core: outer radius 30 mm, axis at x = 36 mm and stack position 35 mm
        self.assertAlmostEqual(cx, 36.0, delta=0.8)
        self.assertAlmostEqual(cy, 35.0, delta=1.0)
        self.assertAlmostEqual(r, 30.0 - 6.35, delta=0.9)
        self.assertAlmostEqual(v.roi.size()[0] / 2, r, delta=1e-6)
        # the cylinder does not move the threshold (so what counts as a hole does not change), and its line is on screen
        self.assertAlmostEqual(v.t_sed, t0)
        self.assertAlmostEqual(v.t_spin.value(), t0, delta=1.0)
        x0, x1 = v.hist.viewRange()[0]
        self.assertTrue(x0 <= v.t_line.value() <= x1)

    def test_cylinder_mask_stays_inside_the_circle_and_thickness_moves_it(self) -> None:
        v = self.v
        v._autofit_clicked()
        v.recompute_mask(sync=True)
        cx, cy, r = v._circle()
        ks = np.where(v.mask3d.any(axis=(1, 2)))[0]
        y = (ks + 0.5) * v.dz
        self.assertTrue(((y - cy) ** 2 < r * r).all())  # no slice outside the circle has any mask
        cols = np.where(v.mask3d.any(axis=(0, 1)))[0]
        self.assertGreaterEqual(cols.min() * v.dc, cx - r - 0.5 * v.dc)
        self.assertLessEqual((cols.max() + 1) * v.dc, cx + r + 0.5 * v.dc)
        a = int(v.mask3d.sum())
        v.liner_spin.setValue(9.35)  # a thicker liner wall: the circle shrinks by 3 mm, its centre stays
        cx2, cy2, r2 = v._circle()
        self.assertAlmostEqual(r2, r - 3.0, delta=1e-6)
        self.assertAlmostEqual(cx2, cx, delta=1e-6)
        v.recompute_mask(sync=True)
        self.assertLess(int(v.mask3d.sum()), a)

    def test_dragging_the_circle_changes_the_mask(self) -> None:
        v = self.v
        v._autofit_clicked()
        cx, cy, r = v._circle()
        v.roi.setSize((r, r))  # diameter r: half the size
        self.assertAlmostEqual(v._circle()[2], r / 2, delta=1e-6)
        self.assertIsNone(v.mask3d)
        v.recompute_mask(sync=True)
        self.assertGreater(int(v.mask3d.sum()), 0)

    def test_switching_the_cylinder_off_restores_the_liner_threshold(self) -> None:
        v = self.v
        t0 = v.t_sed
        v._autofit_clicked()
        v.cyl_box.setChecked(False)
        self.assertIsNone(v._circle())
        self.assertFalse(v.roi.isVisible())
        self.assertAlmostEqual(v.t_sed, t0, delta=1e-6)

    def test_auto_fit_with_no_core_says_so(self) -> None:
        v = self.v
        v.set_depth_mm(-1e6)  # clamps to the top row, which is air in the padded image; fit still may succeed
        v.t_air = 1e6  # nothing is above this, so there is nothing to fit
        self.assertFalse(v.auto_fit())
        self.assertIn("no circle", v.cyl_label.text())

    def test_follow_the_wobble(self) -> None:
        from core_profile.app import Viewer
        from core_stitch.load import load_series

        d = Path(self.tmp.name) / "wobbly"
        write_mpr_cylinder(d, length_mm=120.0, wobble_mm=6.0)
        w = Viewer()
        try:
            w.load_series_obj(load_series(d), sync=True)
            w.follow_box.setChecked(True)
            self.assertTrue(w.cyl_box.isChecked())
            self.assertIsNotNone(w._path)
            cx_r, cy_r = w._path
            self.assertGreater(float(np.ptp(cx_r[40:260])), 3.0)  # the fitted axis does drift
            self.assertIn("following the axis", w.cyl_label.text())
            # the circle sits on the axis at the depth shown, and moves with the depth
            w.set_depth_mm(20.0)
            c_a = w._circle_any()[0]
            w.set_depth_mm(70.0)
            c_b = w._circle_any()[0]
            self.assertAlmostEqual(c_a, float(cx_r[int(round(20.0 / w.dr))]), delta=0.05)
            self.assertAlmostEqual(c_b, float(cx_r[int(round(70.0 / w.dr))]), delta=0.05)
            self.assertGreater(abs(c_a - c_b), 1.0)
            followed = w.cyl_label.text()
            w.recompute_mask(sync=True)
            n_follow = int(w.mask3d.sum())
            out = Path(self.tmp.name) / "wobbly_profile.csv"
            w.save_profile(out)
            lines = out.read_text(encoding="utf-8").splitlines()
            cols = [ln for ln in lines if not ln.startswith("#")][0].split(",")
            self.assertEqual(cols[-2:], ["axis_x_mm", "axis_y_mm"])
            first = [ln for ln in lines if not ln.startswith("#")][1].split(",")
            r0 = int(round(float(first[0]) / w.dr - 0.5))
            self.assertAlmostEqual(float(first[5]), float(w._path[0][r0]), delta=0.001)
            # dragging the circle sideways shifts the whole path
            x0 = w._circle()[0].copy()
            pos = w.roi.pos()
            w.roi.setPos((float(pos[0]) + 2.0, float(pos[1])))
            self.assertAlmostEqual(float(w._off[0]), 2.0, delta=1e-6)
            self.assertAlmostEqual(float(np.mean(w._circle()[0] - x0)), 2.0, delta=1e-6)
            w.roi.setPos((float(pos[0]), float(pos[1])))
            # switching it off leaves one fixed circle at the current depth, which loses sediment where the core drifts
            w.follow_box.setChecked(False)
            self.assertIsNone(w._path)
            w.recompute_mask(sync=True)
            self.assertLess(int(w.mask3d.sum()), n_follow)
            self.assertTrue(followed)
        finally:
            w.close()
            w.deleteLater()

    def test_void_and_clast_checkboxes_default_to_the_old_mask(self) -> None:
        v = self.v
        self.assertFalse(v.void_box.isChecked())
        self.assertFalse(v.clast_box.isChecked())
        self.assertFalse(v.c_line.isVisible())
        self.assertEqual(v._mask_args()[2:5][0], True)  # fill holes on while "remove voids" is off
        self.assertIsNone(v._mask_args()[4])
        v.void_box.setChecked(True)
        self.assertFalse(v._mask_args()[2])

    def test_clast_line_box_and_cut(self) -> None:
        v = self.v
        a = int(v.mask3d.sum())
        v.clast_box.setChecked(True)
        self.assertTrue(v.c_line.isVisible())
        self.assertAlmostEqual(v.c_spin.value(), v.c_line.value(), delta=1.0)
        v.c_spin.setValue(820.0)  # the synthetic core: 800 above the middle, 848 below
        self.assertAlmostEqual(v.c_line.value(), 820.0)
        v.recompute_mask(sync=True)
        self.assertLess(int(v.mask3d.sum()), a)
        self.assertLess(float(np.nanmax(v.profile[0])), 830.0)  # the dense lower half is gone from the profile
        x0, x1 = v.hist.viewRange()[0]
        self.assertTrue(x0 <= 820.0 <= x1)
        v.c_line.setValue(900.0)
        self.assertAlmostEqual(v.c_spin.value(), 900.0)
        v.auto_clast()
        self.assertGreater(v.t_clast, v.t_sed)
        v.clast_box.setChecked(False)
        self.assertFalse(v.c_line.isVisible())
        self.assertIsNone(v._mask_args()[4])

    def test_save_profile_header_and_rows(self) -> None:
        import json

        v = self.v
        v.clast_box.setChecked(True)
        v.c_spin.setValue(820.0)
        v.recompute_mask(sync=True)
        out = Path(self.tmp.name) / "saved_profile.csv"
        self.assertEqual(v.save_profile(out), out)
        self.assertIn("Saved", v.save_label.text())
        lines = out.read_text(encoding="utf-8").splitlines()
        head = [ln for ln in lines if ln.startswith("#")]
        meta = json.loads([ln for ln in head if ln.startswith("# settings_json: ")][0][len("# settings_json: "):])
        self.assertEqual(meta["input"]["folder"], str(v.series.path))
        self.assertEqual(meta["input"]["n_slices"], v.n_frames)
        self.assertEqual(meta["mask"]["remove_clasts"], True)
        self.assertAlmostEqual(meta["mask"]["clast_threshold_hu"], 820.0)
        self.assertAlmostEqual(meta["mask"]["sediment_threshold_hu"], v.t_sed, delta=0.01)
        self.assertEqual(meta["mask"]["cylinder"]["on"], False)
        self.assertAlmostEqual(meta["core_ends"]["top_inset_cm"], v.top_cm, delta=0.001)
        rows = [ln.split(",") for ln in lines if not ln.startswith("#")][1:]
        lo, hi = v.top_cm * 10, (v.extent_cm - v.bottom_cm) * 10
        self.assertTrue(all(lo <= float(r[0]) <= hi for r in rows))
        r0 = int(round(float(rows[0][0]) / v.dr - 0.5))
        self.assertAlmostEqual(float(rows[0][2]), float(v.profile[0][r0]), delta=0.01)

    def test_save_profile_records_the_cylinder_and_axis(self) -> None:
        import json

        v = self.v
        v._autofit_clicked()
        v.recompute_mask(sync=True)
        out = Path(self.tmp.name) / "cyl_profile.csv"
        v.save_profile(out)
        head = [ln for ln in out.read_text(encoding="utf-8").splitlines() if ln.startswith("# settings_json: ")]
        cyl = json.loads(head[0][len("# settings_json: "):])["mask"]["cylinder"]
        self.assertTrue(cyl["on"])
        self.assertAlmostEqual(cyl["radius_mm"], v._circle()[2], delta=0.01)
        self.assertEqual(cyl["follow_axis"], False)
        self.assertAlmostEqual(cyl["liner_wall_mm"], 6.35)
        cols = [ln for ln in out.read_text(encoding="utf-8").splitlines() if not ln.startswith("#")][0]
        self.assertNotIn("axis_x_mm", cols)  # a fixed circle needs no per-row axis

    def test_save_waits_for_the_mask(self) -> None:
        v = self.v
        v.margin_spin.setValue(3.0)  # the stack mask is stale until the pause is over
        self.assertIsNone(v.mask3d)
        self.assertIsNone(v.save_profile(Path(self.tmp.name) / "nope.csv"))
        self.assertIn("still being computed", v.save_label.text())
        self.assertFalse((Path(self.tmp.name) / "nope.csv").exists())

    def test_hide_guides(self) -> None:
        self.v.show_guides.setChecked(False)
        self.assertFalse(self.v.top_line.isVisible())


if __name__ == "__main__":
    unittest.main()
