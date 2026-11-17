"""Sediment mask: thresholds, whole-round and split-core shapes, margin, profile per row."""

from __future__ import annotations

import unittest

import numpy as np

from core_profile.mask import auto_clast_threshold, auto_thresholds, circle_span, fit_axis_path, fit_outer_circle, mask_2d, mask_3d, ransac_circle, row_profile


def round_core(shape=(60, 400, 120), seed=0):
    """Coronal stack of a whole-round core: air -1000, liner 300, sediment 800 (a 6 % step halfway down)."""
    rng = np.random.default_rng(seed)
    nz, nr, nc = shape
    vol = np.full(shape, -1000.0)
    truth = np.zeros(shape, bool)
    for k in range(nz):
        y = (k - nz / 2 + 0.5) * 1.0  # slice offset from the axis, in pixels
        for_r = 40.0
        half_out = np.sqrt(max(for_r**2 - y**2, 0))
        half_in = np.sqrt(max((for_r - 4) ** 2 - y**2, 0))
        cols = np.arange(nc) - nc / 2 + 0.5
        liner = np.abs(cols) < half_out
        sed = np.abs(cols) < half_in
        vol[k][:, liner] = 300.0
        lvl = np.where(np.arange(nr) < nr // 2, 800.0, 848.0)[:, None]
        vol[k][:, sed] = np.broadcast_to(lvl, (nr, sed.sum()))
        truth[k][:, sed] = True
    return vol + rng.normal(0, 15, shape), truth


def split_core(shape=(40, 300, 120), seed=1):
    """Split core: liner trough with an uneven sediment surface; air above the surface (surface depth varies along the core)."""
    rng = np.random.default_rng(seed)
    nz, nr, nc = shape
    vol = np.full(shape, -1000.0)
    truth = np.zeros(shape, bool)
    cols = np.arange(nc)
    for k in range(nz):
        vol[k][:, 20:100] = 300.0  # liner (the trough wall and bottom, seen in this slice)
        surf = 40 + (6 * np.sin(np.arange(nr) / 15.0)).astype(int)  # uneven surface (column index where sediment ends)
        for r in range(nr):
            vol[k][r, 26 : 100 - 6] = np.where(cols[26 : 94] < surf[r] + 30, 800.0, -1000.0)
            truth[k][r, 26 : 100 - 6] = cols[26 : 94] < surf[r] + 30
    return vol + rng.normal(0, 15, shape), truth


DZ, DC, CY, CX, R_OUT, WALL = 2.0, 0.5, 40.0, 30.0, 25.0, 3.0


def tube(split: bool, seed: int = 0, shape=(40, 120, 120)):
    """Coronal stack (2 mm apart, 0.5 mm pixels) of a liner tube along the rows. ``split``: only the lower half of the
    tube (y >= axis) exists and the sediment surface is uneven, as in a split core. Returns (volume, truth)."""
    rng = np.random.default_rng(seed)
    nz, nr, nc = shape
    vol = np.full(shape, -1000.0)
    truth = np.zeros(shape, bool)
    xs = (np.arange(nc) + 0.5) * DC
    for k in range(nz):
        y = (k + 0.5) * DZ
        r = np.hypot(xs - CX, y - CY)
        liner = r < R_OUT
        sed = r < R_OUT - WALL
        if split:
            keep = y >= CY
            surf = CY + 5 + 3 * np.sin(np.arange(nr) / 10.0)[:, None]  # sediment starts at the uneven surface
            liner = liner & keep & ~(r < R_OUT - WALL)  # the half-pipe wall only; above the surface the inside is air
            sed = sed[None, :] & (y >= surf)
            vol[k][:, :] = np.where(liner[None, :], 300.0, -1000.0)
            vol[k][sed] = 800.0
            truth[k] = sed
        else:
            vol[k][:, liner] = 300.0
            vol[k][:, sed] = 800.0
            truth[k][:, sed] = True
    return vol + rng.normal(0, 15, shape), truth


def wobbly_tube(split: bool = False, nr: int = 300, seed: int = 0):
    """Like ``tube`` but the axis drifts along the core: sideways by a few mm (a slow wave) and a steady lean through
    the stack. Returns (volume, truth, cx per row, cy per row)."""
    rng = np.random.default_rng(seed)
    nz, nc = 40, 120
    rows = np.arange(nr)
    cx = CX + 4.0 * np.sin(rows * DC / 40.0)
    cy = CY + 0.02 * rows * DC
    vol = np.full((nz, nr, nc), -1000.0)
    truth = np.zeros(vol.shape, bool)
    xs = (np.arange(nc) + 0.5) * DC
    for k in range(nz):
        y = (k + 0.5) * DZ
        r = np.hypot(xs[None, :] - cx[:, None], y - cy[:, None])  # (rows, cols)
        sed = r < R_OUT - WALL
        liner = (r < R_OUT) & ~sed
        if split:
            below = (y >= cy)[:, None]
            liner &= below
            sed &= below & (y >= cy + 5)[:, None]
        vol[k][liner] = 300.0
        vol[k][sed] = 800.0
        truth[k] = sed
    return vol + rng.normal(0, 15, vol.shape), truth, cx, cy


class VoidAndClastTests(unittest.TestCase):
    @staticmethod
    def image():
        img = np.full((80, 80), -1000.0)
        img[5:75, 5:75] = 800.0
        img[20:26, 20:26] = -1000.0  # an air pocket
        img[50:58, 50:58] = 1600.0  # a dense clast
        return img

    def test_voids_stay_in_or_are_cut_out(self) -> None:
        img = self.image()
        kept = mask_2d(img, 300.0, 0, True)
        cut = mask_2d(img, 300.0, 0, False)
        self.assertTrue(kept[22, 22])
        self.assertFalse(cut[22, 22])
        self.assertTrue(cut[40, 40])

    def test_clasts_are_cut_out_and_not_filled_back(self) -> None:
        img = self.image()
        m = mask_2d(img, 300.0, 0, True, None, 1200.0)
        self.assertFalse(m[53, 53])  # the clast is out although fill_holes is on
        self.assertTrue(m[22, 22])  # the air pocket is still filled in
        self.assertTrue(m[40, 40])
        self.assertEqual(int(mask_2d(img, 300.0, 0, True).sum() - m.sum()), 64)

    def test_margin_also_shrinks_the_cut_outs(self) -> None:
        img = self.image()
        a = mask_2d(img, 300.0, 0, True, None, 1200.0)
        b = mask_2d(img, 300.0, 2, True, None, 1200.0)
        self.assertFalse(b[48, 53])  # two pixels away from the clast edge: the halo is gone too
        self.assertTrue(a[48, 53])

    def test_no_clast_threshold_changes_nothing(self) -> None:
        vol, _ = round_core()
        _, t_sed = auto_thresholds(vol)
        np.testing.assert_array_equal(mask_3d(vol, t_sed, 1, 1.0), mask_3d(vol, t_sed, 1, 1.0, t_clast=None))

    def test_clast_cut_works_with_the_cylinder_and_in_3d(self) -> None:
        vol, truth = tube(False)
        vol[:, 40:60, int(28 / DC) : int(32 / DC)] += np.where(truth[:, 40:60, int(28 / DC) : int(32 / DC)], 900.0, 0.0)  # dense patch
        t_air, t_sed = auto_thresholds(vol)
        circle = (CX, CY, R_OUT - WALL - 0.3)
        a = mask_3d(vol, t_sed, 0, DC, circle=circle, dz_mm=DZ)
        b = mask_3d(vol, t_sed, 0, DC, circle=circle, dz_mm=DZ, t_clast=1200.0)
        self.assertLess(int(b.sum()), int(a.sum()))
        self.assertFalse((b & (vol >= 1200.0)).any())

    def test_auto_clast_threshold_sits_above_the_matrix_below_the_clasts(self) -> None:
        rng = np.random.default_rng(0)
        vol = np.full((20, 100, 100), -1000.0) + rng.normal(0, 10, (20, 100, 100))
        vol[:, 10:90, 10:90] = 800.0 + rng.normal(0, 15, (20, 80, 80))
        vol[5:8, 30:34, 30:34] = 1600.0
        t = auto_clast_threshold(vol, 300.0)
        self.assertTrue(840 < t < 1600, t)


class AxisPathTests(unittest.TestCase):
    def test_path_follows_the_wobble(self) -> None:
        for split in (False, True):
            vol, _, cx, cy = wobbly_tube(split)
            t_air, _ = auto_thresholds(vol)
            fit = fit_axis_path(vol, t_air, DZ, DC, DC, 0, vol.shape[1] - 1)
            self.assertIsNotNone(fit, split)
            cx_r, cy_r, r, n = fit
            self.assertGreater(n, 10)
            self.assertAlmostEqual(r, R_OUT, delta=1.0)
            mid = slice(40, 260)  # away from the ends, where the smoothing window runs off
            self.assertLess(float(np.abs(cx_r[mid] - cx[mid]).max()), 1.2, split)
            self.assertLess(float(np.abs(cy_r[mid] - cy[mid]).max()), 2.0, split)

    def test_following_keeps_sediment_and_a_fixed_circle_does_not(self) -> None:
        vol, truth, cx, cy = wobbly_tube(False)
        t_air, _ = auto_thresholds(vol)
        cx_r, cy_r, r_out, _ = fit_axis_path(vol, t_air, DZ, DC, DC, 0, vol.shape[1] - 1)
        r_in = r_out - WALL - 0.3
        follow = mask_3d(vol, t_air, 0, DC, circle=(cx_r, cy_r, r_in), dz_mm=DZ)
        fixed = mask_3d(vol, t_air, 0, DC, circle=(CX, CY, r_in), dz_mm=DZ)
        liner = (vol > 100) & (vol < 500) & ~truth
        sl = (slice(None), slice(40, 260), slice(None))
        kept = lambda m: (m[sl] & truth[sl]).sum() / truth[sl].sum()
        self.assertGreater(kept(follow), 0.93)
        self.assertLess((follow[sl] & liner[sl]).sum() / follow[sl].sum(), 0.01)
        # the fixed circle either cuts into the sediment or lets the liner in as the core wobbles away from it
        bad = (1 - kept(fixed)) + (fixed[sl] & liner[sl]).sum() / fixed[sl].sum()
        self.assertGreater(bad, 0.10)

    def test_too_few_fits_gives_none(self) -> None:
        vol = np.full((20, 40, 30), -1000.0) + np.random.default_rng(0).normal(0, 5, (20, 40, 30))
        self.assertIsNone(fit_axis_path(vol, -500.0, 2.0, 0.5, 0.5, 0, 39))

    def test_per_row_spans_match_the_fixed_circle_when_the_path_is_constant(self) -> None:
        vol, _ = tube(False)
        a = mask_3d(vol, -100.0, 0, DC, circle=(CX, CY, 21.0), dz_mm=DZ)
        n = vol.shape[1]
        b = mask_3d(vol, -100.0, 0, DC, circle=(np.full(n, CX), np.full(n, CY), 21.0), dz_mm=DZ)
        np.testing.assert_array_equal(a, b)


class CircleTests(unittest.TestCase):
    def test_whole_round_circle_is_found(self) -> None:
        vol, _ = tube(False)
        t_air, _ = auto_thresholds(vol)
        cx, cy, r, n = fit_outer_circle(vol, 60, t_air, DZ, DC)
        self.assertAlmostEqual(cx, CX, delta=0.5)
        self.assertAlmostEqual(cy, CY, delta=0.8)
        self.assertAlmostEqual(r, R_OUT, delta=0.8)
        self.assertGreater(n, 30)

    def test_split_core_circle_is_found_despite_the_flat_face(self) -> None:
        vol, _ = tube(True)
        t_air, _ = auto_thresholds(vol)
        cx, cy, r, _ = fit_outer_circle(vol, 60, t_air, DZ, DC)
        self.assertAlmostEqual(cx, CX, delta=1.0)
        self.assertAlmostEqual(cy, CY, delta=1.5)
        self.assertAlmostEqual(r, R_OUT, delta=1.5)

    def test_no_core_no_circle(self) -> None:
        self.assertIsNone(fit_outer_circle(np.full((20, 30, 40), -1000.0) + np.random.default_rng(0).normal(0, 5, (20, 30, 40)), 15, -500.0, 2.0, 0.5))
        self.assertIsNone(ransac_circle(np.zeros((3, 2))))

    def test_outliers_are_ignored(self) -> None:
        ang = np.linspace(0, 2 * np.pi, 40, endpoint=False)
        pts = np.c_[10 + 5 * np.cos(ang), 20 + 5 * np.sin(ang)]
        pts = np.vstack([pts, np.random.default_rng(0).uniform(0, 30, (12, 2))])
        cx, cy, r, _ = ransac_circle(pts)
        self.assertAlmostEqual(cx, 10, delta=0.2)
        self.assertAlmostEqual(r, 5, delta=0.2)

    def test_span_of_a_slice_through_the_circle(self) -> None:
        circle = (30.0, 40.0, 25.0)
        c0, c1 = circle_span(19, circle, DZ, DC, 120)  # slice centre at 39 mm: nearly through the axis
        self.assertAlmostEqual(c0 * DC, 5.0, delta=1.0)
        self.assertAlmostEqual(c1 * DC, 55.0, delta=1.0)
        self.assertIsNone(circle_span(0, circle, DZ, DC, 120))  # 1 mm from the stack edge, 39 mm from the axis

    def test_cylinder_removes_the_liner_at_the_air_threshold(self) -> None:
        for split in (False, True):
            vol, truth = tube(split)
            t_air, t_sed = auto_thresholds(vol)
            circle = (CX, CY, R_OUT - WALL - 0.2)
            with_c = mask_3d(vol, t_air, 0, DC, circle=circle, dz_mm=DZ)
            without = mask_3d(vol, t_air, 0, DC)
            liner = (vol > 100) & (vol < 500) & ~truth
            self.assertGreater((without & liner).sum(), 0)  # at the air level the liner is in the mask
            self.assertLess((with_c & liner).sum() / max(without.sum(), 1), 0.01)
            self.assertGreater((with_c & truth).sum() / truth.sum(), 0.93, split)  # sediment kept
            self.assertLess((with_c & ~truth).sum() / with_c.sum(), 0.03, split)


class ThresholdTests(unittest.TestCase):
    def test_thresholds_sit_between_the_materials(self) -> None:
        vol, _ = round_core()
        t_air, t_sed = auto_thresholds(vol)
        self.assertTrue(-1000 < t_air < 300)
        self.assertTrue(300 < t_sed < 800, t_sed)


class MaskTests(unittest.TestCase):
    def test_round_core_mask_matches_truth(self) -> None:
        vol, truth = round_core()
        _, t_sed = auto_thresholds(vol)
        m = mask_3d(vol, t_sed, margin_mm=0, pixel_mm=1.0)
        inter = (m & truth).sum()
        self.assertGreater(inter / truth.sum(), 0.97)  # finds the sediment
        self.assertLess((m & ~truth).sum() / m.sum(), 0.03)  # and not the liner

    def test_split_core_follows_uneven_surface(self) -> None:
        vol, truth = split_core()
        _, t_sed = auto_thresholds(vol)
        m = mask_3d(vol, t_sed, margin_mm=0, pixel_mm=1.0, fill_holes=False)
        self.assertGreater((m & truth).sum() / truth.sum(), 0.97)
        self.assertLess((m & ~truth).sum() / m.sum(), 0.03)

    def test_margin_shrinks_mask(self) -> None:
        vol, _ = round_core()
        _, t_sed = auto_thresholds(vol)
        a = mask_3d(vol, t_sed, 0, 1.0)
        b = mask_3d(vol, t_sed, 3, 1.0)
        self.assertLess(b.sum(), a.sum())
        self.assertTrue((b <= a).all())

    def test_biggest_blob_and_fill_holes(self) -> None:
        img = np.full((60, 60), -1000.0)
        img[10:50, 10:50] = 800.0
        img[25:30, 25:30] = -1000.0  # an air pocket inside the sediment
        img[2:5, 2:5] = 800.0  # a stray bright speck
        filled = mask_2d(img, 300.0, 0, True)
        holey = mask_2d(img, 300.0, 0, False)
        self.assertTrue(filled[27, 27] and not holey[27, 27])
        self.assertFalse(filled[3, 3])  # the speck is dropped

    def test_empty_image(self) -> None:
        self.assertFalse(mask_2d(np.full((20, 20), -1000.0), 300.0).any())


class ProfileTests(unittest.TestCase):
    def test_row_profile_recovers_levels_and_step(self) -> None:
        vol, _ = round_core()
        _, t_sed = auto_thresholds(vol)
        m = mask_3d(vol, t_sed, 2, 1.0)
        mean, sd, cnt = row_profile(vol, m)
        n = len(mean)
        self.assertAlmostEqual(float(np.nanmedian(mean[: n // 2 - 10])), 800.0, delta=3)
        self.assertAlmostEqual(float(np.nanmedian(mean[n // 2 + 10 :])), 848.0, delta=3)
        self.assertAlmostEqual(float(np.nanmedian(sd)), 15.0, delta=2)
        self.assertTrue((cnt > 0).all())

    def test_rows_with_too_few_voxels_are_nan(self) -> None:
        vol = np.zeros((3, 4, 5))
        m = np.zeros(vol.shape, bool)
        m[:, 0, :2] = True
        mean, _, cnt = row_profile(vol, m, min_count=20)
        self.assertTrue(np.isnan(mean).all())
        self.assertEqual(int(cnt[0]), 6)


if __name__ == "__main__":
    unittest.main()
