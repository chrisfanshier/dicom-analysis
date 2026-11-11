"""Tests for core_stitch.coronal_mpr: synthetic volumes with known answers (no DICOM files)."""

from __future__ import annotations

import math
import unittest

import numpy as np
import SimpleITK as sitk

from scipy import ndimage as ndi

from core_stitch.coronal_mpr import (
    AxialGeometry,
    AxialStack,
    CoronalSeries,
    PlaneRenderer,
    across_core_edge,
    core_mask,
    difference_stats,
    edge_width,
    pick_plane,
    plane_window,
)


def make_stack(arr: np.ndarray, spacing=(0.5, 0.5, 0.5), origin=(0.0, 0.0, 0.0)) -> AxialStack:
    img = sitk.GetImageFromArray(arr.astype(np.float32))
    img.SetSpacing(spacing)
    img.SetOrigin(origin)
    geom = AxialGeometry(paths=[], z=np.arange(arr.shape[0]) * spacing[2] + origin[2], x0=origin[0], y0=origin[1],
                         px_x=spacing[0], px_y=spacing[1], ncols=arr.shape[2], nrows=arr.shape[1], thickness=spacing[2],
                         description="synthetic")
    return AxialStack(image=img, ones=sitk.Cast(img * 0 + 1, sitk.sitkFloat32), geom=geom, rows=(0, arr.shape[1]))


def make_cor(r, c, ipps, px, rows, cols) -> CoronalSeries:
    r, c = np.asarray(r, float), np.asarray(c, float)
    return CoronalSeries(folder=None, paths=[None] * len(ipps), ipps=np.asarray(ipps, float), r=r, c=c, n=np.cross(r, c),
                         px=px, rows=rows, cols=cols, thickness=2.0, description="synthetic")


class RenderTests(unittest.TestCase):
    def test_axis_aligned_plane_is_the_weighted_row_average(self) -> None:
        rng = np.random.default_rng(1)
        nz, ny, nx = 40, 30, 50
        arr = rng.normal(100, 20, (nz, ny, nx)).astype(np.float32)
        stack = make_stack(arr)
        # coronal plane at y = 5.0 mm (axial row 10); columns run along +x, rows run down the core (-z)
        cor = make_cor([1, 0, 0], [0, 0, -1], [[0.0, 5.0, (nz - 1) * 0.5]], px=0.5, rows=nz, cols=nx)
        out, valid = PlaneRenderer(stack, cor, 0, nx).render_plane(0, thickness=2.0)
        self.assertTrue(valid.all())
        # a 2 mm slab centred on y = 5 spans y = 4..6; the mean of the linear interpolant over it weights the rows
        # at y = 4, 4.5, 5, 5.5, 6 (axial rows 8..12) by 1/8, 1/4, 1/4, 1/4, 1/8
        w = np.array([0.125, 0.25, 0.25, 0.25, 0.125])
        want = np.einsum("k,zkx->zx", w, arr[:, 8:13, :])[::-1]  # row i of the plane is z index nz-1-i
        np.testing.assert_allclose(out, want, atol=2e-3)

    def test_oblique_plane_uses_the_full_orientation_and_pixel_centres(self) -> None:
        # a linear volume: any slab mean along the normal equals the value at the slab centre, so the answer is exact
        nz, ny, nx = 100, 120, 120
        sp, origin = 0.5, (-30.0, -30.0, 0.0)
        a, b, c0, d = 3.0, -2.0, 5.0, 1000.0
        zz, yy, xx = np.meshgrid(np.arange(nz) * sp + origin[2], np.arange(ny) * sp + origin[1], np.arange(nx) * sp + origin[0],
                                 indexing="ij")
        stack = make_stack(a * xx + b * yy + c0 * zz + d, (sp, sp, sp), origin)
        # rotate the plane 8.7 deg about the core axis (z) and 1.2 deg within itself, like a tilted scanner MPR
        t1, t2 = math.radians(8.7), math.radians(1.2)
        r = np.array([math.cos(t1), -math.sin(t1), 0.0])
        c = np.array([0.0, 0.0, -1.0])
        r2 = r * math.cos(t2) + c * math.sin(t2)
        c2 = c * math.cos(t2) - r * math.sin(t2)
        r2, c2 = r2 / np.linalg.norm(r2), c2 / np.linalg.norm(c2)
        ipp = np.array([-12.0, 6.0, 40.0])
        px, rows, cols = 0.4, 50, 70
        cor = make_cor(r2, c2, [ipp], px, rows, cols)
        out, valid = PlaneRenderer(stack, cor, 0, cols).render_plane(0, thickness=2.0)
        self.assertTrue(valid.all())
        ii, jj = np.meshgrid(np.arange(rows), np.arange(cols), indexing="ij")
        pts = ipp[None, None, :] + r2[None, None, :] * (jj * px)[..., None] + c2[None, None, :] * (ii * px)[..., None]
        want = a * pts[..., 0] + b * pts[..., 1] + c0 * pts[..., 2] + d
        np.testing.assert_allclose(out, want, atol=5e-3)

    def test_pixels_outside_the_volume_are_invalid(self) -> None:
        arr = np.full((20, 20, 20), 50.0, np.float32)
        stack = make_stack(arr)
        cor = make_cor([1, 0, 0], [0, 0, -1], [[0.0, 5.0, 14.0]], px=0.5, rows=40, cols=30)  # reaches above and beside the volume
        out, valid = PlaneRenderer(stack, cor, 0, 30).render_plane(0, thickness=2.0)
        self.assertFalse(valid.all())
        self.assertTrue(valid[10, 5])
        self.assertTrue(np.isnan(out[~valid]).all())
        np.testing.assert_allclose(out[valid], 50.0, atol=1e-4)

    def test_mip_is_not_the_mean(self) -> None:
        rng = np.random.default_rng(2)
        stack = make_stack(rng.normal(0, 30, (20, 20, 20)))
        cor = make_cor([1, 0, 0], [0, 0, -1], [[0.0, 5.0, 9.0]], px=0.5, rows=15, cols=15)
        r = PlaneRenderer(stack, cor, 0, 15)
        mean, _ = r.render_plane(0, thickness=2.0)
        mx, _ = r.render_plane(0, thickness=2.0, mode="max")
        self.assertTrue((mx >= mean - 1e-4).all())
        self.assertGreater(float((mx - mean).mean()), 5.0)


class WindowAndMeasureTests(unittest.TestCase):
    def test_plane_window_keeps_only_columns_inside_the_field_of_view(self) -> None:
        geom = AxialGeometry(paths=[], z=np.arange(10) * 0.5, x0=-10.0, y0=-10.0, px_x=0.5, px_y=0.5, ncols=40, nrows=40,
                             thickness=0.5, description="")
        cor = make_cor([1, 0, 0], [0, 0, -1], [[-50.0, 0.0, 5.0], [-50.0, 2.0, 5.0]], px=0.5, rows=10, cols=200)
        j_lo, j_hi, y0, y1 = plane_window(cor, geom)
        # columns at x = -50 + 0.5 j; the field of view is x in [-10, 9.5] (+- 3 mm margin)
        self.assertAlmostEqual(-50 + 0.5 * j_lo, -13.0, delta=0.6)
        self.assertAlmostEqual(-50 + 0.5 * (j_hi - 1), 12.5, delta=0.6)
        self.assertLess(y0, 0.0)
        self.assertGreater(y1, 2.0)

    def test_difference_stats_and_core_mask(self) -> None:
        his = np.full((200, 60), -1000.0)
        his[:, 15:45] = 900.0
        valid = np.ones_like(his, bool)
        core, rows = core_mask(his, valid, 0.5)
        self.assertTrue(core[100, 30])
        self.assertFalse(core[100, 5])
        self.assertFalse(core[2, 30])  # the rows at the end of the coverage are trimmed
        ours = his + 1.0
        s = difference_stats(ours, his, core)
        self.assertAlmostEqual(s["mean"], 1.0, places=6)
        self.assertAlmostEqual(s["sd"], 0.0, places=6)

    def test_edge_width_of_a_known_ramp(self) -> None:
        offs = np.arange(-8, 8.001, 0.25)
        prof = np.clip((offs + 2.0) / 4.0, 0, 1)  # linear ramp from -2 to +2 px: 10-90 % width is 0.8 * 4 = 3.2 px
        self.assertAlmostEqual(edge_width(prof, offs), 3.2, places=2)

    def test_pick_plane_takes_the_middle_of_the_good_planes(self) -> None:
        self.assertEqual(pick_plane([0, 10, 100, 100, 100, 100, 100, 20]), 4)


class AcrossCoreEdgeTests(unittest.TestCase):
    def _core_with_void(self):
        his = np.full((300, 60), -1000.0)
        his[:, 12:48] = 900.0
        his[140:180, 20:40] = -1000.0  # a void inside the core: two edges that run across the core
        his = ndi.gaussian_filter(his, 1.2)
        valid = np.ones_like(his, bool)
        core, rows = core_mask(his, valid, 0.5)
        return his, core

    def test_finds_a_void_edge_and_measures_it_in_both_images(self) -> None:
        his, core = self._core_with_void()
        soft = ndi.gaussian_filter(his, 1.0)  # a blurrier copy
        prof, info = across_core_edge({"his": his, "soft": soft, "same": his.copy()}, his, core, (60, 260))
        self.assertIsNotNone(prof)
        self.assertGreaterEqual(len(info["cols"]), 8)
        self.assertTrue(abs(info["row"] - 140) <= 5 or abs(info["row"] - 180) <= 5)
        from core_stitch.coronal_mpr import edge_width
        w_his, w_same, w_soft = (edge_width(prof[k].mean(axis=0)) for k in ("his", "same", "soft"))
        self.assertAlmostEqual(w_his, w_same, places=6)
        self.assertGreater(w_soft, w_his + 0.3)  # the blurrier image has the wider edge

    def test_a_featureless_segment_has_no_edge(self) -> None:
        his = np.full((300, 60), -1000.0)
        his[:, 12:48] = 900.0
        core, _rows = core_mask(his, np.ones_like(his, bool), 0.5)
        prof, info = across_core_edge({"his": his}, his, core, (60, 260))
        self.assertIsNone(prof)


if __name__ == "__main__":
    unittest.main()
