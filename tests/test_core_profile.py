"""Profile engine on synthetic cores with known matrix level, clasts, voids, burrows and layers."""

from __future__ import annotations

import unittest

import numpy as np

from core_profile.classes import ClassParams, Cls, classify
from core_profile.geometry import fit_geometry
from core_profile.levels import estimate_level
from core_profile.recipe import Recipe
from core_profile.reduce import Region, class_fractions, down_core, footprint_smooth
from core_profile.sedct import sedct_profile

N, H = 140, 120
VOX, STEP = 0.5, 1.0
R_OUT, WALL = 50.0, 6.0
C = (59.0, 61.0)  # true axis (row, col)


def make_core(seed: int = 1, layer: bool = True, sigma: float = 0.03):
    rng = np.random.default_rng(seed)
    yy, xx = np.indices((H, H), dtype=np.float32)
    r = np.hypot(yy - C[0], xx - C[1])
    cup = 1 + 0.12 * (r / (R_OUT - WALL)) ** 2  # rim reads higher
    zz = np.arange(N)
    drift = 1 - 0.0004 * np.abs(zz - N // 2)  # slow level change along the core
    base = 1.2 * np.where(layer & (zz >= 70), 1.06, 1.0)  # 6 % step at slice 70
    vol = np.zeros((N, H, H), np.float32)
    inner = r < R_OUT - WALL
    liner = (r >= R_OUT - WALL) & (r < R_OUT)
    for i in range(N):
        v = np.zeros((H, H), np.float32)
        v[inner] = (base[i] * drift[i] * cup)[inner]
        v[liner] = 1.1
        vol[i] = v
    truth = vol.copy()
    zg = zz[:, None, None]

    def ball(cz, cy, cx, rad):
        return ((zg - cz) ** 2 + (yy[None] - cy) ** 2 + (xx[None] - cx) ** 2) <= rad * rad

    clast = ball(35, 50, 55, 8)
    low = ball(100, 65, 70, 6)
    void = ball(40, 70, 50, 3)
    vol[clast & (truth > 0)] *= 1.3
    vol[low & (truth > 0)] *= 0.88
    vol[void] = 0.05
    vol += rng.normal(0, sigma, vol.shape).astype(np.float32)
    return vol, truth, dict(clast=clast, low=low, void=void)


class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.vol, cls.truth, cls.balls = make_core()
        cls.geom = fit_geometry(cls.vol, VOX, core_thr=0.5)
        cls.lev, _ = estimate_level(cls.vol, cls.geom, STEP, shape_cm=8.0, scalar_cm=2.0)
        cls.cls = classify(cls.vol, cls.geom, cls.lev, STEP, ClassParams(void_below=0.5, min_body_vox=40), trim_cm=2.0)

    def test_geometry(self) -> None:
        g = self.geom
        self.assertLess(np.abs(g.center[:, 0] - C[0]).max(), 1.0)
        self.assertLess(np.abs(g.center[:, 1] - C[1]).max(), 1.0)
        self.assertLess(np.abs(np.median(g.r_out) - R_OUT), 1.5)
        # sediment radius: true 44 px minus the 1.1 mm (2.2 px) safety margin
        self.assertLess(abs(np.median(g.r_in) - (R_OUT - WALL - 1.1 / VOX)), 1.5)

    def test_fixed_inner_radius(self) -> None:
        g = fit_geometry(self.vol[::4], VOX, core_thr=0.5, r_in_mm=(R_OUT - WALL) * VOX)
        self.assertLess(abs(np.median(g.r_in) - (R_OUT - WALL - 1.1 / VOX)), 1.0)

    def test_level_follows_cupping_and_drift(self) -> None:
        i = 20
        m = self.geom.inside(i, (H, H), 3.0) & ~self.balls["clast"][i] & ~self.balls["void"][i]
        err = np.abs(self.lev.level[i][m] / self.truth[i][m] - 1)
        self.assertLess(np.median(err), 0.015)

    def test_clast_found(self) -> None:
        b = self.balls["clast"]
        self.assertGreater((self.cls[b] == Cls.CLAST).mean(), 0.9)

    def test_void_found(self) -> None:
        b = self.balls["void"]
        self.assertGreater((self.cls[b] == Cls.VOID).mean(), 0.95)

    def test_low_density_is_below_matrix_not_above(self) -> None:
        # regression guard: the "low-density" class must be the low side
        b = self.balls["low"]
        self.assertGreater((self.cls[b] == Cls.LOWDENS).mean(), 0.5)
        m = self.cls == Cls.LOWDENS
        self.assertLess(float((self.vol[m] - self.lev.level[m]).mean()), 0)
        d = self.cls == Cls.DENSE
        if d.any():
            self.assertGreater(float((self.vol[d] - self.lev.level[d]).mean()), 0)

    def test_few_false_positives(self) -> None:
        far = np.ones(self.cls.shape, bool)
        for k in self.balls.values():
            from scipy import ndimage as ndi

            far &= ~ndi.binary_dilation(k, iterations=8)
        ins = np.stack([self.geom.inside(i, (H, H)) for i in range(N)])
        far &= ins
        self.assertLess(((self.cls != 0) & far).mean(), 0.01)

    def test_matrix_profile_ignores_clast_and_recovers_layer(self) -> None:
        pm = down_core(self.vol, self.cls, self.geom, STEP, Region("central", 0.8))
        pa = down_core(self.vol, None, self.geom, STEP, Region("central", 0.8))
        i = 35  # slice through the clast
        t = self.truth[i][Region("central", 0.8).mask(i, (H, H), self.geom)].mean()
        self.assertLess(abs(pm.mean[i] / t - 1), 0.01)
        self.assertGreater(pa.mean[i] / t - 1, 0.01)  # plain mean is pulled up by the clast
        step = np.nanmean(pm.mean[80:130]) / np.nanmean(pm.mean[10:60])
        self.assertAlmostEqual(step, 1.06, delta=0.025)

    def test_band_and_whole_regions(self) -> None:
        pb = down_core(self.vol, self.cls, self.geom, STEP, Region("band", thickness_mm=2.0))
        pw = down_core(self.vol, self.cls, self.geom, STEP, Region("whole"))
        self.assertTrue(np.isfinite(pb.mean).all())
        self.assertLess(pb.count.max(), pw.count.max() / 5)
        self.assertGreater(np.nanmean(pw.mean), np.nanmean(pb.mean))  # rim reads higher than the centre (cupping)

    def test_class_fractions_sum_to_one(self) -> None:
        f = class_fractions(self.cls, self.geom)
        tot = sum(f.values())
        self.assertTrue(np.allclose(tot, 1.0))


class SmallTests(unittest.TestCase):
    def test_recipe_roundtrip(self) -> None:
        r = Recipe(region=Region("band", thickness_mm=2.0), use=(Cls.MATRIX, Cls.LOWDENS), core_thr=-300, zero_offset=1000, footprint_fwhm_mm=10, note="x")
        r2 = Recipe.from_json(r.to_json())
        self.assertEqual(r, r2)
        with self.assertRaises(ValueError):
            Recipe.from_json('{"schema": "other"}')

    def test_footprint_smooth_nan_aware(self) -> None:
        z = np.arange(100) * 0.1
        y = np.where(z > 5, 2.0, 1.0)
        y[40:43] = np.nan
        s = footprint_smooth(z, y, fwhm_mm=10)
        self.assertTrue(np.isfinite(s[41]))
        self.assertAlmostEqual(float(s[10]), 1.0, delta=0.02)
        self.assertAlmostEqual(float(s[90]), 2.0, delta=0.02)
        self.assertAlmostEqual(float(s[50]), 1.5, delta=0.1)

    def test_sedct_plateau(self) -> None:
        rng = np.random.default_rng(0)
        # a plateau of ~1000 +- 20 plus a dense tail and a few low values
        y = np.concatenate([rng.normal(1000, 20, 4600), rng.uniform(1400, 3000, 300), rng.uniform(10, 300, 100)])
        Y = np.tile(y[:, None], (1, 6)) + rng.normal(0, 1, (5000, 6))
        gs, sd, px = sedct_profile(Y, backv=0, maxslope=60, trim=100)
        self.assertTrue(np.all(np.abs(gs - 1000) < 15))
        self.assertTrue(np.all(px > 3000))


if __name__ == "__main__":
    unittest.main()
