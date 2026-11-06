"""Phantom tests for the SimpleITK header-geometry engine.

A synthetic core with a known analytic HU value at every physical point is sampled onto two
DICOM series (different pixel sizes, overlapping) whose headers place them correctly.  The
engine's output is checked against that truth, so pixel-centre / orientation mistakes show up.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, generate_uid

from core_stitch.itk_stitch import image_based_move_mm, load_half, stitch_pair

IOP = [1, 0, 0, 0, 0, -1]  # rows run down the core (-z), columns across (+x), slices +y
RNG = np.random.default_rng(7)
BLOBS = [(RNG.uniform(-18, 18), RNG.uniform(-18, 18), RNG.uniform(-95, 95), RNG.uniform(250, 450)) for _ in range(90)]


def phantom(x, y, z):
    """Analytic HU: air -1024; a 30 mm radius cylinder along z with smooth edges and smooth texture."""
    r = np.sqrt(x**2 + y**2)
    body = 0.5 * (1 - np.tanh((r - 30) / 1.5))
    ends = 0.5 * (1 + np.tanh((z + 105) / 1.5)) * 0.5 * (1 - np.tanh((z - 105) / 1.5))
    tex = np.zeros_like(x, dtype=np.float64)
    for bx, by, bz, amp in BLOBS:
        tex += amp * np.exp(-((x - bx) ** 2 + (y - by) ** 2 + (z - bz) ** 2) / (2 * 4.0**2))
    return -1024 + body * ends * (1900 + tex)


def sample(origin, px, nx, ny, nz, dz, shift=(0.0, 0.0, 0.0)):
    """Sample the phantom on a series lattice: columns +x, rows -z, slices +y."""
    i = np.arange(nx)[None, None, :]
    j = np.arange(ny)[None, :, None]
    k = np.arange(nz)[:, None, None]
    x = origin[0] + i * px + shift[0]
    z = origin[2] - j * px + shift[2]
    y = origin[1] + k * dz + shift[1]
    x, y, z = np.broadcast_arrays(x, y, z)
    return x, y, z, np.rint(phantom(x, y, z)).astype(np.int16)


def write_dicom(folder: Path, uid: str, vol: np.ndarray, origin, px: float, dz: float) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    study = generate_uid()
    for k in range(vol.shape[0]):
        meta = FileMetaDataset()
        meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.2"
        meta.MediaStorageSOPInstanceUID = generate_uid()
        meta.TransferSyntaxUID = ExplicitVRLittleEndian
        ds = FileDataset(str(folder / f"{k:03d}.dcm"), {}, file_meta=meta, preamble=b"\0" * 128)
        ds.SOPClassUID = meta.MediaStorageSOPClassUID
        ds.SOPInstanceUID = meta.MediaStorageSOPInstanceUID
        ds.StudyInstanceUID = study
        ds.SeriesInstanceUID = uid
        ds.Modality = "CT"
        ds.InstanceNumber = k + 1
        ds.ImageOrientationPatient = IOP
        ds.ImagePositionPatient = [origin[0], origin[1] + k * dz, origin[2]]
        ds.PixelSpacing = [px, px]
        ds.SliceThickness = dz
        ds.Rows, ds.Columns = vol.shape[1:]
        ds.SamplesPerPixel = 1
        ds.PhotometricInterpretation = "MONOCHROME2"
        ds.BitsAllocated = ds.BitsStored = 16
        ds.HighBit = 15
        ds.PixelRepresentation = 1
        ds.RescaleSlope, ds.RescaleIntercept = 1, 0
        ds.PixelData = vol[k].tobytes()
        ds.is_little_endian, ds.is_implicit_VR = True, False
        ds.save_as(folder / f"{k:03d}.dcm", write_like_original=False)


class Rig:
    """Two overlapping halves of the phantom written to disk."""

    def __init__(self, tmp: Path, px_top=0.5, px_btm=0.4, btm_offset=(0.0, 0.0, 0.0), btm_header_error_z=0.0,
                 btm_header_error_x=0.0, btm_header_error_y=0.0):
        self.dz = 2.0
        self.nz = 16
        self.top_origin = (-45.0, -15.0, 100.0)
        n_top = int(round(120 / px_top))  # z from +100 to -20
        self.btm_origin = (-45.0 + btm_offset[0], -15.0 + btm_offset[1], 10.0 + btm_offset[2])
        n_btm = int(round(110 / px_btm))  # z from +10 to -100
        nx_top, nx_btm = int(round(90 / px_top)), int(round(90 / px_btm))
        *_, self.top = sample(self.top_origin, px_top, nx_top, n_top, self.nz, self.dz)
        # bottom data come from the true phantom at its true origin; the *header* may be wrong
        *_, self.btm = sample(self.btm_origin, px_btm, nx_btm, n_btm, self.nz, self.dz)
        hdr_btm = (self.btm_origin[0] + btm_header_error_x, self.btm_origin[1] + btm_header_error_y,
                   self.btm_origin[2] + btm_header_error_z)
        self.px_top, self.px_btm = px_top, px_btm
        self.top_dir, self.btm_dir = tmp / "top", tmp / "btm"
        write_dicom(self.top_dir, "1.2.3.4", self.top, self.top_origin, px_top, self.dz)
        write_dicom(self.btm_dir, "1.2.3.5", self.btm, hdr_btm, px_btm, self.dz)

    def run(self, **kw):
        return stitch_pair(self.top_dir, "1.2.3.4", self.btm_dir, "1.2.3.5", **kw)


def grid_points(grid):
    """Physical (x, y, z) of every voxel of a SimpleITK grid, each shaped (slices, rows, cols)."""
    nx, ny, nz = grid.GetSize()
    k, j, i = np.meshgrid(np.arange(nz), np.arange(ny), np.arange(nx), indexing="ij")
    idx = np.stack([i, j, k], -1) * np.asarray(grid.GetSpacing())
    d = np.asarray(grid.GetDirection()).reshape(3, 3)
    pts = np.asarray(grid.GetOrigin()) + idx @ d.T
    return pts[..., 0], pts[..., 1], pts[..., 2]


def truth_on_grid(grid, dx_mm=0.0):
    x, y, z = grid_points(grid)
    return phantom(x + dx_mm, y, z)


class ItkStitchPhantomTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_finer_half_is_reference_and_copied_exactly(self) -> None:
        rig = Rig(self.dir)
        res = rig.run()
        self.assertEqual(res.info["reference"], "bottom")  # 0.4 mm is the finer pixel
        self.assertAlmostEqual(res.info["upsample_factor_moving"], 0.5 / 0.4 * 1.0, places=3)
        # reference voxels are the original voxels: compare to the data written to disk
        self.assertEqual(int(res.ref_footprint.sum()), rig.btm.size)
        vals = np.sort(res.ref_vol[res.ref_footprint])
        self.assertTrue(np.array_equal(vals, np.sort(rig.btm.ravel().astype(np.float32))))

    def test_moving_half_matches_analytic_truth(self) -> None:
        """Pixel-centre convention check: error must be small, and a half-pixel error must show."""
        rig = Rig(self.dir)
        res = rig.run()
        truth = truth_on_grid(res.grid)
        sel = (res.mov_footprint & ~res.ref_footprint) & (truth > 600)  # moving-only, inside the core
        self.assertGreater(int(sel.sum()), 1000)
        err = float(np.abs(res.mov_vol - truth)[sel].mean())
        self.assertLess(err, 3.0)
        # the same comparison against a lattice shifted half a top pixel must be clearly worse
        shifted = float(np.abs(res.mov_vol - truth_on_grid(res.grid, 0.25))[sel].mean())
        self.assertGreater(shifted, 5.0 * max(err, 0.1))

    def test_equal_pixel_size_policy(self) -> None:
        rig = Rig(self.dir, px_top=0.5, px_btm=0.5, btm_offset=(0.15, 0.0, 0.0))  # 0.3 px off the lattice
        whole = rig.run(placement="whole_pixel")
        self.assertTrue(whole.info["same_lattice"])
        self.assertTrue(whole.info["snapped_to_pixel"])
        self.assertLessEqual(whole.info["snap_residual_mm"], 0.25 + 1e-6)  # at most half a pixel
        original = set(np.unique(rig.btm).tolist()) | set(np.unique(rig.top).tolist())
        got = set(np.unique(whole.mov_vol[whole.mov_footprint]).astype(int).tolist())
        self.assertTrue(got <= original, "whole-pixel placement must only contain original voxel values")
        sub = rig.run(placement="subpixel")
        self.assertFalse(sub.info["snapped_to_pixel"])
        new = set(np.unique(np.rint(sub.mov_vol[sub.mov_footprint])).astype(int).tolist())
        self.assertFalse(new <= original)

    def test_move_mm_corrects_a_wrong_header(self) -> None:
        rig = Rig(self.dir, btm_header_error_z=6.0)  # header puts the bottom half 6 mm too high
        bad = rig.run()

        def overlap_ncc(res):
            m = res.ref_core & res.mov_core
            a, b = res.ref_vol[m], res.mov_vol[m]
            return float(np.corrcoef(a, b)[0, 1])

        fixed = rig.run(move_mm=(6.0, 0.0, 0.0))  # +6 mm along the core = down
        self.assertLess(overlap_ncc(bad), 0.9)
        self.assertGreater(overlap_ncc(fixed), 0.99)

    def test_overlap_is_reported_and_fused_between_the_inputs(self) -> None:
        rig = Rig(self.dir)
        res = rig.run()
        self.assertAlmostEqual(res.info["overlap_mm"], 30.0, delta=1.5)  # z -20..+10
        both = res.ref_core & res.mov_core
        lo = np.minimum(res.ref_vol, res.mov_vol)[both]
        hi = np.maximum(res.ref_vol, res.mov_vol)[both]
        out = res.volume[both]
        self.assertTrue(np.all(out >= lo - 1e-3) and np.all(out <= hi + 1e-3))
        # no half-value band: the row means of the core never dip far below their typical value
        mid = res.volume.shape[0] // 2
        core = res.ref_core[mid] | res.mov_core[mid]
        rows = [res.volume[mid][r][core[r]].mean() for r in range(core.shape[0]) if core[r].sum() > 20]
        self.assertGreater(min(rows), 0.8 * float(np.median(rows)))

    def test_image_based_fallback_ignores_the_header(self) -> None:
        """The fallback reproduces the image-based placement whatever the header says.

        (It is exactly as accurate as the old mask + profile alignment, which is known to be
        unreliable; this test checks the conversion into a header-relative shift, not that accuracy.)
        """
        starts = []
        for err in (0.0, 6.0):
            rig = Rig(self.dir / f"e{int(err)}", btm_header_error_z=err)
            along, across, depth = image_based_move_mm(rig.top_dir, "1.2.3.4", rig.btm_dir, "1.2.3.5")
            header_along = 90.0 - err  # bottom half's first row is 90 mm below the top's, less the header error
            starts.append(header_along + along)
            self.assertEqual(depth, 0.0)
            self.assertAlmostEqual(across, 0.0, delta=1.0)
        self.assertAlmostEqual(starts[0], starts[1], delta=0.6)

    def test_orientation_mismatch_is_refused(self) -> None:
        rig = Rig(self.dir)
        from core_stitch import itk_stitch

        top = load_half("top", rig.top_dir, "1.2.3.4")
        btm = load_half("bottom", rig.btm_dir, "1.2.3.5")
        btm.col_dir = np.array([0.0, 0.1, -0.995])
        with self.assertRaises(ValueError):
            itk_stitch.stitch_pair(None, None, None, None, halves=(top, btm))


if __name__ == "__main__":
    unittest.main()
