"""Loader, CLI and (if PySide6 is installed) the window, on a small synthetic core."""

from __future__ import annotations

import csv
import json
import os
import tempfile
import unittest
from pathlib import Path

import numpy as np
import tifffile

from core_profile.cli import main as cli_main
from core_profile.loaders import Volume, is_tiff_folder, load_axial_folder, load_dicom_mpr, load_npz, save_npz
from core_profile.recipe import Recipe
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, generate_uid
from test_core_profile import STEP, VOX, make_core

def write_tiffs(folder: Path, vol: np.ndarray, slope: float = 1e-4, offset: float = -0.5) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    for i, sl in enumerate(vol):
        counts = np.clip(np.round((sl - offset) / slope), 0, 65535).astype(np.uint16)
        tifffile.imwrite(folder / f"s{i:04d}.tif", counts, description=f"slope = {slope:.3E} offset = {offset:.3E}")


def write_mpr_cylinder(folder: Path, px: float = 0.4, dz: float = 2.0, length_mm: float = 120.0, r_out: float = 30.0, wall: float = 3.0, wobble_mm: float = 0.0) -> None:
    """Coronal MPR of a straight cylinder (HU): air -1000, liner 300, sediment 800 with a 6 % step halfway down."""
    folder.mkdir(parents=True, exist_ok=True)
    uid, study = generate_uid(), generate_uid()
    rows, cols = int(length_mm / px), int(2 * (r_out + 6) / px)
    xs = (np.arange(cols) - cols / 2 + 0.5) * px
    along = np.arange(rows)[:, None] * px
    shift = wobble_mm * np.sin(along[:, 0] / 40.0)  # the axis drifts sideways along the core
    for i, y in enumerate(np.arange(-(r_out + 4), r_out + 4.01, dz)):
        r = np.hypot(xs[None, :] - shift[:, None], y)
        img = np.full((rows, cols), -1000.0)
        liner = r < r_out
        img[np.broadcast_to(liner, img.shape)] = 300.0
        sed = np.broadcast_to(r < r_out - wall, img.shape)
        lvl = np.where(along < length_mm / 2, 800.0, 848.0) * np.ones((1, cols))
        img[sed] = lvl[sed]
        meta = FileMetaDataset()
        meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.2"
        meta.MediaStorageSOPInstanceUID = generate_uid()
        meta.TransferSyntaxUID = ExplicitVRLittleEndian
        ds = FileDataset(str(folder / f"i{i:03d}.dcm"), {}, file_meta=meta, preamble=b"\0" * 128)
        ds.SOPClassUID, ds.SOPInstanceUID = meta.MediaStorageSOPClassUID, meta.MediaStorageSOPInstanceUID
        ds.StudyInstanceUID, ds.SeriesInstanceUID, ds.SeriesDescription, ds.Modality = study, uid, "Cor", "CT"
        ds.InstanceNumber = i + 1
        ds.ImageOrientationPatient = [1, 0, 0, 0, 0, -1]
        ds.ImagePositionPatient = [0, float(i * dz), 0]
        ds.PixelSpacing = [px, px]
        ds.Rows, ds.Columns, ds.SamplesPerPixel = rows, cols, 1
        ds.PhotometricInterpretation = "MONOCHROME2"
        ds.BitsAllocated = ds.BitsStored = 16
        ds.HighBit, ds.PixelRepresentation = 15, 1
        ds.RescaleSlope, ds.RescaleIntercept = 1, 0
        img = img + np.random.default_rng(i).normal(0, 15, img.shape)  # real scans are noisy; a noiseless air/core image breaks Otsu
        ds.PixelData = img.astype(np.int16).tobytes()
        ds.is_little_endian, ds.is_implicit_VR = True, False
        ds.save_as(folder / f"i{i:03d}.dcm")


class LoaderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.vol, *_ = make_core()

    def test_slope_offset_applied_per_file(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            write_tiffs(Path(d), self.vol[:6])
            v = load_axial_folder(d, VOX, STEP)
            self.assertEqual(v.data.shape, (6,) + self.vol.shape[1:])
            self.assertLess(float(np.abs(v.data - self.vol[:6]).max()), 2e-4)

    def test_binning_and_every_nth(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            write_tiffs(Path(d), self.vol[:10])
            v = load_axial_folder(d, VOX, STEP, z_every=2, binning=2)
            self.assertEqual(v.data.shape, (5, 60, 60))
            self.assertAlmostEqual(v.voxel_mm, VOX * 2)
            self.assertAlmostEqual(v.step_mm, STEP * 2)

    def test_empty_folder(self) -> None:
        with tempfile.TemporaryDirectory() as d, self.assertRaises(FileNotFoundError):
            load_axial_folder(d, VOX)

    def test_npz_roundtrip(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            v = Volume(self.vol[:4].copy(), 0.5, 1.0, "u", "src")
            save_npz(v, Path(d) / "v.npz")
            w = load_npz(Path(d) / "v.npz")
            self.assertEqual((w.voxel_mm, w.step_mm, w.unit), (0.5, 1.0, "u"))
            self.assertLess(float(np.abs(w.data - v.data).max()), 2e-3)  # float16 storage


class DicomTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.tmp = tempfile.TemporaryDirectory()
        cls.dir = Path(cls.tmp.name) / "mpr"
        write_mpr_cylinder(cls.dir)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.tmp.cleanup()

    def test_axial_volume_from_coronal_stack(self) -> None:
        self.assertFalse(is_tiff_folder(self.dir))
        v = load_dicom_mpr(self.dir, target_mm=0.8)
        self.assertAlmostEqual(v.voxel_mm, 0.8)
        self.assertAlmostEqual(v.step_mm, 0.8)
        self.assertEqual(v.unit, "HU")
        n, h, w = v.data.shape
        self.assertAlmostEqual(n * 0.8, 120.0, delta=1.5)  # along the core
        self.assertAlmostEqual(h * 0.8, 2 * 34, delta=4)  # through the stack (interpolated to 0.8 mm)
        mid = v.data[n // 4, h // 2, w // 2]  # core centre, upper half
        self.assertAlmostEqual(float(mid), 800.0, delta=30)  # one noisy voxel (noise sd 15 HU before binning)

    def test_pipeline_on_dicom_with_hu_preset(self) -> None:
        from core_profile.cli import main

        with tempfile.TemporaryDirectory() as d:
            rc = main([str(self.dir), "--out", d, "--region", "central", "--frac", "0.6"])
            self.assertEqual(rc, 0)
            with open(Path(d) / "profile.csv", newline="") as fh:
                rows = list(csv.DictReader(fh))
            m = np.array([float(r["mean"]) for r in rows])
            n = len(m)
            self.assertAlmostEqual(float(np.nanmedian(m[n // 10 : 2 * n // 5])), 800.0, delta=15)
            self.assertAlmostEqual(float(np.nanmedian(m[3 * n // 5 : 9 * n // 10])), 848.0, delta=15)
            rec = Recipe.from_json((Path(d) / "recipe.json").read_text())
            self.assertEqual((rec.core_thr, rec.zero_offset), (-300.0, 1000.0))


class CliTests(unittest.TestCase):
    def test_cli_makes_profile_and_recipe(self) -> None:
        vol, *_ = make_core()
        with tempfile.TemporaryDirectory() as d:
            write_tiffs(Path(d) / "tif", vol)
            rc = cli_main([str(Path(d) / "tif"), "--out", str(Path(d) / "out"), "--voxel-mm", str(VOX), "--slice-mm", str(STEP), "--core-thr", "0.5", "--void-below", "0.5", "--region", "central", "--frac", "0.8"])
            self.assertEqual(rc, 0)
            with open(Path(d) / "out" / "profile.csv", newline="") as fh:
                rows = list(csv.DictReader(fh))
            self.assertEqual(len(rows), len(vol))
            self.assertIn("frac_clast", rows[0])
            mid = float(rows[100]["mean"])
            self.assertAlmostEqual(mid, 1.2 * 1.06, delta=0.1)
            rec = Recipe.from_json((Path(d) / "out" / "recipe.json").read_text())
            self.assertEqual(rec.region.kind, "central")
            self.assertIn("source=tif", rec.note)

    def test_cli_needs_voxel_size_for_tiff_folder(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            write_tiffs(Path(d) / "t", make_core()[0][:3])
            self.assertEqual(cli_main([str(Path(d) / "t"), "--out", d]), 2)

    def test_cli_empty_folder_is_a_clean_error(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            self.assertEqual(cli_main([d, "--out", d]), 2)

    def test_cli_rejects_unknown_class(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            v = Volume(make_core()[0][:30], VOX, STEP)
            save_npz(v, Path(d) / "v.npz")
            self.assertEqual(cli_main([str(Path(d) / "v.npz"), "--out", d, "--use", "nonsense"]), 2)


if __name__ == "__main__":
    unittest.main()
