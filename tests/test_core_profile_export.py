"""Profile CSV: commented header with the input and every choice, then one line per row between the core ends."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from core_profile.export import write_profile_csv


def read(path: Path):
    lines = path.read_text(encoding="utf-8").splitlines()
    head = [ln for ln in lines if ln.startswith("#")]
    body = [ln for ln in lines if not ln.startswith("#")]
    return head, body[0].split(","), [ln.split(",") for ln in body[1:]]


class ExportTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "p.csv"
        self.mean = np.linspace(800, 850, 100)
        self.sd = np.full(100, 15.0)
        self.cnt = np.full(100, 500)
        self.meta = {"input": {"folder": "C:/scans/SE1", "units": "HU"}, "mask": {"sediment_threshold_hu": 364.0, "remove_clasts": False, "clast_threshold_hu": None}}

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_rows_between_the_core_ends_and_values(self) -> None:
        n = write_profile_csv(self.path, self.mean, self.sd, self.cnt, 1.0, 10.0, 90.0, self.meta)
        head, cols, rows = read(self.path)
        self.assertEqual(cols, ["depth_from_image_top_mm", "depth_from_core_top_cm", "mean_hu", "sd_hu", "n_voxels"])
        self.assertEqual(n, len(rows))
        self.assertEqual(n, 80)  # row centres 10.5 .. 89.5 mm
        self.assertAlmostEqual(float(rows[0][0]), 10.5)
        self.assertAlmostEqual(float(rows[0][1]), 0.05)  # 0.5 mm below the core top, in cm
        self.assertAlmostEqual(float(rows[0][2]), self.mean[10], places=2)
        self.assertEqual(rows[0][4], "500")

    def test_header_records_input_and_choices_and_is_parseable(self) -> None:
        write_profile_csv(self.path, self.mean, self.sd, self.cnt, 1.0, 10.0, 90.0, self.meta)
        head, _, _ = read(self.path)
        text = "\n".join(head)
        self.assertIn("# input.folder: C:/scans/SE1", text)
        self.assertIn("# mask.sediment_threshold_hu: 364", text)
        self.assertIn("# mask.remove_clasts: no", text)
        self.assertIn("# mask.clast_threshold_hu: -", text)
        js = [ln for ln in head if ln.startswith("# settings_json: ")]
        self.assertEqual(len(js), 1)
        self.assertEqual(json.loads(js[0][len("# settings_json: "):]), self.meta)

    def test_too_few_voxels_are_blank_not_zero(self) -> None:
        mean = self.mean.copy()
        mean[20] = np.nan
        write_profile_csv(self.path, mean, self.sd, self.cnt, 1.0, 10.0, 90.0, self.meta)
        _, _, rows = read(self.path)
        self.assertEqual(rows[10][2], "")  # row 20 is the eleventh written

    def test_axis_columns_only_when_following(self) -> None:
        ax = (np.linspace(30, 33, 100), np.full(100, 35.0))
        write_profile_csv(self.path, self.mean, self.sd, self.cnt, 1.0, 10.0, 90.0, self.meta, ax)
        head, cols, rows = read(self.path)
        self.assertEqual(cols[-2:], ["axis_x_mm", "axis_y_mm"])
        self.assertAlmostEqual(float(rows[0][5]), ax[0][10], places=3)
        self.assertTrue(any("axis_x_mm" in ln for ln in head))

    def test_readable_by_numpy_skipping_the_header(self) -> None:
        write_profile_csv(self.path, self.mean, self.sd, self.cnt, 1.0, 10.0, 90.0, self.meta)
        n_head = sum(1 for ln in self.path.read_text(encoding="utf-8").splitlines() if ln.startswith("#"))
        a = np.genfromtxt(self.path, delimiter=",", names=True, skip_header=n_head)
        self.assertEqual(a["mean_hu"].shape, (80,))
        self.assertAlmostEqual(float(a["mean_hu"][0]), self.mean[10], places=2)


if __name__ == "__main__":
    unittest.main()
