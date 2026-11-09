import tempfile
import unittest
from pathlib import Path

import numpy as np
import tifffile
from PIL import Image

from core_stitch.hu_tiff import write_hu_tiff


class HuTiffTests(unittest.TestCase):
    def test_signed_hu_round_trip(self):
        hu = np.array([[-2048.0, -1000.4, 0.0], [12.6, 1500.0, 40000.0]], np.float32)
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "x_hu.tif"
            write_hu_tiff(p, hu, 72.0)
            back = tifffile.imread(p)
            self.assertEqual(back.dtype, np.int16)
            self.assertEqual(back.tolist(), [[-2048, -1000, 0], [13, 1500, 32767]])  # rounded, limited to int16
            with Image.open(p) as im:  # other readers see a signed 16-bit image of the same shape
                self.assertEqual(im.size, (3, 2))


if __name__ == "__main__":
    unittest.main()
