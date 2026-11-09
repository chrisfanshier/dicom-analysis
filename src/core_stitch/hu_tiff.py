"""The HU TIFF: signed 16-bit, pixel value = Hounsfield units (no offset)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import tifffile


def write_hu_tiff(path: Path, hu: np.ndarray, dpi: float) -> None:
    """Write ``hu`` (2-D, Hounsfield units) as a signed 16-bit TIFF, deflate-compressed.

    The pixel value *is* the HU (air about -2048, water 0): nothing to add or subtract.  Values are rounded
    to whole HU and limited to the int16 range.  ``dpi`` is stored as the resolution tag.
    """
    arr = np.clip(np.rint(hu), -32768, 32767).astype(np.int16)
    tifffile.imwrite(
        str(path), arr, compression="zlib", resolution=(float(dpi), float(dpi)), resolutionunit="INCH",
        photometric="minisblack",
    )
