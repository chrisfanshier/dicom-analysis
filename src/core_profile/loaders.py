"""Read a volume into the engine: a folder of axial TIFFs, or a saved ``.npz`` volume.

A loaded volume is a ``Volume``: ``data[i]`` is axial slice ``i`` (top first), ``voxel_mm`` the in-plane pixel
size after any binning, ``step_mm`` the spacing between the slices kept, plus the unit the values are in.

Axial TIFF folders: if a TIFF carries ``slope = ... offset = ...`` in its ImageDescription (tag 270), values are
``counts * slope + offset`` per file (the scale can differ between sections, so it is read per file). Without it,
values are the raw numbers in the file (``tifffile`` handles float and signed-HU TIFFs).
"""

from __future__ import annotations

import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

_SLOPE_RE = re.compile(r"slope\s*=\s*([-+0-9.Ee]+)\s+offset\s*=\s*([-+0-9.Ee]+)")


@dataclass
class Volume:
    data: np.ndarray  # (n, h, w) float32
    voxel_mm: float
    step_mm: float
    unit: str = ""
    source: str = ""


def _read_one(path: Path, binning: int) -> np.ndarray:
    with Image.open(path) as im:
        desc = im.tag_v2.get(270, "") if hasattr(im, "tag_v2") else ""
        a = np.asarray(im, np.float32)
    m = _SLOPE_RE.search(str(desc))
    if m:
        a = a * np.float32(m.group(1)) + np.float32(m.group(2))
    if binning > 1:
        h = a.shape[0] // binning * binning
        w = a.shape[1] // binning * binning
        a = a[:h, :w].reshape(h // binning, binning, w // binning, binning).mean((1, 3))
    return a


def load_axial_folder(
    folder: str | Path,
    voxel_mm: float,
    slice_mm: float | None = None,
    z_every: int = 1,
    binning: int = 1,
    first: int = 0,
    last: int | None = None,
    unit: str = "",
    threads: int = 8,
) -> Volume:
    """Load ``*.tif`` / ``*.tiff`` in name order. ``voxel_mm`` is the original pixel size; ``slice_mm`` the original
    spacing between files (default: equal to the pixel size, i.e. isotropic)."""
    files = sorted([*Path(folder).glob("*.tif"), *Path(folder).glob("*.tiff")])[first:last][::z_every]
    if not files:
        raise FileNotFoundError(f"no TIFF files in {folder}")
    with ThreadPoolExecutor(threads) as ex:
        data = np.stack(list(ex.map(lambda f: _read_one(f, binning), files)))
    return Volume(data, voxel_mm * binning, (slice_mm or voxel_mm) * z_every, unit, str(folder))


def save_npz(vol: Volume, path: str | Path) -> None:
    np.savez_compressed(path, data=vol.data.astype(np.float16), voxel_mm=vol.voxel_mm, step_mm=vol.step_mm, unit=vol.unit, source=vol.source)


def load_npz(path: str | Path) -> Volume:
    d = np.load(path)
    if "data" in d:
        return Volume(d["data"].astype(np.float32), float(d["voxel_mm"]), float(d["step_mm"]), str(d["unit"]), str(d["source"]))
    raise ValueError(f"{path}: not a core_profile volume")


def load_dicom_mpr(folder: str | Path, target_mm: float = 0.8, series_uid: str | None = None) -> Volume:
    """Load a coronal MPR DICOM series (one half, or a stitched section) as a core-aligned axial volume.

    A coronal MPR is a stack of flat slices through the core, ``slice`` mm apart (often 2 mm), with the core running
    down the rows. That is a 3-D volume, just coarse in one direction. To get the axial slices the engine needs:

    1. rows and columns are block-averaged to about ``target_mm`` (whole-number factor),
    2. the coarse direction (across the stack) is *linearly interpolated* to the same spacing,
    3. axes are reordered to (along the core, through the stack, across the slice).

    The interpolation does not add detail: things smaller than the slice spacing (a 2 mm clast between two coronals)
    are blurred in that direction. Values are HU as stored (rescale slope/intercept applied).
    """
    from core_stitch.load import load_series  # same repo; imported late so TIFF-only use does not need DICOM libs

    s = load_series(folder, series_uid)
    dz, dr, dc = s.spacing_mm
    if abs(dr - dc) > 0.02 * max(dr, dc):
        raise ValueError(f"row spacing {dr} mm and column spacing {dc} mm differ: not supported")
    f = max(1, int(round(target_mm / dc)))
    v = s.volume
    nz, nr, nc = v.shape
    nr, nc = nr // f * f, nc // f * f
    v = v[:, :nr, :nc].reshape(nz, nr // f, f, nc // f, f).mean((2, 4))
    vox = dc * f
    pos = np.arange(int(np.floor((nz - 1) * dz / vox)) + 1) * vox / dz  # slice-index coordinate of each new plane
    lo = np.minimum(np.floor(pos).astype(int), nz - 1)
    hi = np.minimum(lo + 1, nz - 1)
    w = (pos - lo).astype(np.float32)[:, None, None]
    v = v[lo] * (1 - w) + v[hi] * w
    return Volume(np.ascontiguousarray(v.transpose(1, 0, 2), dtype=np.float32), vox, vox, "HU", str(folder))


def is_tiff_folder(folder: str | Path) -> bool:
    p = Path(folder)
    return any(p.glob("*.tif")) or any(p.glob("*.tiff"))
