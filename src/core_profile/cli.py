"""core-profile-cli: make one down-core profile (CSV) and its recipe (JSON) from an axial TIFF folder or a saved volume."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .classes import ClassParams, Cls
from .loaders import is_tiff_folder, load_axial_folder, load_dicom_mpr, load_npz
from .recipe import UNIT_PRESETS, Recipe
from .reduce import Region
from .run import compute_classes, compute_geometry, compute_level, compute_profile, write_outputs


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="core-profile-cli", description=__doc__)
    p.add_argument("source", help="coronal MPR DICOM folder, folder of axial TIFFs, or a .npz volume saved by core_profile")
    p.add_argument("--hu", action="store_true", help="values are HU: sets --core-thr -300, --zero-offset 1000, --void-below -200 unless given")
    p.add_argument("--target-mm", type=float, default=0.8, help="DICOM MPR: voxel size to resample to")
    p.add_argument("--out", required=True, help="output folder (profile.csv, recipe.json)")
    p.add_argument("--voxel-mm", type=float, help="original pixel size (TIFF folders)")
    p.add_argument("--slice-mm", type=float, help="original spacing between files (default: voxel size)")
    p.add_argument("--binning", type=int, default=1)
    p.add_argument("--z-every", type=int, default=1, help="keep every n-th file")
    p.add_argument("--core-thr", type=float, default=None, help="value separating core+liner from air (default 0.8; -300 with --hu)")
    p.add_argument("--zero-offset", type=float, default=None, help="0 for attenuation-like units (default), 1000 for HU")
    p.add_argument("--void-below", type=float, default=None, help="absolute air-like floor for the void class")
    p.add_argument("--region", choices=("whole", "central", "band"), default="whole")
    p.add_argument("--frac", type=float, default=0.5, help="central region: fraction of the inner radius")
    p.add_argument("--thickness-mm", type=float, default=2.0, help="band region: full width")
    p.add_argument("--use", default="matrix", help="comma list of classes counted: " + ",".join(c.name.lower() for c in Cls))
    p.add_argument("--fwhm-mm", type=float, default=None, help="smooth the profile to this footprint")
    return p


def main(argv: list[str] | None = None) -> int:
    a = build_parser().parse_args(argv)
    src = Path(a.source)
    if src.suffix == ".npz":
        vol = load_npz(src)
    elif is_tiff_folder(src):
        if not a.voxel_mm:
            print("--voxel-mm is required for a TIFF folder", file=sys.stderr)
            return 2
        vol = load_axial_folder(src, a.voxel_mm, a.slice_mm, a.z_every, a.binning)
    else:
        try:
            vol = load_dicom_mpr(src, a.target_mm)
        except (FileNotFoundError, NotADirectoryError, ValueError) as e:
            print(f"cannot read {src} as a TIFF folder, DICOM MPR folder or .npz volume: {e}", file=sys.stderr)
            return 2
        a.hu = True  # DICOM values are HU
    preset = UNIT_PRESETS["HU (medical CT)" if a.hu else "attenuation (positive, Geotek-like)"]
    core_thr = preset["core_thr"] if a.core_thr is None else a.core_thr
    zero_off = preset["zero_offset"] if a.zero_offset is None else a.zero_offset
    void_below = a.void_below if a.void_below is not None else preset["void_below"]
    try:
        use = tuple(Cls[n.strip().upper()] for n in a.use.split(","))
    except KeyError as e:
        print(f"unknown class {e}", file=sys.stderr)
        return 2
    rec = Recipe(Region(a.region, a.frac, a.thickness_mm), use, ClassParams(void_below=void_below), core_thr, zero_off, a.fwhm_mm)
    geom = compute_geometry(vol, rec)
    lev = compute_level(vol, geom, rec)
    cls = compute_classes(vol, geom, lev, rec)
    prof = compute_profile(vol, geom, cls, rec)
    write_outputs(a.out, prof, cls, geom, rec, source=src.name)
    print(f"{len(prof.depth_cm)} slices, {prof.depth_cm[-1]:.1f} cm -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
