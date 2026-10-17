"""Headless coarse-align + export for batch / smoke tests."""

from __future__ import annotations

import argparse
from pathlib import Path

from core_stitch.export import export_accept
from core_stitch.pipeline import coarse_align, set_flip


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="core-stitch-cli")
    p.add_argument("top", type=str, help="TOP series folder")
    p.add_argument("btm", type=str, help="BTM series folder")
    p.add_argument("-o", "--out", type=str, required=True, help="Output directory")
    p.add_argument("--flip", action="store_true", help="Flip mobile half")
    args = p.parse_args(argv)

    state = coarse_align(args.top, args.btm)
    if args.flip:
        set_flip(state, True)
    paths = export_accept(
        state,
        args.out,
        stem="stitch",
        jpeg=True,
        tiff=True,
        dicom=True,
        npy=True,
        json_sidecar=True,
    )
    ov = state.overlap
    for key in ("jpeg", "tiff", "tiff_hu", "json", "dicom_dir", "mpr_npy", "npy"):
        if key in paths:
            print(f"wrote {paths[key]}")
    if ov:
        print(
            f"overlap_mm={ov.offset_mm:.2f} NCC={ov.peak_score:.3f} "
            f"ratio={ov.peak_ratio:.3f} low_conf={ov.low_confidence} flip={state.flip}"
        )
    for w in state.warnings:
        print(f"warning: {w}")
    return 1 if (ov and ov.low_confidence) else 0


if __name__ == "__main__":
    raise SystemExit(main())
