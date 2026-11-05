"""Run the SimpleITK header-geometry stitch on real pairs and check it.

    python tools/itk_compare.py "HLY0501|ctscan/OSU-HLY0501-6JC-8-ctscan" ... [--move ALONG ACROSS DEPTH]

For each pair it prints
  * what was chosen (reference half, pixel sizes, placement, overlap),
  * three error checks:
      1. the reference half's voxels in the output are bit-identical to the original series,
      2. the join: correlation of the two halves over the overlap core, and the same measure with
         the bottom half slid +/- a few pixels along the core (the peak should sit at 0),
      3. where the bottom half starts versus the slide you accepted in an earlier export (sidecar),
  * and writes preview PNGs (whole core, and a zoom on the overlap) to ``out/``.
Read-only on the release; writes only to ``out/``.
"""
from __future__ import annotations

import argparse, glob, json, time, warnings
from pathlib import Path

import numpy as np
from PIL import Image

from core_stitch.catalog import Catalog
from core_stitch.itk_stitch import stitch_pair
from core_stitch.worklist import build_jobs

RELEASE = r"D:\mgr_runs\2026-11_release1"
OUT = Path(__file__).resolve().parents[1] / "out"


def ncc(a, b):
    a = a.astype(np.float64); b = b.astype(np.float64)
    if a.size < 50:
        return float("nan")
    a = (a - a.mean()) / (a.std() + 1e-9); b = (b - b.mean()) / (b.std() + 1e-9)
    return float((a * b).mean())


def join_profile(res, shifts=range(-8, 9)):
    """NCC of ref vs bottom-half voxels over the overlap core, with the moving half slid by s rows."""
    both = res.ref_core & res.mov_core
    rows = np.where(both.any(axis=(0, 2)))[0]
    r0, r1 = rows.min() + 12, rows.max() - 12  # stay off the image-edge rows
    out = {}
    for s in shifts:
        a = res.ref_vol[:, r0:r1, :]
        b = np.roll(res.mov_vol, s, axis=1)[:, r0:r1, :]
        m = res.ref_core[:, r0:r1, :] & np.roll(res.mov_core, s, axis=1)[:, r0:r1, :]
        out[s] = ncc(a[m], b[m])
    return out


def ref_identity(res):
    """Output voxels of the reference half == the original series voxels (same place, no resampling)."""
    lo = res.info["output_offset_in_reference_index"]  # output index o  <->  reference index o + lo
    vol = res.reference.series.volume
    fp = res.ref_footprint
    ok = True
    n = 0
    for z in range(fp.shape[0]):
        rows, cols = np.where(fp[z])
        if rows.size == 0:
            continue
        zz = z + lo[2]
        if not (0 <= zz < vol.shape[0]):
            ok = False
            continue
        a = res.ref_vol[z][rows, cols]
        b = vol[zz][rows + lo[1], cols + lo[0]]
        ok &= bool(np.array_equal(a, b))
        n += rows.size
    return ok, n


def accepted_along_mm(group_key):
    short = group_key.split("|")[-1].replace("ctscan/OSU-", "").replace("-ctscan", "")
    for f in glob.glob(RELEASE + "\\ct_stitched\\**\\" + short + "_coronal_pair.json", recursive=True):
        m = json.load(open(f))
        return m["total_lag_px"] * m["target_mm"]
    return None


def window(a, lo=300.0, hi=1500.0):
    return np.clip((a - lo) / (hi - lo), 0, 1)


def save_previews(res, name):
    OUT.mkdir(exist_ok=True)
    mid = res.volume.shape[0] // 2
    img = res.volume[mid]
    core = res.ref_core[mid] | res.mov_core[mid]
    cols = np.where(core.any(axis=0))[0]
    c0, c1 = max(0, cols.min() - 15), min(img.shape[1], cols.max() + 15)
    g = (window(img[:, c0:c1]) * 255).astype(np.uint8)
    # whole core: shrink to ~1800 px tall
    f = max(1, g.shape[0] // 1800)
    Image.fromarray(g[::f, ::f]).save(OUT / f"itk_{name}_core.png")
    ov = np.where((res.ref_footprint[mid] & res.mov_footprint[mid]).any(axis=1))[0]
    if ov.size:
        a, b = max(0, ov.min() - 60), min(img.shape[0], ov.max() + 60)
        # show: reference | bottom/moving | fused, side by side over the overlap
        trip = np.hstack([
            window(res.ref_vol[mid][a:b, c0:c1]), np.full((b - a, 6), 0.5),
            window(res.mov_vol[mid][a:b, c0:c1]), np.full((b - a, 6), 0.5),
            window(img[a:b, c0:c1]),
        ])
        Image.fromarray((trip * 255).astype(np.uint8)).save(OUT / f"itk_{name}_overlap_ref_mov_fused.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("groups", nargs="+")
    ap.add_argument("--move", nargs=3, type=float, default=(0.0, 0.0, 0.0))
    ap.add_argument("--placement", default="whole_pixel")
    ap.add_argument("--interp", default="linear")
    a = ap.parse_args()
    warnings.simplefilter("ignore")
    cat = Catalog(RELEASE)
    for key in a.groups:
        g = cat.groups[key]
        job = [j for j in build_jobs(g) if j.kind == "pair"][0]
        name = key.split("|")[-1].replace("ctscan/OSU-", "").replace("-ctscan", "")
        t0 = time.time()
        res = stitch_pair(job.a.path, job.a.series_uid, job.b.path, job.b.series_uid,
                          move_mm=tuple(a.move), placement=a.placement, interpolator=a.interp, crop_pad_mm=10.0)
        dt = time.time() - t0
        i = res.info
        print(f"\n== {name}  ({dt:.0f} s)")
        print(f"   reference={i['reference']} ({i['ref_pixel_mm']:.4f} mm)  moving={i['moving']} ({i['moving_pixel_mm']:.4f} mm, x{i['upsample_factor_moving']:.2f})"
              f"  placement={i['placement']} snapped={i['snapped_to_pixel']}  interp={i['interpolation']}")
        print(f"   output {i['output_size_xyz']} (x,y,z)  overlap {i['overlap_mm']:.0f} mm rows {i['overlap_rows']}")
        ok, nvox = ref_identity(res)
        print(f"   CHECK 1  reference voxels identical to the original series: {ok}  ({nvox:,} voxels compared)")
        prof = join_profile(res)
        best = max(prof, key=lambda s: -1 if np.isnan(prof[s]) else prof[s])
        print(f"   CHECK 2  join correlation at header position {prof[0]:.3f};  peak at shift {best:+d} rows ({prof[best]:.3f});"
              f"  at +/-4 rows {prof[-4]:.3f} / {prof[4]:.3f}")
        # CHECK 3: bottom half starts how far below the top half's first row, vs the accepted slide
        top0 = np.asarray(res.reference.image.GetOrigin()) if res.reference.name == "top" else np.asarray(res.moving.image.GetOrigin())
        btm0 = np.asarray(res.moving.image.GetOrigin()) if res.moving.name == "bottom" else np.asarray(res.reference.image.GetOrigin())
        hdr_along = float(np.dot(btm0 - top0, res.reference.col_dir)) + float(i["move_mm_bottom"][0])
        acc = accepted_along_mm(key)
        print(f"   CHECK 3  bottom starts {hdr_along:.1f} mm below the top half's first row" + ("" if acc is None else f";  your accepted stitch: {acc:.1f} mm  (diff {hdr_along - acc:+.1f} mm)"))
        save_previews(res, name)
        print(f"   previews: out/itk_{name}_core.png, out/itk_{name}_overlap_ref_mov_fused.png")


if __name__ == "__main__":
    main()
