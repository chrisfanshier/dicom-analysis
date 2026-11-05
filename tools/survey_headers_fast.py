"""Fast survey: do the DICOM header positions already place TOP and BTM correctly?

Reads only the headers plus ONE middle slice per series, so it takes a few seconds per
pair instead of the ~30-50 s of the full alignment. For each pair it checks, in the
physical (header) frame:
  * across : do the two core centres land at the same physical X?   (header placement)
  * along  : where would the headers put the overlap, and do the two core profiles
             agree there? It also searches +/-150 mm for a better slide.
Read-only; writes only the CSV.

    python tools/survey_headers_fast.py --out out/survey_fast.csv [--limit N]
"""
from __future__ import annotations

import argparse, csv, time, warnings
from pathlib import Path

import numpy as np
import pydicom

from core_stitch.catalog import Catalog
from core_stitch.load import _hu, select_uid_group, sort_slices
from core_stitch.segment import segment_core
from core_stitch.worklist import build_jobs

COLS = ["stitch_group", "plane", "label", "status", "seconds",
        "px_top", "px_btm", "orient_equal", "iop_tilt_deg", "across_diff_mm", "overlap_mm", "ncc_at_header",
        "best_shift_mm", "ncc_best", "depth_diff_mm", "verdict", "reason", "error"]


def read_series_mid(folder: Path, uid: str):
    items = []
    for p in sorted(Path(folder).iterdir()):
        if not p.is_file():
            continue
        try:
            ds = pydicom.dcmread(p, stop_before_pixels=True, force=True)
        except Exception:
            continue
        if "Rows" in ds:
            items.append((p, ds))
    _uid, items = select_uid_group(items, uid)
    items = sort_slices(items)
    path, hdr = items[len(items) // 2]
    full = pydicom.dcmread(path, force=True)
    img = _hu(full)
    iop = np.asarray(hdr.ImageOrientationPatient, float)
    return dict(img=img, row=iop[:3], col=iop[3:], normal=np.cross(iop[:3], iop[3:]),
                ipp=np.asarray(hdr.ImagePositionPatient, float), px=float(hdr.PixelSpacing[0]),
                n=len(items),
                first_ipp=np.asarray(items[0][1].ImagePositionPatient, float))


def core_profile(s):
    """Core centre X (mm along row dir) and mean-HU profile along the column dir (physical mm)."""
    seg = segment_core(s["img"])
    m = seg.mask
    ys, xs = np.nonzero(m)
    px = s["px"]
    # physical position along the core axis of each image row, and across for the centroid
    origin_along = float(np.dot(s["ipp"], s["col"]))
    origin_across = float(np.dot(s["ipp"], s["row"]))
    across = origin_across + float(xs.mean()) * px
    rows = np.where(m.any(axis=1))[0]
    prof = np.full(m.shape[0], np.nan)
    for r in rows:
        prof[r] = s["img"][r, m[r]].mean()
    along = origin_along + np.arange(m.shape[0]) * px
    return across, along, prof


def ncc(a, b):
    k = np.isfinite(a) & np.isfinite(b)
    if k.sum() < 15:
        return np.nan
    a, b = a[k], b[k]
    a = (a - a.mean()) / (a.std() + 1e-6); b = (b - b.mean()) / (b.std() + 1e-6)
    return float(np.dot(a, b) / len(a))


def shifted(x, sh):
    out = np.full_like(x, np.nan)
    if sh >= 0:
        out[sh:] = x[: len(x) - sh] if sh else x
    else:
        out[: sh] = x[-sh:]
    return out


def survey_pair(job):
    t0 = time.time()
    A = read_series_mid(job.a.path, job.a.series_uid)
    B = read_series_mid(job.b.path, job.b.series_uid)
    out = {"px_top": round(A["px"], 4), "px_btm": round(B["px"], 4)}
    out["iop_tilt_deg"] = round(float(np.degrees(np.arctan2(A["row"][1], A["row"][0]))), 2)
    out["orient_equal"] = bool(np.allclose(np.r_[A["row"], A["col"]], np.r_[B["row"], B["col"]], atol=1e-3))
    xa, za, pa = core_profile(A)
    xb, zb, pb = core_profile(B)
    out["across_diff_mm"] = round(xb - xa, 2)
    out["depth_diff_mm"] = round(float(np.dot(B["first_ipp"] - A["first_ipp"], A["normal"])), 2)
    # common 1 mm grid along the core
    lo = min(za.min(), zb.min()); hi = max(za.max(), zb.max())
    g = np.arange(lo, hi, 1.0)
    fa = np.interp(g, za, np.nan_to_num(pa, nan=-1e9)); fa[np.interp(g, za, np.isfinite(pa).astype(float)) < 0.99] = np.nan
    fb = np.interp(g, zb, np.nan_to_num(pb, nan=-1e9)); fb[np.interp(g, zb, np.isfinite(pb).astype(float)) < 0.99] = np.nan
    fa[(g < za.min()) | (g > za.max())] = np.nan   # np.interp clamps outside the image; do not
    fb[(g < zb.min()) | (g > zb.max())] = np.nan
    both = np.isfinite(fa) & np.isfinite(fb)
    out["overlap_mm"] = int(both.sum())
    out["ncc_at_header"] = None if both.sum() < 15 else round(ncc(fa, fb), 3)
    best, bs = (np.nan, None)
    for sh in range(-150, 151, 1):
        v = ncc(fa, shifted(fb, sh))
        if np.isfinite(v) and (bs is None or v > best):
            best, bs = v, sh
    out["best_shift_mm"], out["ncc_best"] = bs, (None if bs is None else round(best, 3))
    reasons = []
    if not out["orient_equal"]: reasons.append("orientations differ")
    if abs(out["across_diff_mm"]) > 5: reasons.append(f"core centres {out['across_diff_mm']:+.0f} mm apart sideways")
    if abs(out["depth_diff_mm"]) > 5: reasons.append(f"stacks offset {out['depth_diff_mm']:+.0f} mm in depth")
    if out["overlap_mm"] < 10: reasons.append("headers imply no overlap")
    elif bs is not None and abs(bs) > 10 and best - (out["ncc_at_header"] or -1) > 0.1:
        reasons.append(f"profiles fit better at {bs:+d} mm")
    out["verdict"] = "headers_good" if not reasons else "needs_image_stitch"
    out["reason"] = "; ".join(reasons)
    out["seconds"] = round(time.time() - t0, 1)
    out["status"] = "ok"
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--release", default=r"D:\mgr_runs\2026-11_release1")
    ap.add_argument("--groups", nargs="*")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default="out/survey_fast.csv")
    a = ap.parse_args()
    warnings.simplefilter("ignore")
    cat = Catalog(a.release)
    outp = Path(a.out); outp.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    new = not outp.exists()
    if not new:
        done = {(r["stitch_group"], r["label"]) for r in csv.DictReader(outp.open(encoding="utf-8"))}
    n = 0
    with outp.open("a", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, COLS)
        if new: w.writeheader()
        for key in (a.groups or list(cat.groups)):
            g = cat.groups.get(key)
            if g is None: continue
            for job in build_jobs(g):
                if job.kind != "pair" or (key, job.label) in done: continue
                if a.limit and n >= a.limit: return
                row = {"stitch_group": key, "plane": job.plane, "label": job.label}
                try:
                    row.update(survey_pair(job))
                except Exception as exc:
                    row.update({"status": "failed", "error": f"{type(exc).__name__}: {exc}"[:200]})
                w.writerow({c: row.get(c, "") for c in COLS}); fh.flush(); n += 1
                print(n, key.split("|")[-1][:32], job.plane[:4], row.get("status"), row.get("seconds", ""), row.get("verdict", ""), flush=True)

if __name__ == "__main__":
    main()
