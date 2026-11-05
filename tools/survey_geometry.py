"""Survey: do the DICOM header positions already place the two halves correctly?

For each TOP x BTM pair, compare the offset implied by ImagePositionPatient with the
offset the image-based alignment (Otsu mask + overlap search) finds.

    python tools/survey_geometry.py --release D:/mgr_runs/2026-11_release1 --out survey.csv
    python tools/survey_geometry.py --groups "HLY0501|ctscan/OSU-HLY0501-6JC-8-ctscan" ...

Read-only: reads the release catalog and the snapshot DICOM, writes only the CSV.
Offsets are millimetres in the TOP image frame:
  along  = down the core (rows),  across = sideways (columns),  depth = through the stack.
"""

from __future__ import annotations

import argparse
import csv
import time
import warnings
from pathlib import Path

import numpy as np
import pydicom

from core_stitch.catalog import Catalog
from core_stitch.pipeline import coarse_align
from core_stitch.worklist import build_jobs

COLUMNS = [
    "stitch_group", "plane", "label", "seconds", "status",
    "header_along_mm", "image_along_mm", "d_along_mm",
    "header_across_mm", "image_across_mm", "d_across_mm",
    "header_depth_mm", "d_depth_assumed_mm",
    "tilt_top_deg", "tilt_btm_deg", "low_confidence", "error",
]


def _first_last(series):
    """Header geometry of a series: (iop, ipp of the first slice by position, pixel mm)."""
    ds0 = pydicom.dcmread(series.slice_paths[0], stop_before_pixels=True)
    iop = np.asarray(ds0.ImageOrientationPatient, float)
    row, col = iop[:3], iop[3:]
    normal = np.cross(row, col)
    ipps = []
    for p in (series.slice_paths[0], series.slice_paths[-1]):
        d = pydicom.dcmread(p, stop_before_pixels=True)
        ipps.append(np.asarray(d.ImagePositionPatient, float))
    pos = [float(np.dot(i, normal)) for i in ipps]
    first = ipps[int(np.argmin(pos))]
    return row, col, normal, first


def survey_pair(job) -> dict:
    out = {"label": job.label, "plane": job.plane}
    t0 = time.time()
    st = coarse_align(job.a.path, job.b.path, job.a.series_uid, job.b.series_uid)
    row, col, normal, ipp_a = _first_last(st.fixed.series)
    row_b, col_b, normal_b, ipp_b = _first_last(st.mobile_base.series)
    if not np.allclose(np.r_[row, col], np.r_[row_b, col_b], atol=1e-3):
        out["error"] = "orientations differ"
    delta = ipp_b - ipp_a
    out["header_along_mm"] = float(np.dot(delta, col))
    out["header_across_mm"] = float(np.dot(delta, row))
    out["header_depth_mm"] = float(np.dot(delta, normal))
    mm = st.spacing_mm
    ov = st.overlap
    out["image_along_mm"] = float(st.total_lag_px * mm)
    out["image_across_mm"] = float((st.mobile_base.lateral_dcol + st.total_col_offset_px) * mm)
    # the image pipeline pairs slice k with slice k (stacks centred), i.e. assumes this depth offset
    out["d_depth_assumed_mm"] = out["header_depth_mm"]
    out["d_along_mm"] = out["image_along_mm"] - out["header_along_mm"]
    out["d_across_mm"] = out["image_across_mm"] - out["header_across_mm"]
    out["tilt_top_deg"] = st.fixed.upright_deg
    out["tilt_btm_deg"] = st.mobile_base.upright_deg
    out["low_confidence"] = None if ov is None else bool(ov.low_confidence)
    out["seconds"] = round(time.time() - t0, 1)
    out["status"] = "ok"
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--release", default=r"D:\mgr_runs\2026-11_release1")
    ap.add_argument("--groups", nargs="*", help="stitch_group keys; default all with a DICOM pair")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default="survey_geometry.csv")
    args = ap.parse_args()
    warnings.simplefilter("ignore")
    cat = Catalog(args.release)
    keys = args.groups or list(cat.groups)
    done = set()
    outp = Path(args.out)
    new = not outp.exists()
    if not new:
        done = {(r["stitch_group"], r["label"]) for r in csv.DictReader(outp.open(encoding="utf-8"))}
    n = 0
    with outp.open("a", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, COLUMNS)
        if new:
            w.writeheader()
        for key in keys:
            g = cat.groups.get(key)
            if g is None:
                continue
            for job in build_jobs(g):
                if job.kind != "pair" or (key, job.label) in done:
                    continue
                if args.limit and n >= args.limit:
                    return
                row = {"stitch_group": key}
                try:
                    row.update(survey_pair(job))
                except Exception as exc:  # keep going; record why
                    row.update({"label": job.label, "plane": job.plane, "status": "failed", "error": f"{type(exc).__name__}: {exc}"[:200]})
                w.writerow({c: row.get(c, "") for c in COLUMNS})
                fh.flush()
                n += 1
                print(key.split("|")[-1], job.plane, row.get("status"), row.get("seconds", ""), row.get("d_along_mm", ""), flush=True)


if __name__ == "__main__":
    main()
