"""The pipeline in stages, so a caller (CLI, app) can recompute only what a changed setting affects.

geometry  <- core_thr
level     <- geometry, zero_offset
classes   <- level, class params
profile   <- classes, region, which classes count, footprint smoothing
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from .classes import classify
from .geometry import Geometry, fit_geometry
from .levels import Level, estimate_level
from .loaders import Volume
from .recipe import Recipe
from .reduce import Profile, class_fractions, down_core, footprint_smooth


def compute_geometry(vol: Volume, recipe: Recipe) -> Geometry:
    return fit_geometry(vol.data, vol.voxel_mm, recipe.core_thr)


def compute_level(vol: Volume, geom: Geometry, recipe: Recipe) -> Level:
    return estimate_level(vol.data, geom, vol.step_mm, zero_offset=recipe.zero_offset)[0]


def compute_classes(vol: Volume, geom: Geometry, lev: Level, recipe: Recipe) -> np.ndarray:
    return classify(vol.data, geom, lev, vol.step_mm, recipe.params)


def compute_profile(vol: Volume, geom: Geometry, cls: np.ndarray, recipe: Recipe, top_cm: float = 0.0) -> Profile:
    p = down_core(vol.data, cls, geom, vol.step_mm, recipe.region, recipe.use, top_cm)
    if recipe.footprint_fwhm_mm:
        p.mean = footprint_smooth(p.depth_cm, p.mean, recipe.footprint_fwhm_mm)
    return p


def write_outputs(out_dir: str | Path, prof: Profile, cls: np.ndarray, geom: Geometry, recipe: Recipe, source: str = "") -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    fr = class_fractions(cls, geom, recipe.region)
    with open(out / "profile.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["depth_cm", "mean", "std", "count", "coverage"] + [f"frac_{k}" for k in fr])
        for i in range(len(prof.depth_cm)):
            w.writerow([f"{prof.depth_cm[i]:.4f}", f"{prof.mean[i]:.6g}", f"{prof.std[i]:.6g}", int(prof.count[i]), f"{prof.coverage[i]:.4f}"] + [f"{fr[k][i]:.5f}" for k in fr])
    r = Recipe(**{**recipe.__dict__, "note": (recipe.note + f" source={source}").strip()})
    (out / "recipe.json").write_text(r.to_json())
