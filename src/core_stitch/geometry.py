"""In-plane geometry: resample, PCA long axis, rotate/translate."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import rotate as nd_rotate
from scipy.ndimage import shift as nd_shift
from skimage.transform import resize


@dataclass(frozen=True)
class CoreGeometry:
    centroid_rc: tuple[float, float]  # row, col in pixel coords
    axis_rc: tuple[float, float]  # unit vector along long axis (row, col)
    length_mm: float
    width_mm: float
    angle_deg: float  # atan2(col, row) of axis — for rotate


def resample_to_mm(
    image: np.ndarray,
    spacing_row_col_mm: tuple[float, float],
    target_mm: float,
) -> np.ndarray:
    """Resample 2D image so pixels are target_mm isotropic."""
    row_mm, col_mm = spacing_row_col_mm
    h, w = image.shape
    out_h = max(1, int(round(h * row_mm / target_mm)))
    out_w = max(1, int(round(w * col_mm / target_mm)))
    if out_h == h and out_w == w:
        return np.asarray(image, dtype=np.float32)
    out = resize(
        image.astype(np.float32),
        (out_h, out_w),
        order=1,
        mode="constant",
        cval=float(np.median(image)) if np.isfinite(image).any() else 0.0,
        anti_aliasing=True,
        preserve_range=True,
    )
    return np.asarray(out, dtype=np.float32)


def resample_mask_to_mm(
    mask: np.ndarray,
    spacing_row_col_mm: tuple[float, float],
    target_mm: float,
) -> np.ndarray:
    row_mm, col_mm = spacing_row_col_mm
    h, w = mask.shape
    out_h = max(1, int(round(h * row_mm / target_mm)))
    out_w = max(1, int(round(w * col_mm / target_mm)))
    if out_h == h and out_w == w:
        return mask.astype(bool)
    out = resize(
        mask.astype(np.float32),
        (out_h, out_w),
        order=0,
        mode="constant",
        cval=0.0,
        anti_aliasing=False,
        preserve_range=True,
    )
    return out >= 0.5


def core_geometry(mask: np.ndarray, spacing_mm: float) -> CoreGeometry:
    coords = np.column_stack(np.where(mask))
    if coords.shape[0] < 10:
        cy, cx = np.array(mask.shape, dtype=np.float64) / 2.0
        return CoreGeometry(
            centroid_rc=(float(cy), float(cx)),
            axis_rc=(1.0, 0.0),
            length_mm=0.0,
            width_mm=0.0,
            angle_deg=0.0,
        )
    centroid = coords.mean(axis=0)
    centered = coords - centroid
    cov = np.cov(centered.T)
    evals, evecs = np.linalg.eigh(cov)
    axis = evecs[:, int(np.argmax(evals))]
    # Consistent polarity: prefer pointing toward increasing row
    if axis[0] < 0:
        axis = -axis
    axis = axis / (np.linalg.norm(axis) + 1e-12)
    # Length/width from projections
    along = centered @ axis
    perp = centered @ np.array([-axis[1], axis[0]])
    length_mm = float((along.max() - along.min()) * spacing_mm)
    width_mm = float((perp.max() - perp.min()) * spacing_mm)
    angle_deg = float(np.degrees(np.arctan2(axis[1], axis[0])))
    return CoreGeometry(
        centroid_rc=(float(centroid[0]), float(centroid[1])),
        axis_rc=(float(axis[0]), float(axis[1])),
        length_mm=length_mm,
        width_mm=width_mm,
        angle_deg=angle_deg,
    )


def rotate_image(
    image: np.ndarray,
    angle_deg: float,
    *,
    cval: float,
    order: int = 1,
) -> np.ndarray:
    """Rotate CCW about center; reshape=True to fit."""
    if abs(angle_deg) < 1e-6:
        return np.asarray(image, dtype=np.float32)
    return np.asarray(
        nd_rotate(
            image,
            angle=angle_deg,
            reshape=True,
            order=order,
            mode="constant",
            cval=cval,
            prefilter=order > 1,
        ),
        dtype=np.float32,
    )


def rotate_mask(mask: np.ndarray, angle_deg: float) -> np.ndarray:
    rot = rotate_image(mask.astype(np.float32), angle_deg, cval=0.0, order=0)
    return rot >= 0.5


def shift_image(
    image: np.ndarray,
    shift_rc: tuple[float, float],
    *,
    cval: float,
    order: int = 1,
) -> np.ndarray:
    """shift_rc = (drow, dcol) in pixels."""
    if abs(shift_rc[0]) < 1e-6 and abs(shift_rc[1]) < 1e-6:
        return np.asarray(image, dtype=np.float32)
    return np.asarray(
        nd_shift(
            image,
            shift=shift_rc,
            order=order,
            mode="constant",
            cval=cval,
            prefilter=order > 1,
        ),
        dtype=np.float32,
    )


def shift_mask(mask: np.ndarray, shift_rc: tuple[float, float]) -> np.ndarray:
    out = shift_image(mask.astype(np.float32), shift_rc, cval=0.0, order=0)
    return out >= 0.5


def air_value(image: np.ndarray) -> float:
    finite = image[np.isfinite(image)]
    if finite.size == 0:
        return 0.0
    return float(np.percentile(finite, 5))
