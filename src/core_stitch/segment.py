"""Otsu core+liner vs air; largest connected component."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from skimage.filters import threshold_otsu
from skimage.measure import label, regionprops


@dataclass(frozen=True)
class SegmentResult:
    mask: np.ndarray  # bool, same shape as input 2D/3D working image
    threshold: float
    n_components: int
    ambiguous: bool  # second component nearly as large


def _otsu_mask(image: np.ndarray) -> tuple[np.ndarray, float]:
    finite = image[np.isfinite(image)]
    if finite.size == 0:
        return np.zeros(image.shape, dtype=bool), 0.0
    # Air is typically very low (e.g. -1024/-2048); core/liner higher
    thr = float(threshold_otsu(finite))
    mask = np.isfinite(image) & (image >= thr)
    # If Otsu picks the wrong side (mask fills most of FOV), invert
    if mask.mean() > 0.5:
        alt = np.isfinite(image) & (image < thr)
        if alt.sum() > 0 and alt.mean() < mask.mean():
            mask = alt
    return mask, thr


def largest_component(mask: np.ndarray, *, ambiguous_ratio: float = 0.6) -> SegmentResult:
    labeled = label(mask.astype(np.uint8), connectivity=2)
    props = regionprops(labeled)
    if not props:
        return SegmentResult(
            mask=np.zeros_like(mask, dtype=bool),
            threshold=0.0,
            n_components=0,
            ambiguous=False,
        )
    props = sorted(props, key=lambda p: p.area, reverse=True)
    keep = props[0].label
    out = labeled == keep
    ambiguous = False
    if len(props) >= 2 and props[0].area > 0:
        ambiguous = (props[1].area / props[0].area) >= ambiguous_ratio
    return SegmentResult(
        mask=out,
        threshold=0.0,
        n_components=len(props),
        ambiguous=ambiguous,
    )


def segment_core(image: np.ndarray) -> SegmentResult:
    """
    Segment core+liner vs air on a 2D coronal image (or mid-slice).

    Returns largest CC mask.
    """
    raw, thr = _otsu_mask(image)
    result = largest_component(raw)
    return SegmentResult(
        mask=result.mask,
        threshold=thr,
        n_components=result.n_components,
        ambiguous=result.ambiguous,
    )


def mid_slice(volume: np.ndarray) -> np.ndarray:
    return np.asarray(volume[volume.shape[0] // 2], dtype=np.float32)
