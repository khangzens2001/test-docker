"""Automatic exposure/brightness normalization for keyframe output images.

Problem
-------
AR-scan sessions use continuous auto-exposure.  The same physical subject can
be captured at very different brightness levels across frames — mean grayscale
intensity swinging from ~75 to ~165 in real scan data.  When these frames are
used as input to 3D reconstruction (NeRF, Gaussian Splatting, photogrammetry),
inconsistent exposure causes colour/appearance inconsistency that hurts quality.

Solution
--------
After keyframe *selection* (which still operates on the original unmodified
images), this module normalises the brightness of the frames that are written
to the output directory.  The original source files are **never modified**.

Three strategies are provided:

``"gamma"`` (recommended)
    Power-law (gamma) correction.  For each frame a per-image gamma is derived
    so that the mean brightness of the transformed image equals ``target``.
    Applies uniformly in RGB — colours are preserved exactly, only luminance
    changes.  Never clips.  Predictable and reversible.

``"linear"``
    Linear scaling: multiply every channel by ``target / source_mean``.
    Values are clipped to [0, 255].  Simple and very fast, but can clip
    highlights when the scale factor is large.

``"clahe"``
    Contrast-Limited Adaptive Histogram Equalisation applied to the L channel
    of LAB colour space.  Balances local contrast rather than global
    brightness.  Colours are largely preserved (L only is touched).  Best
    when frames differ in *local contrast* as well as overall brightness.
    Does **not** use ``target`` — it works independently per image.

Target brightness
-----------------
By default (``target=None``) the target is computed as the **median** of the
mean grayscale brightness across all kept keyframes.  Median is used (rather
than mean) so that a handful of very dark or very bright outlier frames cannot
drag the target to an extreme.

Typical usage (called by ``manifest.py``)::

    from keyframe_select.exposure import compute_target_brightness, normalize_exposure

    paths = [rec.source_path for rec in kept_records]
    target = compute_target_brightness(paths)          # median mean brightness
    for rec, src, dst in zip(kept_records, srcs, dsts):
        orig_mean = normalize_exposure(src, dst, target, strategy="gamma")
        rec.source_mean_brightness = orig_mean
        rec.exposure_normalized = True
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Brightness measurement
# ---------------------------------------------------------------------------

def compute_mean_brightness(path: Path) -> float:
    """Mean grayscale intensity [0.0, 255.0] of an image file.

    Parameters
    ----------
    path:
        Path to a readable image file (JPEG, PNG, …).

    Returns
    -------
    float
        Mean of the grayscale channel.

    Raises
    ------
    ValueError
        If the image cannot be read.
    """
    gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise ValueError(f"Could not read image: {path}")
    return float(gray.mean())


def compute_target_brightness(paths: List[Path]) -> float:
    """Compute the reference brightness for normalisation.

    Returns the **median** of the mean grayscale brightness across all
    supplied images.  Median is robust to outlier frames that are
    pathologically over- or under-exposed.

    Parameters
    ----------
    paths:
        Image paths to measure.  Must be non-empty.

    Returns
    -------
    float
        Target mean brightness value in [0, 255].

    Raises
    ------
    ValueError
        If ``paths`` is empty.
    """
    if not paths:
        raise ValueError("paths must not be empty")
    means = [compute_mean_brightness(p) for p in paths]
    target = float(np.median(means))
    logger.debug(
        "Brightness stats — min=%.1f  median=%.1f  max=%.1f",
        min(means),
        target,
        max(means),
    )
    return target


# ---------------------------------------------------------------------------
# Transform implementations
# ---------------------------------------------------------------------------

def _apply_gamma(img_bgr: np.ndarray, source_mean: float, target: float) -> np.ndarray:
    """Power-law gamma correction to shift mean brightness toward ``target``.

    Derives gamma per-frame as::

        gamma = log(target / 255) / log(source_mean / 255)

    Then applies::

        out = (in / 255) ** gamma * 255

    Edge cases:
    - If ``source_mean`` is essentially 0 or 255, returns the image unchanged
      (no safe gamma can be derived).
    - ``target`` is clamped to [1, 254] to stay in a sensible range.
    """
    target = float(np.clip(target, 1.0, 254.0))

    # Can't compute a finite gamma for black or saturated frames
    if source_mean < 1.0 or source_mean > 254.0:
        logger.debug("gamma: skipping — source_mean=%.1f is out of range", source_mean)
        return img_bgr

    gamma = np.log(target / 255.0) / np.log(source_mean / 255.0)
    gamma = float(np.clip(gamma, 0.1, 10.0))  # safety clamp

    lut = np.array(
        [min(255, max(0, round(((i / 255.0) ** gamma) * 255.0))) for i in range(256)],
        dtype=np.uint8,
    )
    return cv2.LUT(img_bgr, lut)


def _apply_linear(img_bgr: np.ndarray, source_mean: float, target: float) -> np.ndarray:
    """Linear brightness scaling with clip to [0, 255].

    Scale factor = target / source_mean.  Applied uniformly to all channels.
    """
    if source_mean < 1e-6:
        return img_bgr
    scale = target / source_mean
    scaled = img_bgr.astype(np.float32) * scale
    return np.clip(scaled, 0, 255).astype(np.uint8)


def _apply_clahe(img_bgr: np.ndarray) -> np.ndarray:
    """CLAHE on the L channel of LAB colour space.

    Locally normalises contrast.  Does not use an explicit target brightness
    — each image is processed independently.  Colours are largely preserved
    since only the L (luminance) channel is modified.
    """
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l_ch, a_ch, b_ch = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_eq = clahe.apply(l_ch)
    merged = cv2.merge([l_eq, a_ch, b_ch])
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def normalize_exposure(
    src: Path,
    dst: Path,
    target_brightness: float,
    strategy: str = "gamma",
    quality: int = 95,
) -> float:
    """Normalise the brightness of ``src`` and write the result to ``dst``.

    Parameters
    ----------
    src:
        Source image path.
    dst:
        Destination image path.  Parent directory must already exist.
    target_brightness:
        Target mean grayscale brightness [0, 255].  Ignored by ``"clahe"``.
    strategy:
        One of ``"gamma"``, ``"linear"``, ``"clahe"``.
    quality:
        JPEG compression quality [0, 100] used when ``dst`` is a ``.jpg``/
        ``.jpeg`` file.  Ignored for PNG or other lossless formats.

    Returns
    -------
    float
        Original (pre-normalisation) mean brightness of ``src``.

    Raises
    ------
    ValueError
        If ``src`` cannot be read or ``strategy`` is unknown.
    """
    img = cv2.imread(str(src))
    if img is None:
        raise ValueError(f"Could not read image: {src}")

    # Measure on grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    source_mean = float(gray.mean())

    if strategy == "gamma":
        out = _apply_gamma(img, source_mean, target_brightness)
    elif strategy == "linear":
        out = _apply_linear(img, source_mean, target_brightness)
    elif strategy == "clahe":
        out = _apply_clahe(img)
    else:
        raise ValueError(
            f"Unknown exposure strategy: {strategy!r}. "
            "Choose from 'gamma', 'linear', 'clahe'."
        )

    # Write with quality setting for JPEG
    suffix = dst.suffix.lower()
    if suffix in (".jpg", ".jpeg"):
        cv2.imwrite(str(dst), out, [cv2.IMWRITE_JPEG_QUALITY, quality])
    else:
        cv2.imwrite(str(dst), out)

    logger.debug(
        "exposure [%s] %s: mean %.1f → %.1f (target=%.1f)",
        strategy,
        src.name,
        source_mean,
        float(cv2.cvtColor(out, cv2.COLOR_BGR2GRAY).mean()),
        target_brightness,
    )
    return source_mean
