"""Combined overview chart generator.

Combines 3 chart-only images (Joists/Neda, Plywood, CF) into a single overview
image for quick visual comparison.
"""

from __future__ import annotations

import logging
import os

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def render_combined(
    neda_chart_path: str,
    plywood_chart_path: str,
    cf_chart_path: str,
    output_path: str,
) -> str | None:
    """Combine 3 chart-only PNGs (Neda, Plywood, CF) into a single wide image.

    Args:
        neda_chart_path: Path to neda/joists chart PNG.
        plywood_chart_path: Path to plywood chart PNG.
        cf_chart_path: Path to CF chart PNG.
        output_path: Target path to save combined PNG.

    Returns:
        output_path string if successful, or None on failure or missing input.
    """
    try:
        # Load images
        img1 = cv2.imread(str(neda_chart_path))
        img2 = cv2.imread(str(plywood_chart_path))
        img3 = cv2.imread(str(cf_chart_path))

        if img1 is None or img2 is None or img3 is None:
            missing = []
            if img1 is None:
                missing.append(f"Neda: {neda_chart_path}")
            if img2 is None:
                missing.append(f"Plywood: {plywood_chart_path}")
            if img3 is None:
                missing.append(f"CF: {cf_chart_path}")
            logger.warning("Missing images for combined visualization: %s", missing)
            return None

        # Get dimensions
        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]
        h3, w3 = img3.shape[:2]

        if h1 <= 0 or h2 <= 0 or h3 <= 0 or w1 <= 0 or w2 <= 0 or w3 <= 0:
            logger.warning("Invalid image dimensions for combined visualization")
            return None

        # Use height of first image (Neda) as reference height
        target_h = h1

        def resize_to_h(img: np.ndarray, h: int) -> np.ndarray:
            if h == target_h:
                return img
            scale = target_h / h
            w = max(1, int(img.shape[1] * scale))
            return cv2.resize(img, (w, target_h), interpolation=cv2.INTER_LANCZOS4)

        img1 = resize_to_h(img1, h1)
        img2 = resize_to_h(img2, h2)
        img3 = resize_to_h(img3, h3)

        # Add thin separator lines between images
        sep_width = 6
        sep_color = (200, 200, 200)  # Light gray (BGR)
        sep = np.full((target_h, sep_width, 3), sep_color, dtype=np.uint8)

        # Combine horizontally
        combined = np.hstack((img1, sep, img2, sep, img3))

        # Ensure destination directory exists
        parent_dir = os.path.dirname(output_path)
        if parent_dir:
            os.makedirs(parent_dir, exist_ok=True)

        # Save with high quality
        cv2.imwrite(output_path, combined, [cv2.IMWRITE_PNG_COMPRESSION, 3])
        return output_path

    except Exception as e:
        logger.warning("Error generating combined image: %s", e, exc_info=True)
        return None
