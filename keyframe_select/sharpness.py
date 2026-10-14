from pathlib import Path

import cv2


def compute_sharpness(path: Path, normalize: bool = False) -> float:
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Could not read image: {path}")
    lap_var = float(cv2.Laplacian(img, cv2.CV_64F).var())
    if normalize:
        std = float(img.astype("float64").std())
        lap_var = lap_var / (std * std + 1e-6)
    return lap_var
