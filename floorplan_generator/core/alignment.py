import math
from typing import Any, Dict, Optional, Tuple

import numpy as np


def rotate_points_2d(p2d: np.ndarray, angle_deg: float, center: Optional[np.ndarray] = None) -> np.ndarray:
    points = np.asarray(p2d, dtype=float)
    if len(points) == 0:
        return points.copy()
    if center is None:
        center = np.mean(points, axis=0)
    theta = math.radians(float(angle_deg))
    c, s = math.cos(theta), math.sin(theta)
    rotation = np.array([[c, -s], [s, c]], dtype=float)
    return (points - center) @ rotation.T + center


def _robust_bbox_area(p2d: np.ndarray) -> Tuple[float, float, float]:
    if len(p2d) == 0:
        return float("inf"), 0.0, 0.0
    lo = np.percentile(p2d, 1, axis=0)
    hi = np.percentile(p2d, 99, axis=0)
    span = np.maximum(hi - lo, 1e-6)
    return float(span[0] * span[1]), float(span[0]), float(span[1])


def _projection_sharpness(p2d: np.ndarray, bin_size_m: float = 0.04) -> float:
    if len(p2d) < 100:
        return 0.0
    mins = np.min(p2d, axis=0)
    idx_x = np.floor((p2d[:, 0] - mins[0]) / bin_size_m).astype(np.int32)
    idx_y = np.floor((p2d[:, 1] - mins[1]) / bin_size_m).astype(np.int32)
    counts_x = np.bincount(idx_x)
    counts_y = np.bincount(idx_y)
    if counts_x.sum() == 0 or counts_y.sum() == 0:
        return 0.0
    sx = float(np.sum(counts_x.astype(float) ** 2) / (counts_x.sum() ** 2))
    sy = float(np.sum(counts_y.astype(float) ** 2) / (counts_y.sum() ** 2))
    return sx + sy


def _alignment_score(p2d: np.ndarray, angle_deg: float, center: np.ndarray) -> Dict[str, float]:
    rotated = rotate_points_2d(p2d, angle_deg, center)
    area, width, depth = _robust_bbox_area(rotated)
    sharpness = _projection_sharpness(rotated)
    score = -math.log(max(area, 1e-9)) + 0.12 * sharpness
    return {
        "score": float(score),
        "bbox_area": float(area),
        "bbox_width": float(width),
        "bbox_depth": float(depth),
        "sharpness": float(sharpness),
    }


def estimate_manhattan_angle(p2d: np.ndarray, max_points: int = 60000) -> Dict[str, Any]:
    points = np.asarray(p2d, dtype=float)
    if len(points) < 100:
        return {"angle_deg": 0.0, "applied": False, "reason": "too_few_points"}
    if len(points) > max_points:
        stride = max(1, int(math.ceil(len(points) / float(max_points))))
        points = points[::stride]

    center = np.mean(points, axis=0)
    before = _alignment_score(points, 0.0, center)
    best_angle = 0.0
    best = before

    for angle in np.arange(-45.0, 45.0001, 1.0):
        current = _alignment_score(points, float(angle), center)
        if current["score"] > best["score"]:
            best = current
            best_angle = float(angle)

    for angle in np.arange(best_angle - 1.5, best_angle + 1.5001, 0.1):
        current = _alignment_score(points, float(angle), center)
        if current["score"] > best["score"]:
            best = current
            best_angle = float(angle)

    if abs(best_angle) < 0.15:
        best_angle = 0.0
        best = before

    improvement = 0.0
    if before["bbox_area"] > 0:
        improvement = (before["bbox_area"] - best["bbox_area"]) / before["bbox_area"]
    return {
        "angle_deg": float(best_angle),
        "applied": bool(abs(best_angle) > 0.0),
        "method": "robust_bbox_projection_search",
        "score_before": float(before["score"]),
        "score_after": float(best["score"]),
        "bbox_area_before": float(before["bbox_area"]),
        "bbox_area_after": float(best["bbox_area"]),
        "bbox_area_improvement": float(improvement),
        "sharpness_before": float(before["sharpness"]),
        "sharpness_after": float(best["sharpness"]),
        "center_xy": [float(center[0]), float(center[1])],
    }


def align_p2d_manhattan(
    p2d: np.ndarray,
    mode: str = "auto",
    manual_angle_deg: float = 0.0,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    points = np.asarray(p2d, dtype=float)
    mode = (mode or "auto").lower()
    if mode == "off":
        return points, {"enabled": False, "mode": "off", "angle_deg": 0.0, "applied": False}
    if mode == "manual":
        angle = float(manual_angle_deg or 0.0)
        center = np.mean(points, axis=0) if len(points) else np.array([0.0, 0.0])
        return rotate_points_2d(points, angle, center), {
            "enabled": True,
            "mode": "manual",
            "angle_deg": angle,
            "applied": bool(abs(angle) > 0.0),
            "center_xy": [float(center[0]), float(center[1])],
        }

    info = estimate_manhattan_angle(points)
    angle = float(info.get("angle_deg") or 0.0)
    center = np.asarray(info.get("center_xy") or np.mean(points, axis=0), dtype=float) if len(points) else np.array([0.0, 0.0])
    aligned = rotate_points_2d(points, angle, center) if abs(angle) > 0.0 else points
    info.update({"enabled": True, "mode": "auto"})
    return aligned, info
