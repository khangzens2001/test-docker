from __future__ import annotations

import math

import cv2
import numpy as np
from shapely.geometry import Polygon, LineString


SLICE_Z_MIN_M = 0.8
SLICE_Z_MAX_M = 1.8
SLICE_MIN_POINTS = 1000
OCCUPANCY_RESOLUTION_M = 0.01
OCCUPANCY_MAX_PX = 4000
BBOX_FILL_RATIO_THRESHOLD = 0.82
MIN_L_SHAPE_FILL_RATIO = 0.60
MIN_EDGE_SUPPORT = 0.42
MIN_SINGLE_EDGE_SUPPORT = 0.18
MAX_OCC_OUTSIDE_RATIO = 0.10
NOTCH_OPEN_PX = 5
MIN_NOTCH_RATIO = 0.20
MIN_L_SHAPE_NOTCH_SIDE_M = 0.65
MIN_NOTCH_EDGE_SUPPORT = 0.45
MIN_L_SHAPE_IOU = 0.88
MIN_L_SHAPE_SCORE = 0.04
MIN_CONCAVE_AREA_RATIO = 0.03
AXIS_CLOSURE_M = 0.001
DOORWAY_BRIDGE_M = 0.85
DOORWAY_CHOKE_M = 0.78
DOORWAY_CHOKE_KEEP_RATIO = 0.20
SMALL_ROOM_AREA_M2 = 5.0
CHOKE_MAX_DISCARD_RATIO = 0.25
WALL_KEEP_DILATE_M = 0.15
MIN_CHOKE_BBOX_REDUCTION_M = 0.20
MAX_DOOR_TAIL_SPAN_M = 1.20
MIN_DOOR_TAIL_LEN_M = 0.50
BODY_SPAN_M = 1.50
TAIL_FLUSH_M = 0.25
BUILTIN_MIN_DEPTH_M = 0.35
BUILTIN_MAX_DEPTH_M = 0.90
BUILTIN_MIN_LENGTH_M = 0.80
BUILTIN_MIN_FRONT_M = 0.80
BUILTIN_FRONT_TOL_M = 0.14
BUILTIN_OUTER_MARGIN_M = 0.20
BUILTIN_HIGH_Z_M = 1.55
BUILTIN_HIGH_Z_MIN_POINTS = 40
SMALL_ROOM_KEEPOUT_MAX_AREA_M2 = 4.5
SMALL_ROOM_KEEPOUT_MIN_SHORT_SIDE_M = 1.6
VERTEX_DECIMALS = 3
MAX_NGON_VERTICES = 16
MIN_NGON_VERTICES = 8
MIN_NGON_EDGE_M = 0.40
ORTHO_EDGE_ANGLE_DEG = 1.5
NGON_CAVITY_CLOSE_M = 0.05
HEADING_CLASS_DEG = (0.0, 45.0, 90.0, 135.0)
HEADING_SNAP_DEG = 7.5
MIN_CHAMFER_M = 0.25
MAX_CHAMFER_M = 1.80
MIN_FREE_OBLIQUE_M = 0.50
OPEN_CORRIDOR_MIN_WIDTH_M = 0.90
OPEN_CORRIDOR_MIN_LENGTH_M = 1.50
SECOND_ROOM_MIN_AREA_M2 = 6.0



def slice_wall_band(points: np.ndarray, height_m: float) -> np.ndarray:
    pts = np.asarray(points, dtype=float)
    if pts.ndim != 2 or pts.shape[1] < 3 or len(pts) == 0:
        return np.zeros((0, 3), dtype=float)
    z_hi = max(SLICE_Z_MAX_M, float(height_m) - 0.2)
    band = pts[(pts[:, 2] >= SLICE_Z_MIN_M) & (pts[:, 2] <= z_hi)]
    if len(band) >= SLICE_MIN_POINTS:
        return band
    band = pts[(pts[:, 2] >= 0.3) & (pts[:, 2] <= z_hi)]
    if len(band) >= SLICE_MIN_POINTS:
        return band
    band = pts[(pts[:, 2] >= 0.15) & (pts[:, 2] <= z_hi)]
    if len(band) >= SLICE_MIN_POINTS:
        return band
    return np.zeros((0, 3), dtype=float)


def keep_largest_occupancy_component(
    points_xyz: np.ndarray, trajectory: np.ndarray | None = None
) -> np.ndarray:
    pts = np.asarray(points_xyz, dtype=float)
    if pts.ndim != 2 or len(pts) == 0:
        return np.zeros((0, 3), dtype=float)
    p2d = pts[:, :2]
    x_min, y_min = p2d.min(axis=0)
    x_max, y_max = p2d.max(axis=0)
    resolution = OCCUPANCY_RESOLUTION_M
    width = int(np.ceil((x_max - x_min) / resolution)) + 2
    height = int(np.ceil((y_max - y_min) / resolution)) + 2
    if width > OCCUPANCY_MAX_PX or height > OCCUPANCY_MAX_PX:
        scale = max(width / OCCUPANCY_MAX_PX, height / OCCUPANCY_MAX_PX)
        resolution = resolution * scale
        width = int(np.ceil((x_max - x_min) / resolution)) + 2
        height = int(np.ceil((y_max - y_min) / resolution)) + 2
    grid = np.zeros((height, width), dtype=np.uint8)
    ix = np.clip(np.round((p2d[:, 0] - x_min) / resolution).astype(int), 0, width - 1)
    iy = np.clip(np.round((p2d[:, 1] - y_min) / resolution).astype(int), 0, height - 1)
    grid[iy, ix] = 1
    opened = cv2.morphologyEx(
        grid, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    )
    if np.sum(opened) < 0.25 * np.sum(grid):
        opened = cv2.morphologyEx(
            grid, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        )
    nlab, labels, stats, _ = cv2.connectedComponentsWithStats(opened, 8)
    if nlab <= 1:
        return np.zeros((0, 3), dtype=float)
    areas = stats[1:, cv2.CC_STAT_AREA]
    max_area = float(np.max(areas))
    valid_labels = [i + 1 for i, a in enumerate(areas) if a >= 0.15 * max_area or a >= 200]
    if trajectory is not None:
        t_arr = np.asarray(trajectory, dtype=float)
        if t_arr.ndim == 2 and len(t_arr) > 0 and t_arr.shape[1] >= 2:
            t_valid = t_arr[np.all(np.isfinite(t_arr[:, :2]), axis=1)]
            if len(t_valid) > 0:
                t_ix = np.clip(((t_valid[:, 0] - x_min) / resolution).astype(int), 0, width - 1)
                t_iy = np.clip(((t_valid[:, 1] - y_min) / resolution).astype(int), 0, height - 1)
                t_mask = np.zeros((height, width), dtype=np.uint8)
                t_mask[t_iy, t_ix] = 1
                t_dil = cv2.dilate(t_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (81, 81)))
                for i, a in enumerate(areas):
                    lab = i + 1
                    if lab not in valid_labels and a >= 50:
                        if np.any((labels == lab) & (t_dil > 0)):
                            valid_labels.append(lab)
    keep = np.isin(labels[iy, ix], valid_labels)
    return pts[keep]


def robust_profile_bounds(
    profile: np.ndarray,
    full_size: int,
) -> tuple[int, int, bool]:
    """Calculate robust 1D profile bounding limits using moving average convolution.

    Args:
        profile: 1D projection density profile array.
        full_size: Total dimension span (e.g. grid width or height).

    Returns:
        (min_bound, max_bound, bounds_detected_boolean).
    """
    if profile is None or len(profile) == 0 or full_size <= 0:
        return 0, max(0, full_size - 1), False

    if not np.all(np.isfinite(profile)):
        return 0, max(0, full_size - 1), False

    if full_size < 16 or np.max(profile) <= 0:
        return 0, max(0, full_size - 1), False

    window = max(5, int(round(full_size * 0.04)))
    if window % 2 == 0:
        window += 1
    kernel = np.ones(window, dtype=np.float32) / window
    smooth = np.convolve(profile.astype(np.float32), kernel, mode="same")
    if smooth.max() <= 0:
        return 0, full_size - 1, False

    threshold = max(8.0, float(smooth.max()) * 0.38)
    strong = np.flatnonzero(smooth >= threshold)
    if len(strong) == 0:
        return 0, full_size - 1, False

    lo = int(strong[0])
    hi = int(strong[-1])
    if (hi - lo + 1) < full_size * 0.55:
        return 0, full_size - 1, False

    for i in range(hi + 1, full_size):
        if smooth[i] >= 8.0 or profile[i] >= 8.0:
            hi = i

    for i in range(lo - 1, -1, -1):
        if smooth[i] >= 8.0 or profile[i] >= 8.0:
            lo = i

    margin = max(3, int(round(full_size * 0.025)))
    return max(0, lo - margin), min(full_size - 1, hi + margin), True


def clean_occupancy_projection_profile(
    points_xyz: np.ndarray,
    resolution_m: float = 0.01,
    trajectory: np.ndarray | None = None,
) -> tuple[np.ndarray, dict]:
    """Filter 2D/3D point cloud via accumulative density thresholding and 1D profile trimming.

    Layer 1: Accumulative 2D grid rasterization and percentile density thresholding
             to discard isolated flight noise and sparse dust points.
    Layer 2: 1D moving average profile trimming on column and row projection profiles
             to eliminate LiDAR scatter bleeding through glass windows, balconies, or openings.
    Layer 3: Morphological connected component extraction preserving authentic room boundaries.

    Args:
        points_xyz: (N, 3) or (N, >=2) point cloud array.
        resolution_m: Grid cell resolution in meters (default 0.01 = 10mm).

    Returns:
        filtered_points_xyz: Cleaned point cloud array.
        metrics_dict: Filtering diagnostic metrics containing raw_count, clean_points,
                      trimmed_count, density_threshold, bounds_x, bounds_y, did_profile_trim.
    """
    if points_xyz is None or not isinstance(points_xyz, np.ndarray) or points_xyz.size == 0:
        return points_xyz, {
            "applied": False,
            "raw_count": 0,
            "clean_points": 0,
            "trimmed_count": 0,
            "density_threshold": 0,
            "bounds_x": (0, 0),
            "bounds_y": (0, 0),
            "did_profile_trim": False,
        }

    if points_xyz.ndim != 2 or points_xyz.shape[1] < 2 or len(points_xyz) < 4:
        raw_cnt = len(points_xyz) if hasattr(points_xyz, "__len__") else 0
        return points_xyz, {
            "applied": False,
            "raw_count": raw_cnt,
            "clean_points": raw_cnt,
            "trimmed_count": 0,
            "density_threshold": 0,
            "bounds_x": (0, 0),
            "bounds_y": (0, 0),
            "did_profile_trim": False,
        }

    if not np.all(np.isfinite(points_xyz)):
        raw_cnt = len(points_xyz)
        return points_xyz, {
            "applied": False,
            "raw_count": raw_cnt,
            "clean_points": raw_cnt,
            "trimmed_count": 0,
            "density_threshold": 0,
            "bounds_x": (0, 0),
            "bounds_y": (0, 0),
            "did_profile_trim": False,
        }

    resolution = float(resolution_m) if resolution_m is not None else 0.01
    if resolution <= 0 or not np.isfinite(resolution):
        raw_cnt = len(points_xyz)
        return points_xyz, {
            "applied": False,
            "raw_count": raw_cnt,
            "clean_points": raw_cnt,
            "trimmed_count": 0,
            "density_threshold": 0,
            "bounds_x": (0, 0),
            "bounds_y": (0, 0),
            "did_profile_trim": False,
        }

    raw_count = int(len(points_xyz))
    pts = points_xyz
    p2d = pts[:, :2].astype(float)
    min_x, max_x = float(p2d[:, 0].min()), float(p2d[:, 0].max())
    min_y, max_y = float(p2d[:, 1].min()), float(p2d[:, 1].max())
    span_x = max_x - min_x
    span_y = max_y - min_y
    width = int(np.ceil(span_x / resolution)) + 2
    height = int(np.ceil(span_y / resolution)) + 2

    if (
        width <= 0
        or height <= 0
        or width > 10000
        or height > 10000
        or span_x > 100.0
        or span_y > 100.0
    ):
        return points_xyz, {
            "applied": False,
            "raw_count": raw_count,
            "clean_points": raw_count,
            "trimmed_count": raw_count,
            "density_threshold": 0,
            "bounds_x": (0, max(0, width - 1)),
            "bounds_y": (0, max(0, height - 1)),
            "did_profile_trim": False,
        }

    width = max(width, 2)
    height = max(height, 2)

    # Accumulative 2D grid
    grid = np.zeros((height, width), dtype=np.int32)
    idx_x = np.clip(((p2d[:, 0] - min_x) / resolution).astype(int), 0, width - 1)
    idx_y = np.clip(((p2d[:, 1] - min_y) / resolution).astype(int), 0, height - 1)
    np.add.at(grid, (idx_y, idx_x), 1)

    nonzero = grid[grid > 0]
    if len(nonzero) == 0:
        return pts, {
            "applied": False,
            "raw_count": raw_count,
            "clean_points": raw_count,
            "density_threshold": 0,
            "trimmed_count": 0,
            "bounds_x": (0, width - 1),
            "bounds_y": (0, height - 1),
            "did_profile_trim": False,
        }

    # Layer 1: Percentile density thresholding
    density_threshold = max(1, int(np.floor(np.percentile(nonzero, 25))))
    occupied = (grid > density_threshold).astype(np.uint8)
    # Fail-safe: if density threshold cuts too many cells (e.g. uniform corridor or sparse scan), fall back to grid > 0
    if np.count_nonzero(occupied) < max(1, int(0.20 * np.count_nonzero(grid > 0))):
        occupied = (grid > 0).astype(np.uint8)
        density_threshold = 0

    # Layer 2: 1D convolution profile trimming on column profile Px and row profile Py
    col_profile = occupied.sum(axis=0)
    row_profile = occupied.sum(axis=1)
    x0, x1, trimmed_x = robust_profile_bounds(col_profile, width)
    y0, y1, trimmed_y = robust_profile_bounds(row_profile, height)

    if trajectory is not None:
        t_arr = np.asarray(trajectory, dtype=float)
        if t_arr.ndim == 2 and len(t_arr) > 0 and t_arr.shape[1] >= 2:
            t_valid = t_arr[np.all(np.isfinite(t_arr[:, :2]), axis=1)]
            if len(t_valid) > 0:
                t_x0 = float(np.min(t_valid[:, 0])) - 0.25
                t_x1 = float(np.max(t_valid[:, 0])) + 0.25
                t_y0 = float(np.min(t_valid[:, 1])) - 0.25
                t_y1 = float(np.max(t_valid[:, 1])) + 0.25

                ix_t0 = max(0, int(np.floor((t_x0 - min_x) / resolution)))
                ix_t1 = min(width - 1, int(np.ceil((t_x1 - min_x) / resolution)))
                iy_t0 = max(0, int(np.floor((t_y0 - min_y) / resolution)))
                iy_t1 = min(height - 1, int(np.ceil((t_y1 - min_y) / resolution)))

                x0 = min(x0, ix_t0)
                x1 = max(x1, ix_t1)
                y0 = min(y0, iy_t0)
                y1 = max(y1, iy_t1)

    did_profile_trim = bool(
        (trimmed_x and (x0 > 0 or x1 < width - 1))
        or (trimmed_y and (y0 > 0 or y1 < height - 1))
    )

    profile_keep = (idx_x >= x0) & (idx_x <= x1) & (idx_y >= y0) & (idx_y <= y1)

    occupied_trimmed = occupied.copy()
    occupied_trimmed[:, :x0] = 0
    occupied_trimmed[:, x1 + 1:] = 0
    occupied_trimmed[:y0, :] = 0
    occupied_trimmed[y1 + 1:, :] = 0

    # Layer 3: Morphological connected components (CLOSE + connected components + footprint)
    close_kernel = np.ones((5, 5), np.uint8)
    connected = cv2.morphologyEx(occupied_trimmed, cv2.MORPH_CLOSE, close_kernel)
    nlab, labels, stats, _ = cv2.connectedComponentsWithStats(connected, 8)
    if nlab > 1 and density_threshold > 0:
        areas = stats[1:, cv2.CC_STAT_AREA]
        max_area = float(np.max(areas))
        valid_labels = [i + 1 for i, a in enumerate(areas) if a >= 0.10 * max_area or a >= 50]
        if trajectory is not None:
            t_arr = np.asarray(trajectory, dtype=float)
            if t_arr.ndim == 2 and len(t_arr) > 0 and t_arr.shape[1] >= 2:
                t_valid = t_arr[np.all(np.isfinite(t_arr[:, :2]), axis=1)]
                if len(t_valid) > 0:
                    t_ix = np.clip(((t_valid[:, 0] - min_x) / resolution).astype(int), 0, width - 1)
                    t_iy = np.clip(((t_valid[:, 1] - min_y) / resolution).astype(int), 0, height - 1)
                    t_labels = labels[t_iy, t_ix]
                    for tl in np.unique(t_labels):
                        if tl > 0 and tl not in valid_labels:
                            valid_labels.append(int(tl))
        component_mask = np.isin(labels, valid_labels).astype(np.uint8)
        footprint = cv2.dilate(component_mask, np.ones((3, 3), np.uint8))
        keep_points = (footprint[idx_y, idx_x] > 0) & profile_keep
    else:
        keep_points = profile_keep

    if not np.any(keep_points):
        keep_points = profile_keep if np.any(profile_keep) else np.ones(raw_count, dtype=bool)

    filtered = pts[keep_points]
    trimmed_count = int(raw_count - len(filtered))

    metrics = {
        "applied": True,
        "raw_count": raw_count,
        "clean_points": int(len(filtered)),
        "trimmed_count": trimmed_count,
        "density_threshold": int(density_threshold),
        "bounds_x": (int(x0), int(x1)),
        "bounds_y": (int(y0), int(y1)),
        "did_profile_trim": did_profile_trim,
    }
    return filtered, metrics


def rotate_points_2d(
    p2d: np.ndarray,
    angle_deg: float,
    center: np.ndarray | None = None,
) -> np.ndarray:
    """Rotate 2D points around center by angle in degrees.

    Args:
        p2d: (N, 2) or (N, >=2) float ndarray.
        angle_deg: Rotation angle in degrees (counter-clockwise).
        center: Optional (2,) center of rotation. If None, uses mean(p2d[:, :2]).

    Returns:
        Rotated copy of points with same shape.
    """
    pts = np.asarray(p2d, dtype=float)
    if len(pts) == 0 or abs(angle_deg) < 1e-9:
        return pts.copy()
    p_xy = pts[:, :2]
    if center is None:
        center = np.mean(p_xy, axis=0)
    else:
        center = np.asarray(center, dtype=float)[:2]
    theta = math.radians(float(angle_deg))
    c, s = math.cos(theta), math.sin(theta)
    R = np.array([[c, -s], [s, c]], dtype=float)
    rotated_xy = (p_xy - center) @ R.T + center
    out = pts.copy()
    out[:, :2] = rotated_xy
    return out


def estimate_manhattan_angle_bbox_sharpness(
    p2d: np.ndarray,
    bin_size_m: float = 0.04,
    max_points: int = 7000,
) -> dict:
    """Estimate optimal Manhattan alignment angle maximizing BBox compactness and projection sharpness.

    Objective Function:
        Score(theta) = -ln(max(BBox_Area(theta), 1e-9)) + 0.12 * Sharpness(theta)

    Search Grid:
        - Coarse search: theta in [-45.0°, +45.0°] with 1.0° step (91 angles)
        - Fine search: theta in [theta* - 1.5°, theta* + 1.5°] with 0.1° step (31 angles)
        - Deadband snap: if |theta*| < 0.15°, snap to 0.0°

    Args:
        p2d: (N, 2) or (N, >=2) array of 2D coordinates in meters.
        bin_size_m: Histogram bin size in meters for projection sharpness (default 0.04m).
        max_points: Maximum number of points used for grid evaluation (default 7000).

    Returns:
        dict conforming to M1 Manhattan Alignment contract.
    """
    if p2d is None:
        return {
            "best_angle_deg": 0.0,
            "best_score": 0.0,
            "bbox_area": 0.0,
            "sharpness": 0.0,
            "span_x": 0.0,
            "span_y": 0.0,
            "applied": False,
            "angle_deg": 0.0,
            "score_before": 0.0,
            "score_after": 0.0,
            "bbox_area_before": 0.0,
            "bbox_area_after": 0.0,
            "bbox_area_improvement": 0.0,
            "sharpness_before": 0.0,
            "sharpness_after": 0.0,
            "center_xy": [0.0, 0.0],
            "reason": "none_input",
        }

    try:
        pts = np.asarray(p2d, dtype=float)
        if pts.ndim != 2 or pts.shape[0] < 100 or pts.shape[1] < 2:
            return {
                "best_angle_deg": 0.0,
                "best_score": 0.0,
                "bbox_area": 0.0,
                "sharpness": 0.0,
                "span_x": 0.0,
                "span_y": 0.0,
                "applied": False,
                "angle_deg": 0.0,
                "score_before": 0.0,
                "score_after": 0.0,
                "bbox_area_before": 0.0,
                "bbox_area_after": 0.0,
                "bbox_area_improvement": 0.0,
                "sharpness_before": 0.0,
                "sharpness_after": 0.0,
                "center_xy": [0.0, 0.0],
                "reason": "too_few_points" if (pts.ndim == 2 and pts.shape[0] < 100) else "invalid_shape",
            }
        if not np.all(np.isfinite(pts)):
            return {
                "best_angle_deg": 0.0,
                "best_score": 0.0,
                "bbox_area": 0.0,
                "sharpness": 0.0,
                "span_x": 0.0,
                "span_y": 0.0,
                "applied": False,
                "angle_deg": 0.0,
                "score_before": 0.0,
                "score_after": 0.0,
                "bbox_area_before": 0.0,
                "bbox_area_after": 0.0,
                "bbox_area_improvement": 0.0,
                "sharpness_before": 0.0,
                "sharpness_after": 0.0,
                "center_xy": [0.0, 0.0],
                "reason": "non_finite_points",
            }
    except Exception:
        return {
            "best_angle_deg": 0.0,
            "best_score": 0.0,
            "bbox_area": 0.0,
            "sharpness": 0.0,
            "span_x": 0.0,
            "span_y": 0.0,
            "applied": False,
            "angle_deg": 0.0,
            "score_before": 0.0,
            "score_after": 0.0,
            "bbox_area_before": 0.0,
            "bbox_area_after": 0.0,
            "bbox_area_improvement": 0.0,
            "sharpness_before": 0.0,
            "sharpness_after": 0.0,
            "center_xy": [0.0, 0.0],
            "reason": "non_finite_points",
        }

    points = pts[:, :2]

    # Subsampling optimization: if N > max_points, stride uniformly
    n_pts = len(points)
    if n_pts > max_points:
        stride = max(1, int(math.ceil(n_pts / float(max_points))))
        points = points[::stride]
        n_pts = len(points)

    # Degenerate points guard (all points concentrated within 1mm)
    ptp = np.ptp(points, axis=0)
    if ptp[0] < 1e-3 and ptp[1] < 1e-3:
        return {
            "best_angle_deg": 0.0,
            "best_score": 0.0,
            "bbox_area": 0.0,
            "sharpness": 0.0,
            "span_x": float(ptp[0]),
            "span_y": float(ptp[1]),
            "applied": False,
            "angle_deg": 0.0,
            "score_before": 0.0,
            "score_after": 0.0,
            "bbox_area_before": 0.0,
            "bbox_area_after": 0.0,
            "bbox_area_improvement": 0.0,
            "sharpness_before": 0.0,
            "sharpness_after": 0.0,
            "center_xy": [float(np.mean(points[:, 0])), float(np.mean(points[:, 1]))],
            "reason": "degenerate_points",
        }

    center = np.mean(points, axis=0)
    pts_c = points - center  # (N, 2)
    x0 = pts_c[:, 0:1]
    y0 = pts_c[:, 1:2]
    inv_n2 = 1.0 / (float(n_pts) * float(n_pts))
    bin_size = max(float(bin_size_m), 1e-4)
    inv_bin = 1.0 / bin_size
    k1 = int(round(0.01 * (n_pts - 1)))
    k99 = int(round(0.99 * (n_pts - 1)))

    def _eval_angles_batch(angles_deg: np.ndarray):
        rads = np.radians(angles_deg)
        cos_a = np.cos(rads)
        sin_a = np.sin(rads)
        k_angles = len(angles_deg)

        # Batch rotation using broadcasting: (N, 1) * (K,) -> (N, K)
        xs_batch = x0 * cos_a - y0 * sin_a
        ys_batch = x0 * sin_a + y0 * cos_a

        # Robust BBox Area using 1% and 99% percentiles via np.partition
        part_x = np.partition(xs_batch, (k1, k99), axis=0)
        lo_x, hi_x = part_x[k1], part_x[k99]
        part_y = np.partition(ys_batch, (k1, k99), axis=0)
        lo_y, hi_y = part_y[k1], part_y[k99]
        spans_x = np.maximum(hi_x - lo_x, 1e-6)
        spans_y = np.maximum(hi_y - lo_y, 1e-6)
        areas = spans_x * spans_y

        # Projection Sharpness via bin_size histogram
        min_xs = np.min(xs_batch, axis=0)
        min_ys = np.min(ys_batch, axis=0)
        idx_x_all = np.maximum(0, ((xs_batch - min_xs) * inv_bin).astype(np.int32))
        idx_y_all = np.maximum(0, ((ys_batch - min_ys) * inv_bin).astype(np.int32))
        scores = np.empty(k_angles, dtype=float)
        sharpnesses = np.empty(k_angles, dtype=float)
        for i in range(k_angles):
            cx = np.bincount(idx_x_all[:, i])
            cy = np.bincount(idx_y_all[:, i])
            sx = float(np.dot(cx, cx)) * inv_n2
            sy = float(np.dot(cy, cy)) * inv_n2
            sharp = sx + sy
            sharpnesses[i] = sharp
            scores[i] = -math.log(max(areas[i], 1e-9)) + 0.12 * sharp

        return scores, areas, sharpnesses, spans_x, spans_y

    # Stage 1: Coarse search [-45°, +45°] with 1.0° step (91 angles)
    coarse_angles = np.arange(-45.0, 45.0001, 1.0)
    coarse_scores, coarse_areas, coarse_sharp, coarse_sx, coarse_sy = _eval_angles_batch(coarse_angles)
    best_coarse_idx = int(np.argmax(coarse_scores))
    theta_coarse = float(coarse_angles[best_coarse_idx])

    # Extract angle 0.0° baseline from coarse search (coarse_angles contains 0.0° at index 45)
    zero_idx = int(np.argmin(np.abs(coarse_angles)))
    base_score = float(coarse_scores[zero_idx])
    base_area = float(coarse_areas[zero_idx])
    base_sharpness = float(coarse_sharp[zero_idx])
    span_x_0 = float(coarse_sx[zero_idx])
    span_y_0 = float(coarse_sy[zero_idx])

    # Stage 2: Fine search [theta_coarse - 1.5°, theta_coarse + 1.5°] with 0.1° step (31 angles)
    fine_angles = np.arange(theta_coarse - 1.5, theta_coarse + 1.5001, 0.1)
    fine_scores, fine_areas, fine_sharp, fine_sx, fine_sy = _eval_angles_batch(fine_angles)
    best_fine_idx = int(np.argmax(fine_scores))
    best_angle = float(fine_angles[best_fine_idx])

    # Deadband snap: if |best_angle| < 0.15°, snap to 0.0°
    if abs(best_angle) < 0.15:
        best_angle = 0.0
        final_score = base_score
        final_area = base_area
        final_sharp = base_sharpness
        final_span_x = span_x_0
        final_span_y = span_y_0
    else:
        if best_angle > 45.0:
            best_angle -= 90.0
        elif best_angle < -45.0:
            best_angle += 90.0
        best_angle = round(best_angle, 4)
        final_score = float(fine_scores[best_fine_idx])
        final_area = float(fine_areas[best_fine_idx])
        final_sharp = float(fine_sharp[best_fine_idx])
        final_span_x = float(fine_sx[best_fine_idx])
        final_span_y = float(fine_sy[best_fine_idx])

    improvement = (base_area - final_area) / base_area if base_area > 0 else 0.0

    return {
        "best_angle_deg": float(best_angle),
        "best_score": float(final_score),
        "bbox_area": float(final_area),
        "sharpness": float(final_sharp),
        "span_x": float(final_span_x),
        "span_y": float(final_span_y),
        "applied": bool(abs(best_angle) > 0.0),
        "angle_deg": float(best_angle),
        "score_before": float(base_score),
        "score_after": float(final_score),
        "bbox_area_before": float(base_area),
        "bbox_area_after": float(final_area),
        "bbox_area_improvement": float(improvement),
        "sharpness_before": float(base_sharpness),
        "sharpness_after": float(final_sharp),
        "center_xy": [float(center[0]), float(center[1])],
    }


def apply_hough_yaw(points_xyz: np.ndarray) -> tuple[np.ndarray, float, int]:
    """Dual-Stage Manhattan alignment combining global search with residual Hough refinement.

    Stage 1: Global objective grid search via estimate_manhattan_angle_bbox_sharpness
             finds coarse-to-subdegree optimal angle theta_global.
    Stage 2: HoughLinesP detects fine residual Delta_theta on pre-rotated points within
             tight window |Delta_theta| <= 1.5 deg.
             If Hough detects < 4 lines, |Delta_theta| > 1.5 deg, or Hough fails,
             it safely falls back 100% to Delta_theta = 0.0 (preserving theta_global).

    Returns:
        (rotated_points_xyz, total_yaw_deg, line_count)
    """
    if points_xyz is None:
        return points_xyz, 0.0, 0
    try:
        pts = np.asarray(points_xyz, dtype=float)
        if pts.ndim != 2 or pts.shape[1] < 2 or len(pts) < 100:
            return pts, 0.0, 0
        if not np.all(np.isfinite(pts)):
            return points_xyz, 0.0, 0
    except (TypeError, ValueError):
        return points_xyz, 0.0, 0

    p2d = pts[:, :2]

    # --- Stage 1: Global Objective Alignment ---
    stage1 = estimate_manhattan_angle_bbox_sharpness(p2d, max_points=7000)
    theta_global = float(stage1.get("best_angle_deg", stage1.get("angle_deg", 0.0)))

    # Rotate 2D coordinates to intermediate Stage 1 alignment around origin (0, 0)
    if abs(theta_global) >= 1e-6:
        th1 = np.radians(theta_global)
        c1, s1 = np.cos(th1), np.sin(th1)
        R1 = np.array([[c1, -s1], [s1, c1]], dtype=float)
        p2d_rot = p2d @ R1.T
        out_stage1 = pts.copy()
        out_stage1[:, :2] = p2d_rot
    else:
        p2d_rot = p2d
        out_stage1 = pts

    # --- Stage 2: Residual Hough Refinement ---
    delta_theta = 0.0
    line_count = 0

    if not np.all(np.isfinite(p2d_rot)):
        return out_stage1, float(theta_global), 0

    p2d_grid = p2d_rot if len(p2d_rot) <= 10000 else p2d_rot[::max(1, len(p2d_rot) // 10000)]
    x_min, y_min = p2d_grid.min(axis=0)
    x_max, y_max = p2d_grid.max(axis=0)
    resolution = OCCUPANCY_RESOLUTION_M
    span_x = (x_max - x_min) / resolution
    span_y = (y_max - y_min) / resolution

    if not (np.isfinite(span_x) and np.isfinite(span_y)):
        return out_stage1, float(theta_global), 0

    width = int(np.ceil(span_x)) + 2
    height = int(np.ceil(span_y)) + 2

    if not (0 < width <= 10000 and 0 < height <= 10000):
        return out_stage1, float(theta_global), 0

    if 32 <= width <= OCCUPANCY_MAX_PX and 32 <= height <= OCCUPANCY_MAX_PX:
        grid = np.zeros((height, width), dtype=np.uint8)
        ix = np.clip(np.round((p2d_grid[:, 0] - x_min) / resolution).astype(int), 0, width - 1)
        iy = np.clip(np.round((p2d_grid[:, 1] - y_min) / resolution).astype(int), 0, height - 1)
        grid[iy, ix] = 255
        grid_closed = cv2.morphologyEx(
            grid, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        )
        min_line = max(25, int(min(width, height) * 0.15))
        lines = cv2.HoughLinesP(grid_closed, 1, np.pi / 180, 25, minLineLength=min_line, maxLineGap=8)

        if lines is not None:
            residuals, weights = [], []
            for x1, y1, x2, y2 in lines.reshape(-1, 4):
                length = float(np.hypot(x2 - x1, y2 - y1))
                if length < min_line:
                    continue
                angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
                residual = ((angle + 45.0) % 90.0) - 45.0
                if abs(abs(residual) - 45.0) < 1.5:
                    # Ignore 45-degree diagonal lines
                    continue
                if abs(residual) > 20.0:
                    # Ignore non-orthogonal lines
                    continue
                residuals.append(residual)
                weights.append(length ** 2)

            line_count = len(residuals)
            if line_count >= 4:
                residuals_arr = np.asarray(residuals, dtype=float)
                weights_arr = np.asarray(weights, dtype=float)
                bins = np.linspace(-45.0, 45.0, 91)
                hist, bin_edges = np.histogram(residuals_arr, bins=bins, weights=weights_arr)
                mode_idx = int(np.argmax(hist))
                bin_lo = bin_edges[mode_idx]
                bin_hi = bin_edges[mode_idx + 1]
                in_bin = (residuals_arr >= bin_lo - 1.5) & (residuals_arr <= bin_hi + 1.5)
                if np.any(in_bin):
                    best_residual = float(
                        np.sum(residuals_arr[in_bin] * weights_arr[in_bin]) / np.sum(weights_arr[in_bin])
                    )
                else:
                    best_residual = float(0.5 * (bin_lo + bin_hi))

                hough_corr = -best_residual
                # Narrow window constraint: |Delta_theta| <= 1.5 deg
                # Discretization deadband: < 0.15 deg snaps to 0.0
                if abs(hough_corr) <= 1.5 and abs(hough_corr) >= 0.15:
                    delta_theta = hough_corr

    # --- Total Angle Composition & Transformation ---
    theta_total = theta_global + delta_theta
    if abs(theta_total) < 0.15:
        theta_total = 0.0

    if abs(theta_total) < 1e-6:
        return pts, 0.0, int(line_count)

    theta_rad = np.radians(theta_total)
    c, s = np.cos(theta_rad), np.sin(theta_rad)
    R_total = np.array([[c, -s], [s, c]], dtype=float)
    out = pts.copy()
    out[:, :2] = p2d @ R_total.T
    return out, float(theta_total), int(line_count)


def _raster_u8(p2d: np.ndarray) -> tuple[np.ndarray, float, float, float, int, int]:
    x_min, y_min = p2d.min(axis=0)
    x_max, y_max = p2d.max(axis=0)
    resolution = OCCUPANCY_RESOLUTION_M
    width = int(np.ceil((x_max - x_min) / resolution)) + 2
    height = int(np.ceil((y_max - y_min) / resolution)) + 2
    if width > OCCUPANCY_MAX_PX or height > OCCUPANCY_MAX_PX:
        scale = max(width / OCCUPANCY_MAX_PX, height / OCCUPANCY_MAX_PX)
        resolution = resolution * scale
        width = int(np.ceil((x_max - x_min) / resolution)) + 2
        height = int(np.ceil((y_max - y_min) / resolution)) + 2
    grid = np.zeros((height, width), dtype=np.uint8)
    ix = np.clip(np.round((p2d[:, 0] - x_min) / resolution).astype(int), 0, width - 1)
    iy = np.clip(np.round((p2d[:, 1] - y_min) / resolution).astype(int), 0, height - 1)
    grid[iy, ix] = 255
    return grid, float(x_min), float(y_min), float(resolution), width, height


def _odd_px(meters: float, resolution: float, minimum: int = 3) -> int:
    k = int(round(float(meters) / float(resolution)))
    if k % 2 == 0:
        k += 1
    return max(minimum, k)


def choke_doorway_tails(
    points_xyz: np.ndarray,
    trajectory: np.ndarray | None = None,
    neck: dict | None = None,
) -> tuple[np.ndarray, dict]:
    pts = np.asarray(points_xyz, dtype=float)
    empty_info = {
        "doorway_choke_applied": False,
        "pre_choke_bbox_m": [0.0, 0.0],
        "post_choke_bbox_m": [0.0, 0.0],
        "kernel_px": 0,
        "bridge_px": 0,
        "cavity_area_m2": 0.0,
        "discarded_cavity_ratio": 0.0,
        "choke_guard_reason": None,
    }
    if pts.ndim != 2 or pts.shape[1] < 2 or len(pts) == 0:
        return np.zeros((0, 3), dtype=float), empty_info
    occupied, x_min, y_min, resolution, width, height = _raster_u8(pts[:, :2])
    support = (occupied > 0).astype(np.uint8) * 255
    ys, xs = np.nonzero(support)
    if len(xs) == 0:
        return pts, empty_info
    pre = [float((xs.max() - xs.min()) * resolution), float((ys.max() - ys.min()) * resolution)]
    bridge_px = _odd_px(DOORWAY_BRIDGE_M, resolution, 11)
    kernel_px = max(11, _odd_px(DOORWAY_CHOKE_M, resolution, 11))
    info = {
        "doorway_choke_applied": False,
        "pre_choke_bbox_m": pre,
        "post_choke_bbox_m": pre,
        "kernel_px": int(kernel_px),
        "bridge_px": int(bridge_px),
        "cavity_area_m2": 0.0,
        "discarded_cavity_ratio": 0.0,
        "choke_guard_reason": None,
    }
    pad = max(bridge_px, kernel_px) + 2
    padded_support = np.pad(support, pad, mode="constant", constant_values=0)
    bridge_k = cv2.getStructuringElement(cv2.MORPH_RECT, (bridge_px, bridge_px))
    closed = cv2.morphologyEx(padded_support, cv2.MORPH_CLOSE, bridge_k)
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return pts, info

    best_cnt = max(contours, key=cv2.contourArea)
    cavity = np.zeros_like(closed)
    cv2.drawContours(cavity, [best_cnt], -1, 255, thickness=cv2.FILLED)

    cavity_unpadded = cavity[pad:-pad, pad:-pad]
    cav_cells = int(np.count_nonzero(cavity_unpadded))
    info["cavity_area_m2"] = float(cav_cells * resolution * resolution)
    x, y, w, h = cv2.boundingRect(best_cnt)
    best_cnt_span = max(w, h) * resolution
    pre_span = max(pre)

    opened = cv2.morphologyEx(
        cavity, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_px, kernel_px))
    )
    nlab, labels, stats, _ = cv2.connectedComponentsWithStats((opened > 0).astype(np.uint8), 8)
    cav_area = float(np.count_nonzero(cavity))

    dilate_radius_px = max(1, int(round(WALL_KEEP_DILATE_M / resolution)))
    dilate_px = 2 * dilate_radius_px + 1
    mask_cav = None
    if neck is not None and neck.get("cut_coord") is not None and neck.get("axis") is not None and neck.get("neck_kind") in ("door", "open_corridor"):
        axis = int(neck["axis"])
        cut_coord = float(neck["cut_coord"])
        tail_dir = neck.get("tail_dir", "prefix")

        frac_in_tail = 0.0
        all_cams_one_side = True
        if trajectory is not None:
            traj_arr_c = np.asarray(trajectory, dtype=float)
            if traj_arr_c.ndim == 2 and len(traj_arr_c) > 0 and traj_arr_c.shape[1] >= 2:
                coords_c = traj_arr_c[:, axis]
                valid_coords = coords_c[np.isfinite(coords_c)]
                if len(valid_coords) > 0:
                    if tail_dir == "prefix":
                        n_tail = int(np.count_nonzero(valid_coords < cut_coord))
                        t_turn = float(np.min(valid_coords))
                    else:
                        n_tail = int(np.count_nonzero(valid_coords > cut_coord))
                        t_turn = float(np.max(valid_coords))
                    frac_in_tail = float(n_tail) / float(len(valid_coords))
                    if n_tail > 0:
                        all_cams_one_side = False
                    if tail_dir == "prefix":
                        if cut_coord > t_turn:
                            cut_coord = t_turn
                    else:
                        if cut_coord < t_turn:
                            cut_coord = t_turn

        cut_px = int(round((cut_coord - (x_min if axis == 0 else y_min)) / resolution))

        # Topological check: Specular void (mirror) vs doorway choke
        # If all camera poses lie on one side of cut (looking at mirror, not crossing through)
        # and both adjacent orthogonal walls extend past the cut plane -> mark specular_void
        is_specular_void = False
        if all_cams_one_side and 0 < cut_px < (width if axis == 0 else height):
            slice_body = cavity_unpadded[:, :cut_px] if (axis == 0 and tail_dir == "suffix") else (
                cavity_unpadded[:, cut_px:] if (axis == 0 and tail_dir == "prefix") else (
                    cavity_unpadded[:cut_px, :] if (axis == 1 and tail_dir == "suffix") else cavity_unpadded[cut_px:, :]
                )
            )
            slice_tail = cavity_unpadded[:, cut_px:] if (axis == 0 and tail_dir == "suffix") else (
                cavity_unpadded[:, :cut_px] if (axis == 0 and tail_dir == "prefix") else (
                    cavity_unpadded[cut_px:, :] if (axis == 1 and tail_dir == "suffix") else cavity_unpadded[:cut_px, :]
                )
            )
            if np.count_nonzero(slice_body) > 0 and np.count_nonzero(slice_tail) > 0:
                axis_proj = 0 if axis == 0 else 1
                hits_body = np.flatnonzero(np.any(slice_body, axis=axis_proj))
                hits_tail = np.flatnonzero(np.any(slice_tail, axis=axis_proj))
                if len(hits_body) and len(hits_tail):
                    span_body_lo, span_body_hi = hits_body.min(), hits_body.max()
                    span_tail_lo, span_tail_hi = hits_tail.min(), hits_tail.max()
                    if (
                        abs(span_body_lo - span_tail_lo) * resolution <= 0.20
                        and abs(span_body_hi - span_tail_hi) * resolution <= 0.20
                    ):
                        is_specular_void = True

        if is_specular_void:
            info["doorway_choke_applied"] = False
            info["specular_void"] = True
            info["choke_guard_reason"] = "specular_void"
            return pts, info

        mask_cav_unpadded = cavity_unpadded.copy()
        if axis == 0:
            if tail_dir == "prefix":
                mask_cav_unpadded[:, :max(0, cut_px)] = 0
            else:
                mask_cav_unpadded[:, min(width, cut_px):] = 0
        else:
            if tail_dir == "prefix":
                mask_cav_unpadded[:max(0, cut_px), :] = 0
            else:
                mask_cav_unpadded[min(height, cut_px):, :] = 0

        mask_cav = np.pad(mask_cav_unpadded, pad, mode="constant", constant_values=0)
    elif len(contours) > 1 and (pre_span - best_cnt_span) >= MIN_CHOKE_BBOX_REDUCTION_M:
        mask_cav = cavity
    elif nlab > 1:
        best_i = int(np.argmax(stats[1:, cv2.CC_STAT_AREA])) + 1
        if stats[best_i, cv2.CC_STAT_AREA] >= DOORWAY_CHOKE_KEEP_RATIO * max(cav_area, 1.0):
            mask_cav = (labels == best_i).astype(np.uint8) * 255

    if mask_cav is not None:
        cavity_unpadded = cavity[pad:-pad, pad:-pad]
        mask_cav_unpadded = mask_cav[pad:-pad, pad:-pad]
        closed_unpadded = closed[pad:-pad, pad:-pad]
        total_cells = max(int(np.count_nonzero(closed_unpadded)), int(np.count_nonzero(cavity_unpadded)), 1)
        kept_cells = int(np.count_nonzero(mask_cav_unpadded))
        cavity_area_m2 = info["cavity_area_m2"]
        discarded_cavity_ratio = 1.0 - (kept_cells / total_cells)
        info["discarded_cavity_ratio"] = float(discarded_cavity_ratio)
        # Check if discarded region forms a narrow aperture leakage tail
        tail_cells = (cavity_unpadded > 0) & (mask_cav_unpadded == 0)
        is_narrow_tail = False
        if np.count_nonzero(tail_cells) > 0:
            ys_t, xs_t = np.nonzero(tail_cells)
            t_span_x = (xs_t.max() - xs_t.min() + 1) * resolution
            t_span_y = (ys_t.max() - ys_t.min() + 1) * resolution
            t_len = max(t_span_x, t_span_y)
            t_wid = min(t_span_x, t_span_y)
            if t_len / max(t_wid, 0.05) > 1.2:
                is_narrow_tail = True

        w_neck_val = float(neck.get("neck_width_m", 0.0)) if neck else 0.0
        area_a_neck = float(neck.get("area_a_m2", 0.0)) if neck else 0.0
        is_small_room = (
            cavity_area_m2 < 6.0
            or (neck is not None and 0.0 < area_a_neck < 6.0)
        )
        is_narrow_aperture = (w_neck_val > 0.0 and w_neck_val <= 1.60) or is_narrow_tail

        max_discard = (
            0.92 if (is_small_room and trajectory is not None)
            else (0.40 if (is_narrow_tail and trajectory is not None) else CHOKE_MAX_DISCARD_RATIO)
        )
        if cavity_area_m2 < SMALL_ROOM_AREA_M2 and discarded_cavity_ratio > max_discard:
            info["doorway_choke_applied"] = False
            info["choke_guard_reason"] = "small_room_discard_cap"
            return pts, info
        mask_d = cv2.dilate(
            mask_cav, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_px, dilate_px))
        )
        mask = mask_d[pad:-pad, pad:-pad]
        ix = np.clip(np.round((pts[:, 0] - x_min) / resolution).astype(int), 0, width - 1)
        iy = np.clip(np.round((pts[:, 1] - y_min) / resolution).astype(int), 0, height - 1)
        keep = mask[iy, ix] > 0
        out = pts[keep]
        if len(out):
            post = [
                float(out[:, 0].max() - out[:, 0].min()),
                float(out[:, 1].max() - out[:, 1].min()),
            ]
            reduction = max(pre[0] - post[0], pre[1] - post[1])
            if reduction >= MIN_CHOKE_BBOX_REDUCTION_M:
                if trajectory is not None:
                    traj_arr = np.asarray(trajectory, dtype=float)
                    if traj_arr.ndim == 2 and len(traj_arr) > 0 and traj_arr.shape[1] >= 2:
                        t_x0, t_x1 = float(np.min(traj_arr[:, 0])), float(np.max(traj_arr[:, 0]))
                        t_y0, t_y1 = float(np.min(traj_arr[:, 1])), float(np.max(traj_arr[:, 1]))
                        o_x0, o_x1 = float(np.min(out[:, 0])), float(np.max(out[:, 0]))
                        o_y0, o_y1 = float(np.min(out[:, 1])), float(np.max(out[:, 1]))
                        encroach_x = max(0.0, float(o_x0 - t_x0), float(t_x1 - o_x1))
                        encroach_y = max(0.0, float(o_y0 - t_y0), float(t_y1 - o_y1))
                        max_encroach = max(encroach_x, encroach_y)
                        is_small_room_5m2 = (
                            cavity_area_m2 < 5.0
                            or (pre[0] * pre[1] < 5.0)
                            or (neck is not None and 0.0 < area_a_neck < 5.0)
                        )

                        if o_x0 > t_x0 + 0.05 or o_x1 < t_x1 - 0.05 or o_y0 > t_y0 + 0.05 or o_y1 < t_y1 - 0.05:
                            bypass_guard = False
                            if (is_narrow_tail or is_narrow_aperture or is_small_room) and len(traj_arr) > 0:
                                t_ix = np.clip(np.round((traj_arr[:, 0] - x_min) / resolution).astype(int), 0, width - 1)
                                t_iy = np.clip(np.round((traj_arr[:, 1] - y_min) / resolution).astype(int), 0, height - 1)
                                if tail_cells.shape == (height, width):
                                    in_tail_mask = tail_cells[t_iy, t_ix] > 0
                                else:
                                    in_tail_mask = (
                                        (traj_arr[:, 0] < o_x0) | (traj_arr[:, 0] > o_x1) |
                                        (traj_arr[:, 1] < o_y0) | (traj_arr[:, 1] > o_y1)
                                    )
                                n_in_tail = int(np.count_nonzero(in_tail_mask))
                                frac_in_tail = n_in_tail / float(len(traj_arr))
                                L_in_tail = 0.0
                                if n_in_tail > 1:
                                    diffs = np.linalg.norm(np.diff(traj_arr[:, :2], axis=0), axis=1)
                                    step_in = in_tail_mask[:-1] & in_tail_mask[1:] & (diffs < 2.0)
                                    L_in_tail = float(np.sum(diffs[step_in]))
                                t_span_tail = max(float(np.ptp(traj_arr[in_tail_mask, 0])), float(np.ptp(traj_arr[in_tail_mask, 1]))) if n_in_tail > 1 else 0.0
                                is_small_enclosed_neck = (
                                    neck is not None
                                    and neck.get("neck_kind") in ("door", "open_corridor")
                                    and (
                                        min(float(neck.get("area_a_m2", 999.0)), float(neck.get("area_b_m2", 999.0))) < 6.0
                                        or cavity_area_m2 < 10.0
                                    )
                                    and 0.0 < float(neck.get("neck_width_m", 0.0)) <= 1.60
                                )
                                dwell_in_tail = float(n_in_tail / max(len(traj_arr) - n_in_tail, 1))
                                is_true_tail = (frac_in_tail < 0.30 and dwell_in_tail < 0.25)
                                if (is_small_room and is_narrow_aperture) or is_small_enclosed_neck:
                                    bypass_guard = True
                                elif is_small_room and frac_in_tail < 0.40:
                                    bypass_guard = True
                                elif is_true_tail:
                                    if is_small_room and frac_in_tail < 0.30:
                                        bypass_guard = True
                                    elif is_narrow_tail and frac_in_tail < 0.20 and (L_in_tail < 1.50 or t_span_tail < 0.80):
                                        bypass_guard = True

                            if not bypass_guard:
                                info["doorway_choke_applied"] = False
                                info["choke_guard_reason"] = "trajectory_encroachment_guard"
                                return pts, info
                info["doorway_choke_applied"] = True
                info["post_choke_bbox_m"] = post
                return out, info

    return _clip_doorway_profile_tails(
        pts, support, x_min, y_min, resolution, width, height, info, trajectory=trajectory
    )


def _clip_doorway_profile_tails(
    pts: np.ndarray,
    support: np.ndarray,
    x_min: float,
    y_min: float,
    resolution: float,
    width: int,
    height: int,
    info: dict,
    trajectory: np.ndarray | None = None,
) -> tuple[np.ndarray, dict]:
    binary = support > 0

    def spans_along(axis: int) -> np.ndarray:
        n = width if axis == 0 else height
        spans = np.zeros(n, dtype=float)
        for i in range(n):
            hits = np.flatnonzero(binary[:, i] if axis == 0 else binary[i, :])
            if len(hits):
                spans[i] = (hits.max() - hits.min() + 1) * resolution
        return spans

    def ortho_range(axis: int, start: int, end: int) -> tuple[float, float] | None:
        if end < start:
            return None
        if axis == 0:
            hits = np.flatnonzero(np.any(binary[:, start : end + 1], axis=1))
        else:
            hits = np.flatnonzero(np.any(binary[start : end + 1, :], axis=0))
        if len(hits) == 0:
            return None
        return float(hits.min() * resolution), float(hits.max() * resolution)

    def is_flush_arm(tail_rng: tuple[float, float] | None, body_rng: tuple[float, float] | None) -> bool:
        if tail_rng is None or body_rng is None:
            return True
        return (
            abs(tail_rng[0] - body_rng[0]) <= TAIL_FLUSH_M
            or abs(tail_rng[1] - body_rng[1]) <= TAIL_FLUSH_M
        )

    def clip_ends(axis: int, spans: np.ndarray) -> tuple[int, int] | None:
        occ = np.flatnonzero(spans > 0.05)
        if len(occ) < 10:
            return None
        max_span = float(np.max(spans[occ]))
        effective_body_span = min(BODY_SPAN_M, max(0.50, 0.45 * max_span))
        body = np.flatnonzero(spans >= effective_body_span)
        if len(body) < 5:
            return None
        b0, b1 = int(body[0]), int(body[-1])
        lo, hi = int(occ[0]), int(occ[-1])
        body_rng = ortho_range(axis, b0, b1)
        prefix = spans[lo:b0]
        if len(prefix):
            p_occ = np.flatnonzero(prefix > 0.05)
            if len(p_occ):
                plen = (b0 - lo) * resolution
                pspan = float(prefix[p_occ].max())
                tail_rng = ortho_range(axis, lo, b0 - 1)
                if (
                    plen >= MIN_DOOR_TAIL_LEN_M
                    and pspan <= MAX_DOOR_TAIL_SPAN_M
                    and not is_flush_arm(tail_rng, body_rng)
                ):
                    lo = b0
        suffix = spans[b1 + 1 : hi + 1]
        if len(suffix):
            s_occ = np.flatnonzero(suffix > 0.05)
            if len(s_occ):
                slen = (hi - b1) * resolution
                sspan = float(suffix[s_occ].max())
                tail_rng = ortho_range(axis, b1 + 1, hi)
                if (
                    slen >= MIN_DOOR_TAIL_LEN_M
                    and sspan <= MAX_DOOR_TAIL_SPAN_M
                    and not is_flush_arm(tail_rng, body_rng)
                ):
                    hi = b1
        return lo, hi

    x_range = clip_ends(0, spans_along(0))
    y_range = clip_ends(1, spans_along(1))
    if x_range is None or y_range is None:
        return pts, info
    nx_lo, nx_hi = x_range
    ny_lo, ny_hi = y_range
    ix = np.clip(np.round((pts[:, 0] - x_min) / resolution).astype(int), 0, width - 1)
    iy = np.clip(np.round((pts[:, 1] - y_min) / resolution).astype(int), 0, height - 1)
    keep = (ix >= nx_lo) & (ix <= nx_hi) & (iy >= ny_lo) & (iy <= ny_hi)
    out = pts[keep]
    if len(out) == 0:
        return pts, info
    post = [
        float(out[:, 0].max() - out[:, 0].min()),
        float(out[:, 1].max() - out[:, 1].min()),
    ]
    pre = info["pre_choke_bbox_m"]
    reduction = max(pre[0] - post[0], pre[1] - post[1])
    if reduction < MIN_CHOKE_BBOX_REDUCTION_M:
        return pts, info
    info = dict(info)
    discarded_ratio = 1.0 - (len(out) / max(len(pts), 1))
    info["discarded_cavity_ratio"] = float(discarded_ratio)

    discarded_mask = ~keep
    is_narrow_tail = False
    if np.count_nonzero(discarded_mask) > 0:
        disc_pts = pts[discarded_mask, :2]
        d_span_x = float(np.ptp(disc_pts[:, 0])) if len(disc_pts) else 0.0
        d_span_y = float(np.ptp(disc_pts[:, 1])) if len(disc_pts) else 0.0
        d_len = max(d_span_x, d_span_y)
        d_wid = min(d_span_x, d_span_y)
        if d_len / max(d_wid, 0.05) > 1.2:
            is_narrow_tail = True

    is_small_room = (info.get("cavity_area_m2", 0.0) < 6.0)
    max_discard = (
        0.92 if is_small_room and trajectory is not None
        else (0.40 if is_narrow_tail and trajectory is not None else CHOKE_MAX_DISCARD_RATIO)
    )
    if info.get("cavity_area_m2", 0.0) < SMALL_ROOM_AREA_M2 and discarded_ratio > max_discard:
        info["doorway_choke_applied"] = False
        info["choke_guard_reason"] = "small_room_discard_cap"
        return pts, info
    if trajectory is not None:
        traj_arr = np.asarray(trajectory, dtype=float)
        if traj_arr.ndim == 2 and len(traj_arr) > 0 and traj_arr.shape[1] >= 2:
            t_x0, t_x1 = float(np.min(traj_arr[:, 0])), float(np.max(traj_arr[:, 0]))
            t_y0, t_y1 = float(np.min(traj_arr[:, 1])), float(np.max(traj_arr[:, 1]))
            o_x0, o_x1 = float(np.min(out[:, 0])), float(np.max(out[:, 0]))
            o_y0, o_y1 = float(np.min(out[:, 1])), float(np.max(out[:, 1]))
            encroach_x = max(0.0, float(o_x0 - t_x0), float(t_x1 - o_x1))
            encroach_y = max(0.0, float(o_y0 - t_y0), float(t_y1 - o_y1))
            max_encroach = max(encroach_x, encroach_y)
            is_small_room_5m2 = (
                float(info.get("cavity_area_m2", 0.0)) < 5.0
                or (pre[0] * pre[1] < 5.0)
            )

            if o_x0 > t_x0 + 0.05 or o_x1 < t_x1 - 0.05 or o_y0 > t_y0 + 0.05 or o_y1 < t_y1 - 0.05:
                bypass_guard = False
                if (is_narrow_tail or is_small_room) and len(traj_arr) > 0:
                    in_tail_mask = (
                        (traj_arr[:, 0] < o_x0) | (traj_arr[:, 0] > o_x1) |
                        (traj_arr[:, 1] < o_y0) | (traj_arr[:, 1] > o_y1)
                    )
                    n_in_tail = int(np.count_nonzero(in_tail_mask))
                    frac_in_tail = n_in_tail / float(len(traj_arr))
                    L_in_tail = 0.0
                    if n_in_tail > 1:
                        diffs = np.linalg.norm(np.diff(traj_arr[:, :2], axis=0), axis=1)
                        step_in = in_tail_mask[:-1] & in_tail_mask[1:] & (diffs < 2.0)
                        L_in_tail = float(np.sum(diffs[step_in]))
                    t_span_tail = max(float(np.ptp(traj_arr[in_tail_mask, 0])), float(np.ptp(traj_arr[in_tail_mask, 1]))) if n_in_tail > 1 else 0.0
                    dwell_ratio_b = float(frac_in_tail / max(1.0 - frac_in_tail, 0.01))
                    is_true_tail = (frac_in_tail < 0.30 and dwell_ratio_b < 0.25)
                    if is_small_room and frac_in_tail < 0.40:
                        bypass_guard = True
                    elif is_true_tail:
                        if is_small_room and frac_in_tail < 0.30:
                            bypass_guard = True
                        elif frac_in_tail < 0.20 and (L_in_tail < 1.50 or t_span_tail < 0.80):
                            bypass_guard = True
                if not bypass_guard:
                    info["doorway_choke_applied"] = False
                    info["choke_guard_reason"] = "trajectory_encroachment_guard"
                    return pts, info
    info["doorway_choke_applied"] = True
    info["post_choke_bbox_m"] = post
    return out, info


def _filled_bbox_ratio(occupied: np.ndarray, close_m: float, resolution: float) -> tuple[float, float, np.ndarray]:
    close_px = _odd_px(close_m, resolution, 5)
    closed = cv2.morphologyEx(
        occupied,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (close_px, close_px)),
    )
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 1.0, 0.0, closed
    contour = max(contours, key=cv2.contourArea)
    filled = np.zeros(closed.shape, dtype=np.uint8)
    cv2.drawContours(filled, [contour], -1, 255, thickness=cv2.FILLED)
    x, y, w, h = cv2.boundingRect(contour)
    bbox = max(1, w * h)
    filled_area = float(np.count_nonzero(filled[y : y + h, x : x + w]))
    return float(filled_area / bbox), filled_area, closed


def filled_cavity_mask(points_xy: np.ndarray) -> tuple[np.ndarray, float, float, float]:
    pts = np.asarray(points_xy, dtype=float)
    if pts.ndim != 2 or pts.shape[1] < 2 or len(pts) < 3:
        return np.zeros((1, 1), dtype=np.uint8), 0.0, 0.0, OCCUPANCY_RESOLUTION_M
    occupied, x_min, y_min, resolution, width, height = _raster_u8(pts[:, :2])
    close_px = _odd_px(min(NGON_CAVITY_CLOSE_M, MIN_NGON_EDGE_M / 2.0), resolution, 5)
    closed = cv2.morphologyEx(
        occupied,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (close_px, close_px)),
    )
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros(closed.shape, dtype=np.uint8)
    if contours:
        cv2.drawContours(filled, [max(contours, key=cv2.contourArea)], -1, 255, thickness=cv2.FILLED)
    return filled, float(x_min), float(y_min), float(resolution)


def rectilinearize_contour(
    filled: np.ndarray, x_min: float, y_min: float, resolution: float
) -> np.ndarray | None:
    contours, _ = cv2.findContours(filled, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    cnt = max(contours, key=cv2.contourArea)
    approx = cv2.approxPolyDP(cnt, 2.0, closed=True).reshape(-1, 2)
    if len(approx) < 4:
        return None
    pts = [
        [x_min + float(p[0]) * resolution, y_min + float(p[1]) * resolution]
        for p in approx
    ]
    lines: list[list] = []
    n_pts = len(pts)
    for i in range(n_pts):
        p0 = pts[i]
        p1 = pts[(i + 1) % n_pts]
        dx = p1[0] - p0[0]
        dy = p1[1] - p0[1]
        if abs(dx) < 1e-9 and abs(dy) < 1e-9:
            continue
        if abs(dx) >= abs(dy):
            lines.append(["H", float((p0[1] + p1[1]) / 2.0)])
        else:
            lines.append(["V", float((p0[0] + p1[0]) / 2.0)])

    def _merge_collinear(cur_lines: list[list]) -> list[list]:
        changed = True
        while changed and len(cur_lines) > 2:
            changed = False
            n = len(cur_lines)
            for i in range(n):
                next_i = (i + 1) % n
                if cur_lines[i][0] == cur_lines[next_i][0]:
                    cur_lines[i][1] = (cur_lines[i][1] + cur_lines[next_i][1]) / 2.0
                    cur_lines.pop(next_i)
                    changed = True
                    break
        return cur_lines

    def _compute_vertices(cur_lines: list[list]) -> np.ndarray:
        n = len(cur_lines)
        verts = []
        for i in range(n):
            l_prev = cur_lines[(i - 1) % n]
            l_cur = cur_lines[i]
            if l_prev[0] == "H" and l_cur[0] == "V":
                verts.append([l_cur[1], l_prev[1]])
            elif l_prev[0] == "V" and l_cur[0] == "H":
                verts.append([l_prev[1], l_cur[1]])
            else:
                verts.append([0.0, 0.0])
        return np.asarray(verts, dtype=float)

    lines = _merge_collinear(lines)
    if len(lines) < 4 or len(lines) % 2 != 0:
        return None

    while len(lines) > 4:
        verts = _compute_vertices(lines)
        n = len(lines)
        lengths = [float(np.hypot(*(verts[(i + 1) % n] - verts[i]))) for i in range(n)]
        short = [i for i, L in enumerate(lengths) if L < MIN_NGON_EDGE_M]
        if not short:
            break
        i = min(short, key=lambda k: lengths[k])
        prev_idx = (i - 1) % n
        next_idx = (i + 1) % n
        chosen_coord = lines[prev_idx][1] if lengths[prev_idx] >= lengths[next_idx] else lines[next_idx][1]
        lines[prev_idx][1] = chosen_coord
        for idx in sorted([i, next_idx], reverse=True):
            lines.pop(idx)
        lines = _merge_collinear(lines)
        if len(lines) < 4 or len(lines) % 2 != 0:
            return None

    lines = _merge_collinear(lines)
    if len(lines) < 4 or len(lines) % 2 != 0:
        return None

    verts = _compute_vertices(lines)
    verts = close_orthogonal_polygon(verts)
    if len(verts) < 4 or len(verts) % 2 != 0:
        return None

    n = len(verts)
    for i in range(n):
        d = verts[(i + 1) % n] - verts[i]
        if min(abs(d[0]), abs(d[1])) >= 1e-4:
            return None
        if float(np.hypot(d[0], d[1])) < MIN_NGON_EDGE_M - 1e-4:
            return None

    poly = Polygon(verts)
    if not poly.is_valid or poly.geom_type != "Polygon" or poly.area <= 0.0:
        return None

    return verts



def filled_bbox_fill_ratio(points_xy: np.ndarray) -> float:
    pts = np.asarray(points_xy, dtype=float)
    if len(pts) < 3:
        return 1.0
    occupied, x_min, y_min, resolution, width, height = _raster_u8(pts[:, :2])
    ratio_small, filled_small, _closed_small = _filled_bbox_ratio(occupied, 0.05, resolution)
    ratio_door, filled_door, _ = _filled_bbox_ratio(occupied, DOORWAY_BRIDGE_M, resolution)
    # Door-width close is for hollow wall rings: 5 cm fill stays a thin stroke
    # while door-width fill becomes the room. A solid L already fills at 5 cm.
    if filled_door > 1.0 and filled_small / filled_door < 0.50:
        return ratio_door
    return ratio_small


def close_orthogonal_polygon(xy: np.ndarray) -> np.ndarray:
    pts = np.asarray(xy, dtype=float).reshape(-1, 2)
    n = len(pts)
    lengths, axes, signs = [], [], []
    for i in range(n):
        d = pts[(i + 1) % n] - pts[i]
        if abs(d[0]) >= abs(d[1]):
            axes.append("x")
            signs.append(1.0 if d[0] >= 0 else -1.0)
            lengths.append(abs(float(d[0])))
        else:
            axes.append("y")
            signs.append(1.0 if d[1] >= 0 else -1.0)
            lengths.append(abs(float(d[1])))
    for axis in ("x", "y"):
        idx = [i for i, a in enumerate(axes) if a == axis]
        delta = sum(signs[i] * lengths[i] for i in idx)
        if abs(delta) <= AXIS_CLOSURE_M:
            continue
        j = max(idx, key=lambda i: lengths[i])
        lengths[j] = lengths[j] - delta / signs[j]
        if lengths[j] <= AXIS_CLOSURE_M:
            return pts
    out = [pts[0].copy()]
    for i in range(n - 1):
        cur = out[-1]
        if axes[i] == "x":
            out.append(np.array([cur[0] + signs[i] * lengths[i], cur[1]]))
        else:
            out.append(np.array([cur[0], cur[1] + signs[i] * lengths[i]]))
    return np.vstack(out)


def classify_heading(heading_deg: float, length_m: float) -> tuple[str, float]:
    """Classify a wall segment heading in degrees.

    Returns:
        ("class", class_deg) if within HEADING_SNAP_DEG of a canonical heading
                             (or if length_m < MIN_FREE_OBLIQUE_M and snaps to nearest).
        ("obl", heading_deg) if free oblique and length_m >= MIN_FREE_OBLIQUE_M.
    """
    h = float(heading_deg) % 180.0
    if h < 0.0:
        h += 180.0
    if h >= 180.0:
        h -= 180.0

    best_c = HEADING_CLASS_DEG[0]
    min_diff = 180.0
    for c in HEADING_CLASS_DEG:
        diff = min(abs(h - c), 180.0 - abs(h - c))
        if diff < min_diff:
            min_diff = diff
            best_c = c

    if min_diff <= HEADING_SNAP_DEG:
        return ("class", float(best_c))
    if length_m >= MIN_FREE_OBLIQUE_M:
        return ("obl", float(h))
    return ("class", float(best_c))


def close_polygon_xy(xy: np.ndarray) -> np.ndarray | None:
    from shapely.geometry import Polygon

    pts = np.asarray(xy, dtype=float).reshape(-1, 2)
    n = len(pts)
    if n < 3:
        return None

    diffs = [pts[(i + 1) % n] - pts[i] for i in range(n)]
    lengths = [float(np.hypot(d[0], d[1])) for d in diffs]
    if any(L < AXIS_CLOSURE_M for L in lengths):
        return None

    u_vecs = []
    for i in range(n):
        d = diffs[i]
        deg = float(np.degrees(np.arctan2(d[1], d[0])))
        kind, class_deg = classify_heading(deg, lengths[i])
        if kind == "class":
            k = round(deg / 45.0)
            rad = np.radians(k * 45.0)
            u = np.array([np.cos(rad), np.sin(rad)])
        else:
            u = d / lengths[i]
        u_vecs.append(u)

    u_mat = np.column_stack(u_vecs)  # 2 x n
    E = u_mat @ np.array(lengths)  # 2-vector closure error

    # Solve least-squares length adjustments: u_mat @ delta_L = -E
    AAT = u_mat @ u_mat.T
    det = AAT[0, 0] * AAT[1, 1] - AAT[0, 1] * AAT[1, 0]
    if abs(det) < 1e-9:
        return None

    delta_L = u_mat.T @ np.linalg.solve(AAT, -E)
    new_lengths = np.array(lengths) + delta_L
    if any(L <= AXIS_CLOSURE_M for L in new_lengths):
        return None

    # Reconstruct vertices
    out = [pts[0].copy()]
    for i in range(n - 1):
        out.append(out[-1] + new_lengths[i] * u_vecs[i])
    res = np.vstack(out)

    # Check closure residual
    residual = (res[0] - (res[-1] + new_lengths[-1] * u_vecs[-1]))
    if float(np.hypot(residual[0], residual[1])) > AXIS_CLOSURE_M:
        return None

    try:
        poly = Polygon(res)
        if not poly.is_valid or poly.geom_type != "Polygon" or poly.is_empty:
            return None
    except Exception:
        return None

    return res


def fit_orthogonal_ngon(points_xy: np.ndarray) -> np.ndarray | None:
    pts = np.asarray(points_xy, dtype=float)
    if pts.ndim != 2 or pts.shape[1] < 2 or len(pts) < 3:
        return None
    filled, x_min, y_min, resolution = filled_cavity_mask(pts[:, :2])
    xy = rectilinearize_contour(filled, x_min, y_min, resolution)
    if xy is None:
        return None
    n = len(xy)
    if n < MIN_NGON_VERTICES or n > MAX_NGON_VERTICES or n % 2 != 0:
        return None
    for i in range(n):
        d = xy[(i + 1) % n] - xy[i]
        ang = abs(np.degrees(np.arctan2(d[1], d[0]))) % 90.0
        err = min(ang, 90.0 - ang)
        if err > ORTHO_EDGE_ANGLE_DEG:
            return None
        if float(np.hypot(*d)) < MIN_NGON_EDGE_M - 1e-4:
            return None
    xy = close_orthogonal_polygon(xy)
    try:
        raw_poly = Polygon(xy)
        geom = raw_poly
        if not geom.is_valid:
            geom = geom.buffer(0)
        if geom.is_empty or geom.geom_type != "Polygon":
            return None
        if not raw_poly.is_valid and (abs(geom.area - raw_poly.area) / max(geom.area, 1e-6) > 0.02):
            return None
    except Exception:
        return None
    return np.asarray(xy, dtype=float)


def fit_missing_corner_l_shape(
    points_xy: np.ndarray, *, return_info: bool = False
) -> np.ndarray | tuple[np.ndarray, float] | None:
    p2d = np.asarray(points_xy, dtype=float)[:, :2]
    if len(p2d) < 6:
        return None
    fill = filled_bbox_fill_ratio(p2d)
    if fill < MIN_L_SHAPE_FILL_RATIO:
        return None
    occupied, x_min, y_min, resolution, width, height = _raster_u8(p2d)
    support = (occupied > 0).astype(np.uint8) * 255
    ys, xs = np.nonzero(support)
    if len(xs) == 0:
        return None
    span_x = (xs.max() - xs.min()) * resolution
    span_y = (ys.max() - ys.min()) * resolution
    if span_x < 0.5 or span_y < 0.5 or (xs.max() - xs.min()) < 8 or (ys.max() - ys.min()) < 8:
        return None

    dilated = cv2.dilate(support, np.ones((11, 11), np.uint8))
    close_px = _odd_px(min(NGON_CAVITY_CLOSE_M, MIN_NGON_EDGE_M / 2.0), resolution, 5)
    closed = cv2.morphologyEx(
        occupied,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (close_px, close_px)),
    )
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cavity = np.zeros_like(occupied)
    if contours:
        cv2.drawContours(cavity, [max(contours, key=cv2.contourArea)], -1, 255, thickness=cv2.FILLED)
    target_mask = np.maximum(cavity, occupied)
    target_count = int(cv2.countNonZero(target_mask))
    _line_cache = {}

    notch_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (NOTCH_OPEN_PX, NOTCH_OPEN_PX))
    support_open = cv2.morphologyEx(support, cv2.MORPH_OPEN, notch_kernel)
    integral_open = cv2.integral((support_open > 0).astype(np.uint8))
    integral_dilated = cv2.integral((dilated > 0).astype(np.uint8))
    integral_support = cv2.integral((support > 0).astype(np.uint8))

    H, W = dilated.shape

    def poly_iou(mask: np.ndarray, poly: list[tuple[int, int]]) -> float:
        poly_mask = np.zeros(mask.shape, dtype=np.uint8)
        cv2.fillPoly(poly_mask, [np.asarray(poly, dtype=np.int32)], 255)
        poly_count = int(cv2.countNonZero(poly_mask))
        if poly_count == 0:
            return 0.0
        inter = int(cv2.countNonZero(cv2.bitwise_and(mask, poly_mask)))
        union = target_count + poly_count - inter
        return float(inter / union) if union > 0 else 0.0

    def line_sup(px: int, py: int, qx: int, qy: int) -> float:
        k = (px, py, qx, qy)
        if k in _line_cache:
            return _line_cache[k]
        if py == qy:
            y = 0 if py < 0 else (H - 1 if py >= H else py)
            p_lo = px if px <= qx else qx
            p_hi = qx if px <= qx else px
            x_lo = 0 if p_lo < 0 else (W - 1 if p_lo >= W else p_lo)
            x_hi = 0 if p_hi < 0 else (W - 1 if p_hi >= W else p_hi)
            cnt = x_hi - x_lo + 1
            if cnt <= 0:
                res = 0.0
            else:
                r_sum = int(
                    integral_dilated[y + 1, x_hi + 1]
                    - integral_dilated[y, x_hi + 1]
                    - integral_dilated[y + 1, x_lo]
                    + integral_dilated[y, x_lo]
                )
                res = float(r_sum / cnt)
            _line_cache[k] = res
            return res
        elif px == qx:
            x = 0 if px < 0 else (W - 1 if px >= W else px)
            p_lo = py if py <= qy else qy
            p_hi = qy if py <= qy else py
            y_lo = 0 if p_lo < 0 else (H - 1 if p_lo >= H else p_lo)
            y_hi = 0 if p_hi < 0 else (H - 1 if p_hi >= H else p_hi)
            cnt = y_hi - y_lo + 1
            if cnt <= 0:
                res = 0.0
            else:
                c_sum = int(
                    integral_dilated[y_hi + 1, x + 1]
                    - integral_dilated[y_lo, x + 1]
                    - integral_dilated[y_hi + 1, x]
                    + integral_dilated[y_lo, x]
                )
                res = float(c_sum / cnt)
            _line_cache[k] = res
            return res
        else:
            n_samp = max(1, abs(qx - px) + abs(qy - py))
            xs_e = np.clip(np.linspace(px, qx, n_samp).astype(int), 0, W - 1)
            ys_e = np.clip(np.linspace(py, qy, n_samp).astype(int), 0, H - 1)
            return float((dilated[ys_e, xs_e] > 0).mean())

    def calc_support(poly: list[tuple[int, int]]) -> tuple[float, list[float]]:
        hits = []
        n_p = len(poly)
        for i in range(n_p):
            p = poly[i]
            q = poly[(i + 1) % n_p]
            hits.append(line_sup(p[0], p[1], q[0], q[1]))
        return float(np.mean(hits)) if hits else 0.0, hits

    best = None
    for p in (0.0, 1.5, 2.5):
        x0 = int(np.percentile(xs, p))
        x1 = int(np.percentile(xs, 100.0 - p)) + 1
        y0 = int(np.percentile(ys, p))
        y1 = int(np.percentile(ys, 100.0 - p)) + 1
        bbox_w, bbox_h = x1 - x0, y1 - y0
        bbox_area = max(1, bbox_w * bbox_h)
        total_occ = int(
            integral_support[y1, x1]
            - integral_support[y0, x1]
            - integral_support[y1, x0]
            + integral_support[y0, x0]
        )

        rect = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
        iou_rect = poly_iou(target_mask, rect)
        rect_sup, _ = calc_support(rect)

        min_notch_side = max(10, int(round(min(bbox_w, bbox_h) * 0.08)))
        x_cands = np.unique(np.concatenate([
            np.linspace(x0 + min_notch_side, x1 - min_notch_side, 28).astype(int),
            np.percentile(xs, np.linspace(15, 85, 15)).astype(int),
        ]))
        y_cands = np.unique(np.concatenate([
            np.linspace(y0 + min_notch_side, y1 - min_notch_side, 28).astype(int),
            np.percentile(ys, np.linspace(15, 85, 15)).astype(int),
        ]))
        x_cands = x_cands[(x_cands > x0 + min_notch_side) & (x_cands < x1 - min_notch_side)]
        y_cands = y_cands[(y_cands > y0 + min_notch_side) & (y_cands < y1 - min_notch_side)]

        for xs_mid in x_cands.tolist():
            for ys_mid in y_cands.tolist():
                missing = {
                    "top_left": (x0, ys_mid, xs_mid, y1),
                    "top_right": (xs_mid, ys_mid, x1, y1),
                    "bottom_right": (xs_mid, y0, x1, ys_mid),
                    "bottom_left": (x0, y0, xs_mid, ys_mid),
                }
                for corner, (mx0, my0, mx1, my1) in missing.items():
                    notch_w = mx1 - mx0
                    notch_h = my1 - my0
                    notch_w_m = notch_w * resolution
                    notch_h_m = notch_h * resolution
                    if min(notch_w_m, notch_h_m) < MIN_L_SHAPE_NOTCH_SIDE_M:
                        continue
                    notch_area = notch_w * notch_h
                    if notch_area / bbox_area < MIN_CONCAVE_AREA_RATIO:
                        continue
                    if (bbox_w - notch_w) * resolution < 0.25 or (bbox_h - notch_h) * resolution < 0.25:
                        continue

                    open_count = int(
                        integral_open[my1, mx1]
                        - integral_open[my0, mx1]
                        - integral_open[my1, mx0]
                        + integral_open[my0, mx0]
                    )
                    occ_out = float(open_count) / max(total_occ, 1)
                    if occ_out > MAX_OCC_OUTSIDE_RATIO:
                        continue

                    # Inward notch edges
                    sup_h = line_sup(mx0, ys_mid, mx1, ys_mid)
                    if sup_h < MIN_NOTCH_EDGE_SUPPORT:
                        continue
                    sup_v = line_sup(xs_mid, my0, xs_mid, my1)
                    if sup_v < MIN_NOTCH_EDGE_SUPPORT:
                        continue

                    # Construct polygon only when candidate passes all gates so far
                    if corner == "bottom_right":
                        poly = [
                            (x0, y0),
                            (xs_mid, y0),
                            (xs_mid, ys_mid),
                            (x1, ys_mid),
                            (x1, y1),
                            (x0, y1),
                        ]
                    elif corner == "bottom_left":
                        poly = [
                            (x0, ys_mid),
                            (xs_mid, ys_mid),
                            (xs_mid, y0),
                            (x1, y0),
                            (x1, y1),
                            (x0, y1),
                        ]
                    elif corner == "top_right":
                        poly = [
                            (x0, y0),
                            (x1, y0),
                            (x1, ys_mid),
                            (xs_mid, ys_mid),
                            (xs_mid, y1),
                            (x0, y1),
                        ]
                    else:
                        poly = [
                            (x0, y0),
                            (x1, y0),
                            (x1, y1),
                            (xs_mid, y1),
                            (xs_mid, ys_mid),
                            (x0, ys_mid),
                        ]

                    edge_support, edge_hits = calc_support(poly)
                    min_edge_sup = min(edge_hits) if edge_hits else 0.0
                    if edge_support < MIN_EDGE_SUPPORT or min_edge_sup < MIN_SINGLE_EDGE_SUPPORT:
                        continue

                    iou_l = poly_iou(target_mask, poly)
                    score = (iou_l - iou_rect) + 0.10 * (edge_support - rect_sup)

                    if score < MIN_L_SHAPE_SCORE or iou_l < MIN_L_SHAPE_IOU:
                        continue

                    if best is None or score > best[0]:
                        best = (score, poly, edge_support)

    if best is None:
        return None
    fit_missing_corner_l_shape.last_edge_support = float(best[2])
    pix = np.array(best[1], dtype=float)
    metres = np.column_stack(
        [pix[:, 0] * resolution + x_min, pix[:, 1] * resolution + y_min]
    )
    if return_info:
        return metres, float(best[2])
    return metres


def _rect_polygon(x0: float, y0: float, x1: float, y1: float) -> list[list[float]]:
    return [
        [float(x0), float(y0)],
        [float(x1), float(y0)],
        [float(x1), float(y1)],
        [float(x0), float(y1)],
    ]


def _keepout_strip_along_axis(
    occupied: np.ndarray,
    origin_a: float,
    origin_b: float,
    resolution: float,
    *,
    along_axis: int,
    toward_high: bool,
) -> dict | None:
    """Find a built-in strip on one AABB side.

    along_axis=0 scans columns (strip along X, depth in Y).
    along_axis=1 scans rows (strip along Y, depth in X).
    toward_high selects the + side of the depth axis.
    """
    ny, nx = occupied.shape
    n_along = nx if along_axis == 0 else ny
    n_depth = ny if along_axis == 0 else nx
    if n_along < 8 or n_depth < 8:
        return None
    mid = n_depth // 2
    firsts = np.full(n_along, np.nan)
    lasts = np.full(n_along, np.nan)
    for i in range(n_along):
        line = occupied[:, i] if along_axis == 0 else occupied[i, :]
        hits = np.flatnonzero(line > 0)
        if len(hits) == 0:
            continue
        if toward_high:
            outer_hits = hits[hits >= mid]
            if len(outer_hits) == 0:
                continue
            firsts[i] = float(outer_hits.min())
            lasts[i] = float(outer_hits.max())
        else:
            outer_hits = hits[hits <= mid]
            if len(outer_hits) == 0:
                continue
            firsts[i] = float(outer_hits.max())
            lasts[i] = float(outer_hits.min())
    valid = np.isfinite(firsts) & np.isfinite(lasts)
    if not np.any(valid):
        return None
    if toward_high:
        first_m = origin_b + firsts * resolution
        last_m = origin_b + lasts * resolution
        outer_m = float(np.nanpercentile(last_m[valid], 95))
        inset = outer_m - first_m
    else:
        first_m = origin_b + firsts * resolution
        last_m = origin_b + lasts * resolution
        outer_m = float(np.nanpercentile(last_m[valid], 5))
        inset = first_m - outer_m
    in_band = valid & (inset >= BUILTIN_MIN_DEPTH_M) & (inset <= BUILTIN_MAX_DEPTH_M)
    if int(np.count_nonzero(in_band)) * resolution < BUILTIN_MIN_FRONT_M:
        return None
    mode_depth = float(np.median(inset[in_band]))
    front = valid & (np.abs(inset - mode_depth) <= BUILTIN_FRONT_TOL_M)
    cavity = valid & (inset < (mode_depth - BUILTIN_FRONT_TOL_M))
    if int(np.count_nonzero(front)) * resolution < BUILTIN_MIN_FRONT_M:
        return None
    front_i = np.flatnonzero(front)
    lo, hi = int(front_i.min()), int(front_i.max())
    mask = np.zeros(n_along, dtype=bool)
    mask[lo : hi + 1] = front[lo : hi + 1] | cavity[lo : hi + 1]
    if int(np.count_nonzero(mask)) * resolution < BUILTIN_MIN_LENGTH_M:
        return None
    mi = np.flatnonzero(mask)
    a0 = origin_a + mi.min() * resolution
    a1 = origin_a + (mi.max() + 1) * resolution
    if toward_high:
        b1 = float(outer_m)
        b0 = b1 - mode_depth
    else:
        b0 = float(outer_m)
        b1 = b0 + mode_depth
    if along_axis == 0:
        poly = _rect_polygon(a0, b0, a1, b1)
        side = "+y" if toward_high else "-y"
    else:
        poly = _rect_polygon(b0, a0, b1, a1)
        side = "+x" if toward_high else "-x"
    return {
        "side": side,
        "polygon": poly,
        "depth_m": float(round(mode_depth, 3)),
        "length_m": float(round((mi.max() - mi.min() + 1) * resolution, 3)),
        "open_bay": bool(np.any(cavity[lo : hi + 1])),
    }


def detect_builtin_keepouts(
    points_xy: np.ndarray, *, trajectory: np.ndarray | None = None
) -> dict:
    empty = {"strips": [], "envelope": None}
    pts = np.asarray(points_xy, dtype=float)
    if pts.ndim != 2 or len(pts) < 50:
        return empty
    if pts.shape[1] < 2:
        return empty

    p2d = pts[:, :2]
    dx = float(p2d[:, 0].max() - p2d[:, 0].min())
    dy = float(p2d[:, 1].max() - p2d[:, 1].min())
    p_dx = float(np.percentile(p2d[:, 0], 98) - np.percentile(p2d[:, 0], 2))
    p_dy = float(np.percentile(p2d[:, 1], 98) - np.percentile(p2d[:, 1], 2))
    initial_area = min(dx * dy, p_dx * p_dy)
    short_side = min(p_dx, p_dy)

    is_small_room = (
        initial_area < SMALL_ROOM_KEEPOUT_MAX_AREA_M2
        or short_side < SMALL_ROOM_KEEPOUT_MIN_SHORT_SIDE_M
    )
    if trajectory is not None and len(trajectory):
        traj_arr = np.asarray(trajectory, dtype=float)
        if traj_arr.ndim == 2 and traj_arr.shape[0] and traj_arr.shape[1] >= 2:
            t_dx = float(traj_arr[:, 0].max() - traj_arr[:, 0].min())
            t_dy = float(traj_arr[:, 1].max() - traj_arr[:, 1].min())
            traj_room_area = (t_dx + 0.50) * (t_dy + 0.50)
            traj_short_side = min(t_dx, t_dy) + 0.35
            if traj_room_area < SMALL_ROOM_KEEPOUT_MAX_AREA_M2 or traj_short_side < SMALL_ROOM_KEEPOUT_MIN_SHORT_SIDE_M:
                is_small_room = True

    if is_small_room:
        return empty

    occupied, x_min, y_min, resolution, width, height = _raster_u8(pts[:, :2])
    close_px = _odd_px(0.03, resolution, 3)
    occupied = cv2.morphologyEx(
        occupied,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (close_px, close_px)),
    )
    envelope = {
        "x_min": float(pts[:, 0].min()),
        "x_max": float(pts[:, 0].max()),
        "y_min": float(pts[:, 1].min()),
        "y_max": float(pts[:, 1].max()),
    }
    strips = []
    for along_axis, toward_high in ((0, True), (0, False), (1, True), (1, False)):
        origin_a = x_min if along_axis == 0 else y_min
        origin_b = y_min if along_axis == 0 else x_min
        strip = _keepout_strip_along_axis(
            occupied,
            origin_a,
            origin_b,
            resolution,
            along_axis=along_axis,
            toward_high=toward_high,
        )
        if strip is not None:
            strips.append(strip)
    if pts.shape[1] >= 3:
        strips = [s for s in strips if _keepout_has_high_z(s, pts)]
    return {"strips": strips, "envelope": envelope if strips else None}


def _keepout_has_high_z(strip: dict, pts_xyz: np.ndarray) -> bool:
    poly = np.asarray(strip.get("polygon"), dtype=float)
    if poly.ndim != 2 or len(poly) < 4:
        return False
    x0, y0 = float(poly[:, 0].min()), float(poly[:, 1].min())
    x1, y1 = float(poly[:, 0].max()), float(poly[:, 1].max())
    inside = (
        (pts_xyz[:, 0] >= x0)
        & (pts_xyz[:, 0] <= x1)
        & (pts_xyz[:, 1] >= y0)
        & (pts_xyz[:, 1] <= y1)
        & (pts_xyz[:, 2] >= BUILTIN_HIGH_Z_M)
    )
    return int(np.count_nonzero(inside)) >= BUILTIN_HIGH_Z_MIN_POINTS


def finishable_floor_polygon(
    room_xy: np.ndarray, keepout_polygons: list
) -> dict:
    from shapely.geometry import MultiPolygon
    from shapely.geometry import Polygon as ShapelyPolygon

    room = ShapelyPolygon(np.asarray(room_xy, dtype=float))
    if not room.is_valid:
        room = room.buffer(0)
    cut = room
    for raw in keepout_polygons:
        poly = ShapelyPolygon(np.asarray(raw, dtype=float))
        if not poly.is_valid:
            poly = poly.buffer(0)
        if poly.is_empty:
            continue
        cut = cut.difference(poly)
    if cut.is_empty:
        return {"vertices": np.asarray(room_xy, dtype=float).tolist(), "area_m2": 0.0}
    if isinstance(cut, MultiPolygon):
        cut = max(cut.geoms, key=lambda g: g.area)
    verts = [
        [float(round(x, VERTEX_DECIMALS)), float(round(y, VERTEX_DECIMALS))]
        for x, y in cut.exterior.coords[:-1]
    ]
    return {"vertices": verts, "area_m2": float(cut.area)}
 
 
def _is_flush_arm(tail_rng: tuple[float, float] | None, body_rng: tuple[float, float] | None) -> bool:
    if tail_rng is None or body_rng is None:
        return True
    return (
        abs(tail_rng[0] - body_rng[0]) <= TAIL_FLUSH_M
        or abs(tail_rng[1] - body_rng[1]) <= TAIL_FLUSH_M
    )


def _eval_trajectory_split(
    traj: np.ndarray | None, axis: int, cut_coord: float
) -> tuple[int, int, float, float, float, float]:
    """Evaluates trajectory split into region 1 (coord < cut_coord) and region 2 (coord >= cut_coord).
    Returns (tc_1, tc_2, L_1, L_2, span_1, span_2) where span is the traversal span along the cut axis.
    """
    if traj is None:
        return 0, 0, 0.0, 0.0, 0.0, 0.0
    t_arr = np.asarray(traj, dtype=float)
    if t_arr.ndim != 2 or t_arr.shape[0] == 0 or t_arr.shape[1] < 2:
        return 0, 0, 0.0, 0.0, 0.0, 0.0
    coords = t_arr[:, 0] if axis == 0 else t_arr[:, 1]
    m1 = coords < cut_coord
    m2 = coords >= cut_coord
    tc_1 = int(np.count_nonzero(m1))
    tc_2 = int(np.count_nonzero(m2))
    L_1, L_2 = 0.0, 0.0
    span_1, span_2 = 0.0, 0.0
    if len(t_arr) > 1:
        diffs = np.linalg.norm(np.diff(t_arr[:, :2], axis=0), axis=1)
        valid = diffs < 2.0
        if tc_1 > 1:
            in_1_step = m1[:-1] & m1[1:] & valid
            L_1 = float(np.sum(diffs[in_1_step]))
            span_1 = float(np.ptp(coords[m1]))
            L_1 = max(L_1, span_1)
        if tc_2 > 1:
            in_2_step = m2[:-1] & m2[1:] & valid
            L_2 = float(np.sum(diffs[in_2_step]))
            span_2 = float(np.ptp(coords[m2]))
            L_2 = max(L_2, span_2)
    return tc_1, tc_2, L_1, L_2, span_1, span_2


def _eval_split_and_pick_tail(
    trajectory: np.ndarray | None,
    axis: int,
    cut_coord: float,
    area_1: float,
    area_2: float,
) -> tuple[dict, str]:
    """Evaluates split into region 1 (coord < cut_coord) and region 2 (coord >= cut_coord),
    and determines which region is the main body ('a') vs tail ('b').
    Returns (metrics_dict, tail_dir) where tail_dir is 'suffix' if region 2 is tail,
    or 'prefix' if region 1 is tail.
    """
    tc_1, tc_2, L_1, L_2, sp_1, sp_2 = _eval_trajectory_split(trajectory, axis, cut_coord)
    if area_1 >= 1.5 * max(area_2, 0.1) and area_1 >= 1.5:
        is_1_body = True
    elif area_2 >= 1.5 * max(area_1, 0.1) and area_2 >= 1.5:
        is_1_body = False
    else:
        tot_tc = tc_1 + tc_2
        if tot_tc >= 10:
            if tc_1 >= 0.60 * tot_tc:
                is_1_body = True
            elif tc_2 >= 0.60 * tot_tc:
                is_1_body = False
            else:
                if abs(L_1 - L_2) > 0.50:
                    is_1_body = (L_1 > L_2)
                elif abs(area_1 - area_2) > 0.20:
                    is_1_body = (area_1 >= area_2)
                else:
                    is_1_body = (tc_1 >= tc_2)
        else:
            is_1_body = (area_1 >= area_2)

    if is_1_body:
        tail_dir = "suffix"
        m = {
            "area_a": area_1,
            "area_b": area_2,
            "tc_a": tc_1,
            "tc_b": tc_2,
            "L_a": L_1,
            "L_b": L_2,
            "sp_a": sp_1,
            "sp_b": sp_2,
        }
    else:
        tail_dir = "prefix"
        m = {
            "area_a": area_2,
            "area_b": area_1,
            "tc_a": tc_2,
            "tc_b": tc_1,
            "L_a": L_2,
            "L_b": L_1,
            "sp_a": sp_2,
            "sp_b": sp_1,
        }
    return m, tail_dir


def classify_occupancy_necks(
    points_xyz: np.ndarray, trajectory: np.ndarray | None = None
) -> dict:
    empty = {
        "neck_kind": "door",
        "neck_width_m": 0.0,
        "tail_length_m": 0.0,
        "area_a_m2": 0.0,
        "area_b_m2": 0.0,
        "traj_count_a": 0,
        "traj_count_b": 0,
        "start_xy": [],
        "end_xy": [],
    }
    pts = np.asarray(points_xyz, dtype=float)
    if pts.ndim != 2 or pts.shape[1] < 2 or len(pts) < 10:
        return empty

    cavity, x_min, y_min, resolution = filled_cavity_mask(pts[:, :2])
    bbox_w = float(pts[:, 0].max() - pts[:, 0].min())
    bbox_h = float(pts[:, 1].max() - pts[:, 1].min())
    bbox_area = bbox_w * bbox_h
    cav_cells = int(np.count_nonzero(cavity)) if cavity is not None else 0
    if cavity is None or cav_cells * resolution * resolution < 0.20 * max(bbox_area, 1.0):
        occupied, x_min, y_min, resolution, width, height = _raster_u8(pts[:, :2])
        bridge_px = _odd_px(DOORWAY_BRIDGE_M, resolution, 11)
        closed = cv2.morphologyEx(
            occupied,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_RECT, (bridge_px, bridge_px)),
        )
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            cavity = np.zeros_like(closed)
            cv2.drawContours(cavity, [max(contours, key=cv2.contourArea)], -1, 255, thickness=cv2.FILLED)
            cav_cells = int(np.count_nonzero(cavity))

    if cavity is None or cav_cells < 100:
        return empty

    binary = cavity > 0
    H, W = binary.shape
    candidates = []

    for axis in (0, 1):
        n_scan = W if axis == 0 else H
        spans = np.zeros(n_scan, dtype=float)
        rngs = []
        for i in range(n_scan):
            h = np.flatnonzero(binary[:, i] if axis == 0 else binary[i, :])
            if len(h):
                spans[i] = (h.max() - h.min() + 1) * resolution
                rngs.append((float(h.min() * resolution), float(h.max() * resolution)))
            else:
                rngs.append(None)

        occ = np.flatnonzero(spans > 0.05)
        if len(occ) < 10:
            continue
        lo, hi = int(occ[0]), int(occ[-1])

        max_span = float(np.max(spans[occ]))
        effective_body_span = min(BODY_SPAN_M, max(0.50, 0.45 * max_span))
        is_body = spans >= effective_body_span
        diff = np.diff(is_body.astype(int))
        starts = np.flatnonzero(diff == 1) + 1
        if is_body[0]:
            starts = np.r_[0, starts]
        ends = np.flatnonzero(diff == -1)
        if is_body[-1]:
            ends = np.r_[ends, n_scan - 1]

        runs = []
        for s, e in zip(starts, ends):
            if (e - s + 1) * resolution >= 0.30:
                runs.append((s, e))

        if len(runs) >= 2:
            for r_idx in range(len(runs) - 1):
                e_k = runs[r_idx][1]
                s_next = runs[r_idx + 1][0]
                gap_len = (s_next - e_k) * resolution
                gap_spans = spans[e_k:s_next]
                if len(gap_spans) == 0:
                    continue
                neck_idx = (e_k + s_next) // 2
                h = np.flatnonzero(binary[:, neck_idx] if axis == 0 else binary[neck_idx, :])
                if len(h) == 0:
                    continue
                w_neck = (h.max() - h.min() + 1) * resolution
                if axis == 0:
                    cut_coord = x_min + neck_idx * resolution
                    s_xy = [float(round(cut_coord, 3)), float(round(y_min + h.min() * resolution, 3))]
                    e_xy = [float(round(cut_coord, 3)), float(round(y_min + h.max() * resolution, 3))]
                    area_1 = float(round(np.count_nonzero(cavity[:, :neck_idx]) * resolution * resolution, 3))
                    area_2 = float(round(np.count_nonzero(cavity[:, neck_idx:]) * resolution * resolution, 3))
                else:
                    cut_coord = y_min + neck_idx * resolution
                    s_xy = [float(round(x_min + h.min() * resolution, 3)), float(round(cut_coord, 3))]
                    e_xy = [float(round(x_min + h.max() * resolution, 3)), float(round(cut_coord, 3))]
                    area_1 = float(round(np.count_nonzero(cavity[:neck_idx, :]) * resolution * resolution, 3))
                    area_2 = float(round(np.count_nonzero(cavity[neck_idx:, :]) * resolution * resolution, 3))

                m, tail_dir = _eval_split_and_pick_tail(trajectory, axis, cut_coord, area_1, area_2)
                dwell_ratio_b = float(m["tc_b"] / max(m["tc_a"], 1))

                candidates.append({
                    "neck_kind": "candidate",
                    "axis": axis,
                    "cut_coord": float(round(cut_coord, 3)),
                    "tail_dir": tail_dir,
                    "is_tail": False,
                    "neck_width_m": float(round(w_neck, 3)),
                    "tail_length_m": float(round(gap_len, 3)),
                    "area_a_m2": m["area_a"],
                    "area_b_m2": m["area_b"],
                    "traj_count_a": m["tc_a"],
                    "traj_count_b": m["tc_b"],
                    "traj_len_a_m": float(round(m["L_a"], 3)),
                    "traj_len_b_m": float(round(m["L_b"], 3)),
                    "traj_span_a_m": float(round(m["sp_a"], 3)),
                    "traj_span_b_m": float(round(m["sp_b"], 3)),
                    "dwell_ratio_b": float(round(dwell_ratio_b, 3)),
                    "start_xy": s_xy,
                    "end_xy": e_xy,
                })
        elif len(runs) == 1:
            b0, b1 = runs[0]
            body_hits = [rngs[i] for i in range(b0, b1 + 1) if rngs[i] is not None]
            body_rng = None
            if body_hits:
                body_rng = (min(h[0] for h in body_hits), max(h[1] for h in body_hits))

            # Prefix tail
            if (b0 - lo) * resolution >= MIN_DOOR_TAIL_LEN_M:
                prefix_spans = spans[lo:b0]
                p_occ = np.flatnonzero(prefix_spans > 0.05)
                if len(p_occ):
                    pspan = float(prefix_spans[p_occ].max())
                    h = np.flatnonzero(binary[:, b0 - 1] if axis == 0 else binary[b0 - 1, :])
                    if len(h):
                        w_neck = (h.max() - h.min() + 1) * resolution
                        if axis == 0:
                            cut_coord = x_min + b0 * resolution
                            s_xy = [float(round(cut_coord, 3)), float(round(y_min + h.min() * resolution, 3))]
                            e_xy = [float(round(cut_coord, 3)), float(round(y_min + h.max() * resolution, 3))]
                            area_1 = float(round(np.count_nonzero(cavity[:, :b0]) * resolution * resolution, 3))
                            area_2 = float(round(np.count_nonzero(cavity[:, b0:]) * resolution * resolution, 3))
                        else:
                            cut_coord = y_min + b0 * resolution
                            s_xy = [float(round(x_min + h.min() * resolution, 3)), float(round(cut_coord, 3))]
                            e_xy = [float(round(x_min + h.max() * resolution, 3)), float(round(cut_coord, 3))]
                            area_1 = float(round(np.count_nonzero(cavity[:b0, :]) * resolution * resolution, 3))
                            area_2 = float(round(np.count_nonzero(cavity[b0:, :]) * resolution * resolution, 3))

                        m, tail_dir = _eval_split_and_pick_tail(trajectory, axis, cut_coord, area_1, area_2)
                        dwell_ratio_b = float(m["tc_b"] / max(m["tc_a"], 1))
                        tlen = (b0 - lo) * resolution if tail_dir == "prefix" else (hi - b0) * resolution
                        tail_hits = [rngs[i] for i in (range(lo, b0) if tail_dir == "prefix" else range(b0, hi + 1)) if rngs[i] is not None]
                        tail_rng = (min(h[0] for h in tail_hits), max(h[1] for h in tail_hits)) if tail_hits else None
                        flush = _is_flush_arm(tail_rng, body_rng)

                        if not (flush and (m["tc_a"] < 5 or m["tc_b"] < 5)):
                            candidates.append({
                                "neck_kind": "candidate",
                                "axis": axis,
                                "cut_coord": float(round(cut_coord, 3)),
                                "tail_dir": tail_dir,
                                "is_tail": True,
                                "neck_width_m": float(round(w_neck, 3)),
                                "tail_length_m": float(round(tlen, 3)),
                                "area_a_m2": m["area_a"],
                                "area_b_m2": m["area_b"],
                                "traj_count_a": m["tc_a"],
                                "traj_count_b": m["tc_b"],
                                "traj_len_a_m": float(round(m["L_a"], 3)),
                                "traj_len_b_m": float(round(m["L_b"], 3)),
                                "traj_span_a_m": float(round(m["sp_a"], 3)),
                                "traj_span_b_m": float(round(m["sp_b"], 3)),
                                "dwell_ratio_b": float(round(dwell_ratio_b, 3)),
                                "start_xy": s_xy,
                                "end_xy": e_xy,
                            })

            # Suffix tail
            if (hi - b1) * resolution >= MIN_DOOR_TAIL_LEN_M:
                suffix_spans = spans[b1 + 1 : hi + 1]
                s_occ = np.flatnonzero(suffix_spans > 0.05)
                if len(s_occ):
                    sspan = float(suffix_spans[s_occ].max())
                    h = np.flatnonzero(binary[:, b1 + 1] if axis == 0 else binary[b1 + 1, :])
                    if len(h):
                        w_neck = (h.max() - h.min() + 1) * resolution
                        if axis == 0:
                            cut_coord = x_min + (b1 + 1) * resolution
                            s_xy = [float(round(cut_coord, 3)), float(round(y_min + h.min() * resolution, 3))]
                            e_xy = [float(round(cut_coord, 3)), float(round(y_min + h.max() * resolution, 3))]
                            area_1 = float(round(np.count_nonzero(cavity[:, :b1+1]) * resolution * resolution, 3))
                            area_2 = float(round(np.count_nonzero(cavity[:, b1+1:]) * resolution * resolution, 3))
                        else:
                            cut_coord = y_min + (b1 + 1) * resolution
                            s_xy = [float(round(x_min + h.min() * resolution, 3)), float(round(cut_coord, 3))]
                            e_xy = [float(round(x_min + h.max() * resolution, 3)), float(round(cut_coord, 3))]
                            area_1 = float(round(np.count_nonzero(cavity[:b1+1, :]) * resolution * resolution, 3))
                            area_2 = float(round(np.count_nonzero(cavity[b1+1:, :]) * resolution * resolution, 3))

                        m, tail_dir = _eval_split_and_pick_tail(trajectory, axis, cut_coord, area_1, area_2)
                        dwell_ratio_b = float(m["tc_b"] / max(m["tc_a"], 1))
                        tlen = (hi - b1) * resolution if tail_dir == "suffix" else (b1 - lo) * resolution
                        tail_hits = [rngs[i] for i in (range(b1 + 1, hi + 1) if tail_dir == "suffix" else range(lo, b1 + 1)) if rngs[i] is not None]
                        tail_rng = (min(h[0] for h in tail_hits), max(h[1] for h in tail_hits)) if tail_hits else None
                        flush = _is_flush_arm(tail_rng, body_rng)

                        if not (flush and (m["tc_a"] < 5 or m["tc_b"] < 5)):
                            candidates.append({
                                "neck_kind": "candidate",
                                "axis": axis,
                                "cut_coord": float(round(cut_coord, 3)),
                                "tail_dir": tail_dir,
                                "is_tail": True,
                                "neck_width_m": float(round(w_neck, 3)),
                                "tail_length_m": float(round(tlen, 3)),
                                "area_a_m2": m["area_a"],
                                "area_b_m2": m["area_b"],
                                "traj_count_a": m["tc_a"],
                                "traj_count_b": m["tc_b"],
                                "traj_len_a_m": float(round(m["L_a"], 3)),
                                "traj_len_b_m": float(round(m["L_b"], 3)),
                                "traj_span_a_m": float(round(m["sp_a"], 3)),
                                "traj_span_b_m": float(round(m["sp_b"], 3)),
                                "dwell_ratio_b": float(round(dwell_ratio_b, 3)),
                                "start_xy": s_xy,
                                "end_xy": e_xy,
                            })

    if not candidates:
        return empty

    # Priority 1: room_connector
    for c in candidates:
        dwell_b = c.get("dwell_ratio_b", c["traj_count_b"] / max(c["traj_count_a"], 1))
        L_b = c.get("traj_len_b_m", 0.0)
        span_b = c.get("traj_span_b_m", 1.0)
        has_kinetic_movement = (
            c["traj_count_b"] >= 5
            and L_b >= 1.50
            and span_b >= 0.80
            and dwell_b >= 0.15
        )
        if (
            has_kinetic_movement
            and c["area_a_m2"] >= SECOND_ROOM_MIN_AREA_M2
            and c["area_b_m2"] >= SECOND_ROOM_MIN_AREA_M2
        ):
            c["neck_kind"] = "room_connector"
            return c

    # Priority 2: open_corridor
    for c in candidates:
        dwell_b = c.get("dwell_ratio_b", c["traj_count_b"] / max(c["traj_count_a"], 1))
        L_b = c.get("traj_len_b_m", 0.0)
        span_b = c.get("traj_span_b_m", 1.0)
        has_kinetic_movement = (
            c["traj_count_b"] >= 5
            and L_b >= 1.50
            and span_b >= 0.80
            and dwell_b >= 0.15
        )
        is_small_room = float(c.get("area_a_m2", 0.0)) < 6.0
        is_narrow_neck = float(c.get("neck_width_m", 0.0)) <= 1.60
        if (
            has_kinetic_movement
            and c["tail_length_m"] >= OPEN_CORRIDOR_MIN_LENGTH_M
            and c["neck_width_m"] >= OPEN_CORRIDOR_MIN_WIDTH_M
            and not (is_small_room and is_narrow_neck)
        ):
            c["neck_kind"] = "open_corridor"
            return c

    # Priority 3: door
    door_cands = [c for c in candidates if c.get("is_tail", True)]
    if door_cands:
        best = max(door_cands, key=lambda c: c["neck_width_m"])
        best["neck_kind"] = "door"
        return best
    return empty


def keepout_is_interior_hole(
    keepout_poly: list | np.ndarray, room_xy: np.ndarray
) -> bool:
    try:
        k_pts = np.asarray(keepout_poly, dtype=float)
        r_pts = np.asarray(room_xy, dtype=float)
        if k_pts.ndim != 2 or len(k_pts) < 3 or r_pts.ndim != 2 or len(r_pts) < 3:
            return False

        keepout_poly_geom = Polygon(k_pts[:, :2])
        if not keepout_poly_geom.is_valid:
            keepout_poly_geom = keepout_poly_geom.buffer(0)
        room_poly_geom = Polygon(r_pts[:, :2])
        if not room_poly_geom.is_valid:
            room_poly_geom = room_poly_geom.buffer(0)

        if keepout_poly_geom.is_empty or room_poly_geom.is_empty:
            return False

        keepout_diff = keepout_poly_geom.difference(room_poly_geom)
        if keepout_diff.area / max(keepout_poly_geom.area, 1e-6) > 0.10:
            return False

        n_pts = len(r_pts)
        for i in range(n_pts):
            p1 = r_pts[i, :2]
            p2 = r_pts[(i + 1) % n_pts, :2]
            L = float(np.hypot(p2[0] - p1[0], p2[1] - p1[1]))
            if L < 1e-6:
                continue
            edge_line = LineString([p1, p2]).buffer(0.03)
            overlap = keepout_poly_geom.intersection(edge_line)
            if not overlap.is_empty and overlap.area > 1e-7:
                u = (p2 - p1) / L
                coords = []
                if hasattr(overlap, "geoms"):
                    for g in overlap.geoms:
                        if hasattr(g, "exterior"):
                            coords.extend(list(g.exterior.coords))
                        elif hasattr(g, "coords"):
                            coords.extend(list(g.coords))
                elif hasattr(overlap, "exterior"):
                    coords = list(overlap.exterior.coords)
                elif hasattr(overlap, "coords"):
                    coords = list(overlap.coords)
                if coords:
                    projs = [float(np.dot(np.array(pt[:2]) - p1, u)) for pt in coords]
                    t_min = max(0.0, min(projs))
                    t_max = min(L, max(projs))
                    overlap_len = max(0.0, t_max - t_min)
                    if overlap_len / L >= 0.50:
                        return False

        return True
    except Exception:
        return False



