"""Manhattan frame alignment and metric scaling for 3D point clouds.

Provides robust gravity/normal-constrained floor detection, floor leveling,
minimum-bounding-box-area yaw rotation search, and metric ground truth scaling.
"""
from __future__ import annotations

import logging
from typing import Tuple

import numpy as np
import open3d as o3d
from scipy.ndimage import gaussian_filter1d

logger = logging.getLogger(__name__)


def rotation_from_vectors(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Compute the 3x3 rotation matrix that aligns source vector to target vector."""
    source = np.asarray(source, dtype=float)
    target = np.asarray(target, dtype=float)
    source /= np.linalg.norm(source) + 1e-9
    target /= np.linalg.norm(target) + 1e-9

    cross = np.cross(source, target)
    dot = float(np.clip(np.dot(source, target), -1.0, 1.0))
    cross_norm = np.linalg.norm(cross)
    if cross_norm < 1e-9:
        if dot < -0.999999:
            # Vectors are opposite (180 deg rotation). Pick an orthogonal axis.
            ortho = np.array([1.0, 0.0, 0.0]) if abs(source[0]) < 0.8 else np.array([0.0, 1.0, 0.0])
            axis = np.cross(source, ortho)
            axis /= np.linalg.norm(axis)
            return 2.0 * np.outer(axis, axis) - np.eye(3)
        return np.eye(3)

    vx = np.array([
        [0.0, -cross[2], cross[1]],
        [cross[2], 0.0, -cross[0]],
        [-cross[1], cross[0], 0.0],
    ])
    return np.eye(3) + vx + vx @ vx * ((1.0 - dot) / (cross_norm ** 2))


def angle_to_axis(vector: np.ndarray) -> Tuple[int, float]:
    """Find dominant axis index (0=X, 1=Y, 2=Z) and deviation angle in degrees."""
    vector = vector / (np.linalg.norm(vector) + 1e-9)
    axis = int(np.argmax(np.abs(vector)))
    angle = np.degrees(np.arccos(np.clip(abs(vector[axis]), -1.0, 1.0)))
    return axis, float(angle)


def fit_plane_svd(points: np.ndarray) -> np.ndarray:
    """Fit plane normal to 3D point array via SVD."""
    center = points.mean(axis=0)
    _, _, vh = np.linalg.svd(points - center, full_matrices=False)
    normal = vh[-1]
    return normal / (np.linalg.norm(normal) + 1e-9)


def detect_floor_normal(
    points: np.ndarray,
    max_sample: int = 220_000,
    reference_up: np.ndarray | None = None,
    max_tilt_deg: float = 15.0,
) -> tuple:
    """Detect dominant floor plane normal using Open3D RANSAC plane segmentation.

    If reference_up is specified, only planes whose normal is within max_tilt_deg
    of reference_up will be accepted as the floor. This strictly prevents vertical walls
    or oblique planes from being mistakenly chosen as the floor.
    """
    points = np.asarray(points, dtype=float)
    if len(points) == 0:
        ref = reference_up if reference_up is not None else np.array([0.0, 0.0, 1.0])
        axis = int(np.argmax(np.abs(ref)))
        return axis, ref, 0, 0, 0.0, 0.02

    finite_mask = np.isfinite(points).all(axis=1)
    if not np.all(finite_mask):
        points = points[finite_mask]

    if len(points) < 50:
        ref = reference_up if reference_up is not None else np.array([0.0, 0.0, 1.0])
        axis = int(np.argmax(np.abs(ref)))
        return axis, ref, 0, len(points), 0.0, 0.02

    sample = points
    if len(sample) > max_sample:
        rng = np.random.default_rng(7)
        sample = sample[rng.choice(len(sample), max_sample, replace=False)]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(sample)

    diagonal = float(np.linalg.norm(np.ptp(sample, axis=0)))
    voxel_size = max(diagonal * 0.008, 0.02)
    p_down = pcd.voxel_down_sample(voxel_size)
    if len(p_down.points) < 50:
        p_down = pcd

    threshold = max(diagonal * 0.01, 0.03)

    if reference_up is not None:
        ref_vec = np.asarray(reference_up, dtype=float)
        ref_vec /= np.linalg.norm(ref_vec) + 1e-9

        # Iteratively segment up to 6 candidate planes to find the one matching reference_up
        rem = p_down
        best_candidate = None
        best_inliers = -1
        best_fallback_cand = None
        best_fallback_inliers = -1

        for _ in range(6):
            if len(rem.points) < 50:
                break
            plane_model, inlier_indices = rem.segment_plane(
                distance_threshold=threshold,
                ransac_n=3,
                num_iterations=1000,
            )
            a, b, c, d = plane_model
            norm = np.linalg.norm([a, b, c])
            if norm < 1e-9:
                break
            cand_normal = np.array([a, b, c]) / norm
            if np.dot(cand_normal, ref_vec) < 0:
                cand_normal = -cand_normal

            dot_val = float(np.clip(np.dot(cand_normal, ref_vec), -1.0, 1.0))
            cand_tilt = float(np.degrees(np.arccos(dot_val)))

            n_inl = len(inlier_indices)
            if cand_tilt <= max_tilt_deg and n_inl > best_inliers:
                best_inliers = n_inl
                best_candidate = (cand_normal, cand_tilt, n_inl)
            elif cand_tilt <= 45.0 and n_inl > best_fallback_inliers:
                best_fallback_inliers = n_inl
                best_fallback_cand = (cand_normal, cand_tilt, n_inl)

            rem = rem.select_by_index(inlier_indices, invert=True)

        chosen = best_candidate if best_candidate is not None else best_fallback_cand
        if chosen is not None:
            normal, tilt, inliers = chosen
            axis = int(np.argmax(np.abs(ref_vec)))
            return axis, normal, inliers, len(p_down.points), tilt, threshold

        # Fallback to reference_up
        axis = int(np.argmax(np.abs(ref_vec)))
        return axis, ref_vec, 0, len(p_down.points), 0.0, threshold

    # No reference_up given: find dominant plane and ensure axis tilt is valid
    plane_model, inlier_indices = p_down.segment_plane(
        distance_threshold=threshold,
        ransac_n=3,
        num_iterations=1000,
    )
    a, b, c, d = plane_model
    norm = np.linalg.norm([a, b, c])
    if norm < 1e-9:
        return 2, np.array([0.0, 0.0, 1.0]), 0, len(sample), 0.0, threshold

    normal = np.array([a, b, c]) / norm
    centroid = sample.mean(axis=0)
    floor_pt = -normal * (d / norm)
    if np.dot(centroid - floor_pt, normal) < 0:
        normal = -normal

    axis, tilt = angle_to_axis(normal)
    inliers = len(inlier_indices)
    candidates = len(p_down.points)
    return axis, normal, inliers, candidates, tilt, threshold


def yaw_rotation_matrix(up_axis: int, angle_deg: float) -> np.ndarray:
    """Create 3x3 rotation matrix for yaw angle around specified up_axis."""
    theta = np.deg2rad(angle_deg)
    c, s = np.cos(theta), np.sin(theta)
    if up_axis == 0:
        return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])
    if up_axis == 1:
        return np.array([[c, 0.0, -s], [0.0, 1.0, 0.0], [s, 0.0, c]])
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _detect_manhattan_yaw_aabb(points: np.ndarray, up_axis: int = 2, max_sample: int = 50_000) -> float:
    """Fallback: Find yaw rotation angle [-45 deg, 45 deg] that minimizes 2D bounding box area."""
    points = np.asarray(points, dtype=float)
    finite_mask = np.isfinite(points).all(axis=1)
    if not np.all(finite_mask):
        points = points[finite_mask]
    if len(points) < 10:
        return 0.0

    rng = np.random.default_rng(11)
    sample = points
    if len(sample) > max_sample:
        sample = sample[rng.choice(len(sample), max_sample, replace=False)]

    axes = [axis for axis in range(3) if axis != up_axis]
    footprint = sample[:, axes]
    best_angle = 0.0
    best_area = float("inf")

    for angle in np.linspace(-45.0, 45.0, 181):
        theta = np.deg2rad(angle)
        c, s = np.cos(theta), np.sin(theta)
        rotated = footprint @ np.array([[c, -s], [s, c]]).T
        extent = rotated.max(axis=0) - rotated.min(axis=0)
        area = float(extent[0] * extent[1])
        if area < best_area:
            best_area = area
            best_angle = float(angle)

    return float(best_angle)


def detect_manhattan_yaw(points: np.ndarray, up_axis: int = 2, max_sample: int = 220_000) -> float:
    """Find dominant wall yaw rotation angle in [-45 deg, 45 deg] to align walls with X/Y axes.

    Uses Dominant Wall Normal Consensus:
    1. Filter for vertical wall normals (|n_z| < 0.25).
    2. Compute theta = atan2(n_y, n_x) mod 90 deg.
    3. Peak detection using circular histogram with sub-bin refinement.
    4. Compute correction angle so walls align with coordinate axes (0 and 90 deg).
    """
    points = np.asarray(points, dtype=float)
    finite_mask = np.isfinite(points).all(axis=1)
    if not np.all(finite_mask):
        points = points[finite_mask]
    if len(points) < 50:
        return 0.0

    rng = np.random.default_rng(11)
    sample = points
    if len(sample) > max_sample:
        sample = sample[rng.choice(len(sample), max_sample, replace=False)]

    axes = [axis for axis in range(3) if axis != up_axis]
    x_ax, y_ax = axes[0], axes[1]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(sample)
    diagonal = float(np.linalg.norm(np.ptp(sample, axis=0)))
    voxel_size = max(diagonal * 0.005, 0.01)
    p_down = pcd.voxel_down_sample(voxel_size)
    if len(p_down.points) < 50:
        p_down = pcd

    p_down.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=30))
    normals = np.asarray(p_down.normals)

    wall_mask = np.abs(normals[:, up_axis]) < 0.25
    if np.sum(wall_mask) < 20:
        return _detect_manhattan_yaw_aabb(points, up_axis=up_axis)

    n_x = normals[wall_mask, x_ax]
    n_y = normals[wall_mask, y_ax]
    theta_raw = np.degrees(np.arctan2(n_y, n_x))
    theta_mod = theta_raw % 90.0

    n_bins = 180
    counts, edges = np.histogram(theta_mod, bins=n_bins, range=(0.0, 90.0))
    bin_centers = (edges[:-1] + edges[1:]) / 2.0

    # 3x periodic tiling for circular convolution
    counts_3x = np.tile(counts, 3)
    smoothed = gaussian_filter1d(counts_3x.astype(float), sigma=1.5)[n_bins:2 * n_bins]
    peak_idx = int(np.argmax(smoothed))
    peak_angle = float(bin_centers[peak_idx])

    # Circular median refinement around peak (+/- 3 deg)
    diff = (theta_mod - peak_angle + 45.0) % 90.0 - 45.0
    near = np.abs(diff) < 3.0
    if np.sum(near) > 0:
        refined_peak = (peak_angle + float(np.median(diff[near]))) % 90.0
    else:
        refined_peak = peak_angle

    delta = refined_peak if refined_peak <= 45.0 else (refined_peak - 90.0)
    return float(-delta)


def align_point_cloud_to_manhattan(
    pcd: o3d.geometry.PointCloud,
    R_init: np.ndarray | None = None,
    reference_up: np.ndarray = np.array([0.0, 0.0, 1.0]),
) -> tuple[o3d.geometry.PointCloud, dict]:
    """Level floor normal to vertical (+Z), align walls to X/Y axes, and set floor Z=0.

    If R_init is provided (e.g. camera-to-world rotation R_wc0), points are first
    rotated into the world frame before leveling.
    """
    work = o3d.geometry.PointCloud(pcd)
    work = work.remove_non_finite_points(remove_nan=True, remove_infinite=True)
    if len(work.points) < 50:
        return work, {"applied": False, "reason": "too_few_points"}

    points = np.asarray(work.points).copy()
    if R_init is not None:
        R_init = np.asarray(R_init, dtype=float).reshape(3, 3)
        points = points @ R_init.T

    center = points.mean(axis=0)
    up_axis, normal, inliers, candidates, floor_tilt, threshold = detect_floor_normal(
        points, reference_up=reference_up, max_tilt_deg=15.0
    )

    target = np.array([0.0, 0.0, 1.0])
    level_rotation = rotation_from_vectors(normal, target)
    leveled = (points - center) @ level_rotation.T + center

    yaw_deg = detect_manhattan_yaw(leveled, up_axis=2)
    yaw_rotation = yaw_rotation_matrix(2, yaw_deg)
    aligned = (leveled - center) @ yaw_rotation.T + center

    # Shift floor plane to Z = 0.0
    floor_z = float(np.percentile(aligned[:, 2], 1.5))
    aligned[:, 2] -= floor_z

    work.points = o3d.utility.Vector3dVector(aligned)
    if work.has_normals():
        normals = np.asarray(work.normals).copy()
        if R_init is not None:
            normals = normals @ R_init.T
        combined_rot = yaw_rotation @ level_rotation
        work.normals = o3d.utility.Vector3dVector(normals @ combined_rot.T)

    info = {
        "applied": True,
        "input_up_axis": "XYZ"[up_axis],
        "output_up_axis": "Z",
        "floor_normal_before": normal.tolist(),
        "floor_tilt_deg": float(floor_tilt),
        "floor_inliers": int(inliers),
        "yaw_correction_deg": float(yaw_deg),
        "z_floor_offset": floor_z,
    }
    return work, info


def clean_and_align_point_cloud(
    pcd: o3d.geometry.PointCloud,
    voxel: float = 0.003,
    stat_neighbors: int = 16,
    stat_std: float = 2.8,
    R_init: np.ndarray | None = None,
    reference_up: np.ndarray = np.array([0.0, 0.0, 1.0]),
) -> tuple[o3d.geometry.PointCloud, dict]:
    """Clean point cloud (non-finite removal, voxel downsampling, SOR outlier removal) and align to Manhattan axes."""
    work = o3d.geometry.PointCloud(pcd)
    work = work.remove_non_finite_points(remove_nan=True, remove_infinite=True)
    orig_count = len(work.points)

    if voxel > 0 and len(work.points) > 100:
        work = work.voxel_down_sample(voxel)
    after_voxel_count = len(work.points)

    if len(work.points) > 50:
        work, _ = work.remove_statistical_outlier(nb_neighbors=stat_neighbors, std_ratio=stat_std)
    clean_count = len(work.points)

    aligned_pcd, align_info = align_point_cloud_to_manhattan(
        work, R_init=R_init, reference_up=reference_up
    )
    align_info["original_points"] = orig_count
    align_info["after_voxel_points"] = after_voxel_count
    align_info["output_points"] = clean_count
    return aligned_pcd, align_info


def scale_point_cloud_to_metric(
    pcd: o3d.geometry.PointCloud,
    target_height_m: float | None = None,
    target_length_m: float | None = None,
    target_width_m: float | None = None,
    uniform: bool = True,
) -> o3d.geometry.PointCloud:
    """Scale and orient Manhattan-aligned point cloud to exact Ground Truth metric dimensions.

    Guarantees:
    - Point cloud is verified and aligned to Manhattan axes before scaling.
    - Long side is oriented along X, short side along Y (matching floorplan.json).
    - Preserves orthogonal 90° room geometry.
    - If uniform=True: uses isotropic Sim(3) scale s = median(s_x, s_y, s_z).
    - If uniform=False: uses axis-aligned metric scaling with ratio clamping [0.5, 2.0].
    - Origin (0, 0, 0) aligned with the room interior corner.
    """
    work = o3d.geometry.PointCloud(pcd)
    work = work.remove_non_finite_points(remove_nan=True, remove_infinite=True)
    if len(work.points) < 50:
        return work

    pts = np.asarray(work.points).copy()
    z_floor = float(np.percentile(pts[:, 2], 1.5))
    z_ceil = float(np.percentile(pts[:, 2], 98.5))
    h_unscaled = max(0.2, z_ceil - z_floor)

    # Scale factor mapping unscaled vertical extent to true metric height
    target_h = float(target_height_m) if (target_height_m is not None and target_height_m > 0) else None
    s_z = (target_h / h_unscaled) if target_h is not None else 1.0

    # Verify and enforce Manhattan yaw alignment before scaling so orthogonal corners are preserved
    residual_yaw = detect_manhattan_yaw(pts, up_axis=2)
    rotated_yaw = False
    r_yaw = np.eye(3)
    if abs(residual_yaw) > 0.2:
        r_yaw = yaw_rotation_matrix(2, residual_yaw)
        c_xy = pts.mean(axis=0)
        pts = (pts - c_xy) @ r_yaw.T + c_xy
        rotated_yaw = True

    # Orient long side along X, short side along Y (matching floorplan.json convention)
    span_x0 = float(np.percentile(pts[:, 0], 98) - np.percentile(pts[:, 0], 2))
    span_y0 = float(np.percentile(pts[:, 1], 98) - np.percentile(pts[:, 1], 2))
    rotated_90 = False
    r90 = np.eye(3)
    if span_y0 > span_x0:
        r90 = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
        pts = pts @ r90.T
        rotated_90 = True

    # Compute span and origin on X and Y
    x_p02 = float(np.percentile(pts[:, 0], 2))
    x_p98 = float(np.percentile(pts[:, 0], 98))
    span_x = max(0.2, x_p98 - x_p02)

    y_p02 = float(np.percentile(pts[:, 1], 2))
    y_p98 = float(np.percentile(pts[:, 1], 98))
    span_y = max(0.2, y_p98 - y_p02)

    s_x = (float(target_length_m) / span_x) if (target_length_m is not None and target_length_m > 0) else s_z
    s_y = (float(target_width_m) / span_y) if (target_width_m is not None and target_width_m > 0) else s_z

    if uniform:
        candidate_scales = [s for s in (s_x, s_y, s_z) if s is not None and s > 0 and np.isfinite(s)]
        s = float(np.median(candidate_scales)) if candidate_scales else 1.0
        s_x = s
        s_y = s
        s_z = s
    else:
        # Guard against unnatural distortion: clamp scale to within [0.5, 2.0] of s_z
        s_x = float(np.clip(s_x, 0.5 * s_z, 2.0 * s_z))
        s_y = float(np.clip(s_y, 0.5 * s_z, 2.0 * s_z))

    # Translate origin to (0.0, 0.0, 0.0) and scale to metric dimensions
    pts[:, 0] = (pts[:, 0] - x_p02) * s_x
    pts[:, 1] = (pts[:, 1] - y_p02) * s_y
    pts[:, 2] = (pts[:, 2] - z_floor) * s_z

    work.points = o3d.utility.Vector3dVector(pts)

    if work.has_normals():
        normals = np.asarray(work.normals).copy()
        if rotated_yaw:
            normals = normals @ r_yaw.T
        if rotated_90:
            normals = normals @ r90.T
        scale_vec = np.array([s_x, s_y, s_z], dtype=float)
        normals = normals / scale_vec
        n_norms = np.linalg.norm(normals, axis=1, keepdims=True) + 1e-9
        work.normals = o3d.utility.Vector3dVector(normals / n_norms)

    return work

