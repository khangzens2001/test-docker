"""Manhattan frame alignment and metric scaling for 3D point clouds.

Provides robust gravity/normal-constrained floor detection, floor leveling,
minimum-bounding-box-area yaw rotation search, and metric ground truth scaling.
"""
from __future__ import annotations

import csv
import json
import logging
import os
from typing import Tuple

import numpy as np
import open3d as o3d
from scipy.ndimage import gaussian_filter1d

logger = logging.getLogger(__name__)


def estimate_vggt_gravity_up(session_dir: str | None) -> np.ndarray | None:
    """Estimate the room upward gravity vector in the VGGT world frame by fusing
    multi-frame low-acceleration IMU gravity vectors across all selected keyframes."""
    if not session_dir or not os.path.isdir(session_dir):
        return None
    cam_path = os.path.join(session_dir, "vggt", "cameras.json")
    manifest_path = os.path.join(session_dir, "vggt", "keyframe_manifest.json")
    if not os.path.isfile(manifest_path):
        manifest_path = os.path.join(session_dir, "keyframes_selected", "keyframes_manifest.json")
    vio_path = os.path.join(session_dir, "processed_vio.csv")
    if not os.path.isfile(vio_path):
        for alt_vio in ("processed_vio_raw.csv", "processed_odometry.csv", "odometry.csv"):
            cand_vio = os.path.join(session_dir, alt_vio)
            if os.path.isfile(cand_vio):
                vio_path = cand_vio
                break
    imu_path = os.path.join(session_dir, "processed_phone_imu.csv")
    if not os.path.isfile(imu_path):
        imu_path = os.path.join(session_dir, "phone_imu.csv")
    if not (
        os.path.isfile(cam_path)
        and os.path.isfile(manifest_path)
        and os.path.isfile(vio_path)
        and os.path.isfile(imu_path)
    ):
        return None
    try:
        with open(cam_path, "r", encoding="utf-8") as f:
            cams = json.load(f).get("cameras", [])
        with open(manifest_path, "r", encoding="utf-8") as f:
            frames = json.load(f).get("frames", [])
        if not cams or not frames:
            return None

        fn_to_src: dict[str, int] = {}
        for fr in frames:
            fn = fr.get("vggt_filename")
            sidx = fr.get("source_frame_index")
            if sidx is None and "source_filename" in fr:
                try:
                    sidx = int(str(fr["source_filename"]).split(".")[0])
                except (ValueError, TypeError):
                    sidx = None
            if fn is not None and sidx is not None:
                fn_to_src[str(fn)] = int(sidx)

        src_to_ts: dict[int, int] = {}
        with open(vio_path, "r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                k_fr = "frame_index" if "frame_index" in row else "frame"
                if k_fr not in row:
                    continue
                if "device_timestamp_ns" in row:
                    src_to_ts[int(float(row[k_fr]))] = int(float(row["device_timestamp_ns"]))
                elif "timestamp_nanos" in row:
                    src_to_ts[int(float(row[k_fr]))] = int(float(row["timestamp_nanos"]))
                elif "timestamp" in row:
                    src_to_ts[int(float(row[k_fr]))] = int(round(float(row["timestamp"]) * 1e9))

        imu_ts_list: list[int] = []
        imu_acc_list: list[list[float]] = []
        with open(imu_path, "r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                ts_k = (
                    "timestamp_ns"
                    if "timestamp_ns" in row
                    else ("timestamp_nanos" if "timestamp_nanos" in row else "device_timestamp_ns")
                )
                ax_k = "ax_m_s2" if "ax_m_s2" in row else "ax"
                ay_k = "ay_m_s2" if "ay_m_s2" in row else "ay"
                az_k = "az_m_s2" if "az_m_s2" in row else "az"
                imu_ts_list.append(int(float(row[ts_k])))
                imu_acc_list.append(
                    [float(row[ax_k]), float(row[ay_k]), float(row[az_k])]
                )
        if not imu_ts_list:
            return None

        imu_ts = np.asarray(imu_ts_list, dtype=np.int64)
        imu_acc = np.asarray(imu_acc_list, dtype=float)
        r_cam_imu = np.diag([1.0, -1.0, -1.0])

        up_world_samples: list[np.ndarray] = []
        for cam in cams:
            fn = str(cam.get("vggt_filename", ""))
            sidx = fn_to_src.get(fn)
            if sidx is None or sidx not in src_to_ts:
                continue
            ts = src_to_ts[sidx]
            idx = int(np.argmin(np.abs(imu_ts - ts)))
            g_imu = imu_acc[idx]
            g_norm = float(np.linalg.norm(g_imu))
            if not (8.0 <= g_norm <= 11.5):
                continue
            g_unit = g_imu / g_norm
            ext = np.asarray(cam["extrinsic_3x4"], dtype=float).reshape(3, 4)
            r_cw = ext[:, :3]
            up_world = r_cw.T @ (r_cam_imu @ g_unit)
            u_norm = float(np.linalg.norm(up_world))
            if u_norm > 1e-6:
                up_world_samples.append(up_world / u_norm)

        if len(up_world_samples) < 3:
            return None
        up_arr = np.asarray(up_world_samples, dtype=float)
        med_up = np.median(up_arr, axis=0)
        med_norm = float(np.linalg.norm(med_up))
        if med_norm < 1e-6:
            return None
        return med_up / med_norm
    except Exception:
        logger.debug("estimate_vggt_gravity_up failed for %s", session_dir, exc_info=True)
        return None


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

    If reference_up is specified, planes near reference_up (floor/ceiling) and planes
    orthogonal to reference_up (vertical walls) are jointly used to determine the exact
    vertical normal while rejecting oblique surfaces.
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
    o3d.utility.random.seed(42)

    if reference_up is not None:
        ref_vec = np.asarray(reference_up, dtype=float)
        ref_vec /= np.linalg.norm(ref_vec) + 1e-9

        rem = p_down
        best_candidate = None
        best_inliers = -1
        best_fallback_cand = None
        best_fallback_inliers = -1
        wall_normals: list[np.ndarray] = []
        wall_weights: list[float] = []
        extracted_planes: list[tuple[np.ndarray, int]] = []

        for _ in range(12):
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
            if n_inl >= 50:
                extracted_planes.append((cand_normal, n_inl))
            if cand_tilt <= max(max_tilt_deg, 18.0) and n_inl > best_inliers:
                best_inliers = n_inl
                best_candidate = (cand_normal, cand_tilt, n_inl)
            elif cand_tilt <= 45.0 and n_inl > best_fallback_inliers:
                best_fallback_inliers = n_inl
                best_fallback_cand = (cand_normal, cand_tilt, n_inl)
            if abs(cand_tilt - 90.0) <= 25.0 and n_inl >= 50:
                wall_normals.append(cand_normal)
                wall_weights.append(float(n_inl))

            rem = rem.select_by_index(inlier_indices, invert=True)

        chosen = best_candidate
        if chosen is None and len(wall_normals) >= 2:
            W_mat = np.zeros((3, 3), dtype=float)
            for wn, ww in zip(wall_normals, wall_weights):
                W_mat += ww * np.outer(wn, wn)
            eigvals, eigvecs = np.linalg.eigh(W_mat)
            ortho_up = eigvecs[:, 0]
            if np.dot(ortho_up, ref_vec) < 0:
                ortho_up = -ortho_up
            ortho_up /= np.linalg.norm(ortho_up) + 1e-9
            dot_val = float(np.clip(np.dot(ortho_up, ref_vec), -1.0, 1.0))
            ortho_tilt = float(np.degrees(np.arccos(dot_val)))
            if ortho_tilt <= 45.0:
                chosen = (ortho_up, ortho_tilt, int(sum(wall_weights)))
        if chosen is None:
            chosen = best_fallback_cand

        if chosen is not None:
            base_normal, base_tilt, base_inliers = chosen
            # Jointly refine vertical normal using all horizontalplanes and orthogonal vertical walls
            M_joint = np.zeros((3, 3), dtype=float)
            n_joint_planes = 0
            for p_norm, p_inl in extracted_planes:
                ang = float(np.degrees(np.arccos(np.clip(abs(np.dot(p_norm, base_normal)), 0.0, 1.0))))
                if ang <= 15.0:
                    M_joint += float(p_inl) * np.outer(p_norm, p_norm)
                    n_joint_planes += 1
                elif abs(ang - 90.0) <= 15.0:
                    M_joint -= float(p_inl) * np.outer(p_norm, p_norm)
                    n_joint_planes += 1
            if n_joint_planes >= 2:
                _, eigvecs_j = np.linalg.eigh(M_joint)
                joint_up = eigvecs_j[:, -1]
                if np.dot(joint_up, ref_vec) < 0:
                    joint_up = -joint_up
                joint_up /= np.linalg.norm(joint_up) + 1e-9
                if np.dot(joint_up, base_normal) >= np.cos(np.radians(8.0)):
                    base_normal = joint_up
                    base_tilt = float(
                        np.degrees(np.arccos(np.clip(np.dot(base_normal, ref_vec), -1.0, 1.0)))
                    )

            axis = int(np.argmax(np.abs(ref_vec)))
            return axis, base_normal, base_inliers, len(p_down.points), base_tilt, threshold

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

    Uses Dominant Wall Normal Consensus combined with 2D mid-height wall-slice
    bounding-box sharpness verification so curved interior fixtures cannot bias the wall yaw.
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
    yaw_norm = float(-delta)

    # Verify against 2D mid-height wall-slice bounding-box sharpness when vertical span is present
    try:
        z_vals = sample[:, up_axis]
        z_lo = float(np.percentile(z_vals, 2))
        z_hi = float(np.percentile(z_vals, 98))
        dz = z_hi - z_lo
        if dz > 0.30:
            mid_mask = (z_vals >= z_lo + 0.20 * dz) & (z_vals <= z_lo + 0.85 * dz)
            slice_xy = sample[mid_mask][:, [x_ax, y_ax]] if int(np.count_nonzero(mid_mask)) >= 200 else sample[:, [x_ax, y_ax]]
        else:
            slice_xy = sample[:, [x_ax, y_ax]]
        if len(slice_xy) >= 200:
            from app.services.occupancy_layout import estimate_manhattan_angle_bbox_sharpness
            sharp_info = estimate_manhattan_angle_bbox_sharpness(slice_xy)
            yaw_sharp = -float(sharp_info.get("angle_deg", 0.0))
            if abs(yaw_sharp - yaw_norm) > 2.0:
                def _bbox_area_at(ang_deg: float) -> float:
                    th = np.deg2rad(ang_deg)
                    c, s = float(np.cos(th)), float(np.sin(th))
                    rot = slice_xy @ np.array([[c, -s], [s, c]], dtype=float).T
                    sp = np.percentile(rot, 98, axis=0) - np.percentile(rot, 2, axis=0)
                    return float(max(sp[0], 1e-3) * max(sp[1], 1e-3))

                area_norm = _bbox_area_at(yaw_norm)
                area_sharp = _bbox_area_at(yaw_sharp)
                if area_sharp < area_norm * 0.96:
                    return float(yaw_sharp)
    except Exception:
        pass

    return yaw_norm


def align_point_cloud_to_manhattan(
    pcd: o3d.geometry.PointCloud,
    R_init: np.ndarray | None = None,
    reference_up: np.ndarray = np.array([0.0, 0.0, 1.0]),
    session_dir: str | None = None,
) -> tuple[o3d.geometry.PointCloud, dict]:
    """Level floor normal to vertical (+Z), align walls to X/Y axes, and set floor Z=0.

    If R_init is provided (e.g. camera-to-world rotation R_wc0), points are first
    rotated into the world frame before leveling. If session_dir is provided,
    multi-frame IMU gravity consensus is used to initialize reference_up.
    """
    work = o3d.geometry.PointCloud(pcd)
    work = work.remove_non_finite_points(remove_nan=True, remove_infinite=True)
    if len(work.points) < 50:
        return work, {"applied": False, "reason": "too_few_points"}

    points = np.asarray(work.points).copy()
    if R_init is not None:
        R_init = np.asarray(R_init, dtype=float).reshape(3, 3)
        points = points @ R_init.T

    imu_up = estimate_vggt_gravity_up(session_dir) if session_dir else None
    if imu_up is not None:
        if R_init is not None:
            imu_up = R_init @ imu_up
        reference_up = imu_up

    center = points.mean(axis=0)
    target = np.array([0.0, 0.0, 1.0])
    if imu_up is not None:
        from app.services.occupancy_layout import estimate_manhattan_angle_bbox_sharpness
        normal = imu_up / (np.linalg.norm(imu_up) + 1e-9)
        up_axis, floor_tilt = angle_to_axis(normal)
        inliers = len(points)
        level_rotation = rotation_from_vectors(normal, target)
        leveled = (points - center) @ level_rotation.T + center
        for ax in (0, 1):
            info_ax = estimate_manhattan_angle_bbox_sharpness(leveled[:, [ax, 2]])
            ang_a = float(info_ax.get("angle_deg", 0.0))
            if abs(ang_a) >= 45.0:
                ang_a = ang_a - 90.0 if ang_a > 0 else ang_a + 90.0
            if abs(ang_a) <= 15.0 and float(info_ax.get("bbox_area_improvement", 0.0)) >= 0.02:
                rad = np.radians(ang_a)
                ca, sa = float(np.cos(rad)), float(np.sin(rad))
                R_a = (
                    np.array([[ca, 0.0, -sa], [0.0, 1.0, 0.0], [sa, 0.0, ca]], dtype=float)
                    if ax == 0
                    else np.array([[1.0, 0.0, 0.0], [0.0, ca, -sa], [0.0, sa, ca]], dtype=float)
                )
                leveled = (leveled - center) @ R_a.T + center
                level_rotation = R_a @ level_rotation
    else:
        up_axis, normal, inliers, candidates, floor_tilt, threshold = detect_floor_normal(
            points, reference_up=reference_up, max_tilt_deg=15.0
        )
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
        "imu_gravity_used": bool(imu_up is not None),
    }
    return work, info


def clean_and_align_point_cloud(
    pcd: o3d.geometry.PointCloud,
    voxel: float = 0.003,
    stat_neighbors: int = 16,
    stat_std: float = 2.8,
    R_init: np.ndarray | None = None,
    reference_up: np.ndarray = np.array([0.0, 0.0, 1.0]),
    session_dir: str | None = None,
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
        work, R_init=R_init, reference_up=reference_up, session_dir=session_dir
    )
    align_info["original_points"] = orig_count
    align_info["after_voxel_points"] = after_voxel_count
    align_info["output_points"] = clean_count
    return aligned_pcd, align_info


