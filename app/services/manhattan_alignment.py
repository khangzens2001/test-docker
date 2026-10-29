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
            cam_doc = json.load(f)
        cams = cam_doc.get("cameras", [])
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
                if k_fr not in row or row[k_fr] in (None, ""):
                    continue
                try:
                    fr_val = int(float(row[k_fr]))
                except (ValueError, TypeError):
                    continue
                if fr_val < 0:
                    continue
                if "device_timestamp_ns" in row and row["device_timestamp_ns"] not in (None, ""):
                    src_to_ts[fr_val] = int(float(row["device_timestamp_ns"]))
                elif "timestamp_nanos" in row and row["timestamp_nanos"] not in (None, ""):
                    src_to_ts[fr_val] = int(float(row["timestamp_nanos"]))
                elif "timestamp" in row and row["timestamp"] not in (None, ""):
                    t_raw = float(row["timestamp"])
                    src_to_ts[fr_val] = int(round(t_raw if abs(t_raw) > 1e11 else t_raw * 1e9))

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
        if cam_doc.get("image_orientation") == "upright":
            r_candidates = (np.diag([1.0, -1.0, -1.0]),)
        else:
            r_candidates = (
                np.diag([1.0, -1.0, -1.0]),
                np.array([[0.0, -1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, -1.0]], dtype=float),
            )

        best_up: np.ndarray | None = None
        best_coherence = -1.0
        for r_cam_imu in r_candidates:
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
                continue
            up_arr = np.asarray(up_world_samples, dtype=float)
            med_up = np.median(up_arr, axis=0)
            med_norm = float(np.linalg.norm(med_up))
            if med_norm < 1e-6:
                continue
            mean_norm = float(np.linalg.norm(np.mean(up_arr, axis=0)))
            coherence = 0.5 * (med_norm + mean_norm)
            if coherence > best_coherence + 0.03:
                best_coherence = coherence
                best_up = med_up / med_norm

        npz_path = os.path.join(session_dir, "vggt", "project_point_cloud.npz")
        if best_up is not None and os.path.isfile(npz_path):
            try:
                with np.load(npz_path) as npz_data:
                    pts_npz = (
                        np.asarray(npz_data["points"], dtype=float)
                        if "points" in npz_data
                        else np.empty((0, 3))
                    )
                if len(pts_npz) >= 500:
                    _, refined_up, n_inl, _, _, _ = detect_floor_normal(
                        pts_npz, reference_up=best_up, max_tilt_deg=42.0
                    )
                    if n_inl >= 200 and float(np.linalg.norm(refined_up)) > 1e-6:
                        best_up = refined_up / float(np.linalg.norm(refined_up))
            except Exception:
                pass

        return best_up
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
            if eigvals[1] > 0.10 * max(eigvals[2], 1e-9):
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


def detect_manhattan_yaw(points: np.ndarray, up_axis: int = 2, max_sample: int = 250_000) -> float:
    """Find dominant wall yaw rotation angle in [-45 deg, 45 deg] to align walls with X/Y axes.

    Uses SO(2)-equivariant and scale-invariant KNN-30 surface normal estimation and
    continuous circular mode refinement on vertical wall normals.
    """
    points = np.asarray(points, dtype=float)
    finite_mask = np.isfinite(points).all(axis=1)
    if not np.all(finite_mask):
        points = points[finite_mask]
    if len(points) < 50:
        return 0.0

    axes = [axis for axis in range(3) if axis != up_axis]
    x_ax, y_ax = axes[0], axes[1]

    sample = points
    c_xy = np.mean(sample[:, [x_ax, y_ax]], axis=0)
    r2 = (sample[:, x_ax] - c_xy[0]) ** 2 + (sample[:, y_ax] - c_xy[1]) ** 2
    r_rms = float(np.sqrt(np.mean(r2))) if len(r2) else 1.05
    s_xy = max(0.25, r_rms / 1.05)
    if len(sample) > max_sample:
        order = np.lexsort((sample[:, up_axis], np.round(r2, 8)))
        idx = np.linspace(0, len(sample) - 1, max_sample, dtype=int)
        sample = sample[order[idx]]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(sample)
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.12 * s_xy, max_nn=min(30, len(sample)))
    )
    normals = np.asarray(pcd.normals, dtype=float)

    wall_mask = np.abs(normals[:, up_axis]) < 0.25
    if int(np.count_nonzero(wall_mask)) < 20:
        return _detect_manhattan_yaw_aabb(points, up_axis=up_axis)

    n_x = normals[wall_mask, x_ax]
    n_y = normals[wall_mask, y_ax]
    theta_mod = np.degrees(np.arctan2(n_y, n_x)) % 90.0

    n_bins = 3600
    counts, edges = np.histogram(theta_mod, bins=n_bins, range=(0.0, 90.0))
    bin_centers = 0.5 * (edges[:-1] + edges[1:])
    counts_f = counts.astype(float)

    smoothed = gaussian_filter1d(counts_f, sigma=60.0, mode="wrap")
    peak = float(bin_centers[int(np.argmax(smoothed))])

    for _ in range(5):
        d = (bin_centers - peak + 45.0) % 90.0 - 45.0
        w = counts_f * np.exp(-0.5 * (d / 2.0) ** 2) * (np.abs(d) <= 4.5)
        w_sum = float(np.sum(w))
        if w_sum <= 1e-9:
            break
        peak = (peak + float(np.sum(w * d) / w_sum)) % 90.0

    delta = peak if peak <= 45.0 else (peak - 90.0)
    return float(-delta)


def align_point_cloud_to_manhattan(
    pcd: o3d.geometry.PointCloud,
    R_init: np.ndarray | None = None,
    reference_up: np.ndarray | None = None,
    gravity: np.ndarray | None = None,
    session_dir: str | None = None,
) -> tuple[o3d.geometry.PointCloud, dict]:
    """Level floor normal to vertical (+Z), align walls to X/Y axes, and set floor Z=0.

    If R_init is provided (e.g. camera-to-world rotation R_wc0), points are first
    rotated into the world frame before leveling. If session_dir is provided,
    multi-frame IMU gravity consensus is used to initialize reference_up.
    If gravity is provided, reference_up is derived from -gravity.
    """
    work = o3d.geometry.PointCloud(pcd)
    work = work.remove_non_finite_points(remove_nan=True, remove_infinite=True)
    if len(work.points) < 50:
        return work, {"applied": False, "reason": "too_few_points"}

    if gravity is not None and reference_up is None:
        g = np.asarray(gravity, dtype=float)
        g_norm = float(np.linalg.norm(g))
        if g_norm > 1e-6:
            reference_up = -g / g_norm

    if reference_up is None:
        reference_up = np.array([0.0, 0.0, 1.0])

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
        _, normal_3d, inl_3d, _, _, _ = detect_floor_normal(
            points, reference_up=imu_up, max_tilt_deg=42.0
        )
        if inl_3d >= 200 and float(np.linalg.norm(normal_3d)) > 1e-6:
            normal = normal_3d / (np.linalg.norm(normal_3d) + 1e-9)
            inliers = int(inl_3d)
            max_sec_tilt = 2.0
        else:
            normal = imu_up / (np.linalg.norm(imu_up) + 1e-9)
            inliers = len(points)
            max_sec_tilt = 20.0
        up_axis, floor_tilt = angle_to_axis(normal)
        level_rotation = rotation_from_vectors(normal, target)
        leveled = (points - center) @ level_rotation.T + center
        yaw_deg = detect_manhattan_yaw(leveled, up_axis=2)
        yaw_rotation = yaw_rotation_matrix(2, yaw_deg)
        aligned = (leveled - center) @ yaw_rotation.T + center
        for ax in (0, 1):
            info_ax = estimate_manhattan_angle_bbox_sharpness(aligned[:, [ax, 2]])
            ang_a = float(info_ax.get("angle_deg", 0.0))
            if abs(ang_a) >= 45.0:
                ang_a = ang_a - 90.0 if ang_a > 0 else ang_a + 90.0
            impr_a = float(info_ax.get("bbox_area_improvement", 0.0))
            if abs(ang_a) <= max_sec_tilt and impr_a >= 0.02:
                rad = np.radians(ang_a)
                ca, sa = float(np.cos(rad)), float(np.sin(rad))
                R_a = (
                    np.array([[ca, 0.0, -sa], [0.0, 1.0, 0.0], [sa, 0.0, ca]], dtype=float)
                    if ax == 0
                    else np.array([[1.0, 0.0, 0.0], [0.0, ca, -sa], [0.0, sa, ca]], dtype=float)
                )
                aligned = (aligned - center) @ R_a.T + center
                yaw_rotation = R_a @ yaw_rotation
    else:
        up_axis, normal, inliers, candidates, floor_tilt, threshold = detect_floor_normal(
            points, reference_up=reference_up, max_tilt_deg=42.0
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
    reference_up: np.ndarray | None = None,
    gravity: np.ndarray | None = None,
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
        work,
        R_init=R_init,
        reference_up=reference_up,
        gravity=gravity,
        session_dir=session_dir,
    )
    align_info["original_points"] = orig_count
    align_info["after_voxel_points"] = after_voxel_count
    align_info["output_points"] = clean_count
    return aligned_pcd, align_info


def run_manhattan_alignment(
    pcd: o3d.geometry.PointCloud,
    gravity: np.ndarray | None = None,
    reference_up: np.ndarray | None = None,
    session_dir: str | None = None,
    pose_source: str | None = None,
    camera_frame_convention: str | None = None,
    R_init: np.ndarray | None = None,
) -> tuple[o3d.geometry.PointCloud, dict]:
    """Gravity-aware Manhattan frame alignment for 3D point cloud.

    Accepts explicit gravity vector or reference_up, or infers reference_up from
    camera_frame_convention / pose_source (e.g. OpenGL/ARCore: Y-up, OpenCV/ROS: Z-up).
    """
    if reference_up is None:
        if gravity is not None:
            g = np.asarray(gravity, dtype=float)
            g_norm = float(np.linalg.norm(g))
            if g_norm > 1e-6:
                reference_up = -g / g_norm
        elif camera_frame_convention in ("opengl", "arcore") or pose_source in ("opengl", "arcore"):
            reference_up = np.array([0.0, 1.0, 0.0])
        elif camera_frame_convention in ("opencv", "ros") or pose_source in ("opencv", "ros"):
            reference_up = np.array([0.0, 0.0, 1.0])
        else:
            reference_up = np.array([0.0, 0.0, 1.0])

    return align_point_cloud_to_manhattan(
        pcd,
        R_init=R_init,
        reference_up=reference_up,
        gravity=gravity,
        session_dir=session_dir,
    )


