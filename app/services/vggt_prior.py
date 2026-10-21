"""VGGT topology prior: Sim(3) camera-centre alignment, T2 LiDAR snap, artifact load."""
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd

from app.services.occupancy_layout import (
    BBOX_FILL_RATIO_THRESHOLD,
    DOORWAY_BRIDGE_M,
    MIN_L_SHAPE_FILL_RATIO,
    _clip_doorway_profile_tails,
    _filled_bbox_ratio,
    _raster_u8,
    apply_hough_yaw,
    close_orthogonal_polygon,
    estimate_manhattan_angle_bbox_sharpness,
    filled_bbox_fill_ratio,
    fit_missing_corner_l_shape,
    keep_largest_occupancy_component,
    slice_wall_band,
)

logger = logging.getLogger(__name__)

N_VGGT_MAX = 44
SIM3_RMSE_MAX_M = 0.08
SIM3_SCALE_MIN = 0.2
SIM3_SCALE_MAX = 5.0
SIM3_MIN_CAMERAS = 3
SIM3_MIN_RMS_RADIUS_M = 0.15
SIM3_SVD_COLLINEAR_RATIO = 0.05
T2_WINDOW_M = 0.35
T2_BIN_M = 0.05
T2_PEAK_MIN_COUNT = 30
T2_CLIP_MARGIN_M = 0.08
VGGT_MIN_POLYGON_AREA_M2 = 0.50
YAW_BINS = 8
VGGT_SKIP_REASONS = (
    "missing_artifacts",
    "invalid_npz",
    "missing_poses",
    "too_few_correspondences",
    "degenerate_sim3",
    "sim3_rmse",
    "sim3_scale",
    "degenerate_polygon",
    "unsupported_shape",
    "exception",
)


@dataclass
class Sim3Result:
    ok: bool
    s: float | None
    R: np.ndarray | None
    t: np.ndarray | None
    rmse: float | None
    n: int
    reason: str | None = None


@dataclass
class WallHint:
    n: np.ndarray
    pos_hint: float
    p0: np.ndarray
    p1: np.ndarray


@dataclass
class T2Wall:
    n: np.ndarray
    pos_hint: float
    pos_metric: float
    source: str


@dataclass
class T2Result:
    walls: list[T2Wall]
    clipped_xy: np.ndarray
    n_fallback_hints: int
    all_walls_hint: bool


@dataclass
class WallAxesResult:
    hints: list[WallHint] | None
    skip_reason: str | None = None
    leveled_height_m: float | None = None


@dataclass
class VggtPrior:
    vggt_filenames: list[str]
    t_vio: np.ndarray
    C_vggt: np.ndarray
    points: np.ndarray
    confidence: np.ndarray | None = None
    frame: str = "vggt_prediction"
    R0: np.ndarray | None = None
    colors: np.ndarray | None = None
    clean_ply_path: str | None = None
    imu_leveled_h_vggt: float | None = None
    imu_leveled_w_vggt: float | None = None


@dataclass
class VggtPriorResult:
    prior: VggtPrior | None
    skip_reason: str | None = None


def apply_sim3(points: np.ndarray, s: float, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=float).reshape(-1, 3)
    return (float(s) * (pts @ np.asarray(R, dtype=float).T)) + np.asarray(t, dtype=float).reshape(3)


def occupancy_frame_from_vio(
    points_vio: np.ndarray, r_align: np.ndarray, level_frame: dict
) -> np.ndarray:
    pts = np.asarray(points_vio, dtype=float).reshape(-1, 3)
    r_align = np.asarray(r_align, dtype=float).reshape(3, 3)
    r_floor = np.asarray(level_frame.get("rotation_3x3", np.eye(3)), dtype=float).reshape(3, 3)
    t_floor = np.asarray(level_frame.get("translation", [0.0, 0.0, 0.0]), dtype=float).reshape(3)
    work = pts @ r_align.T
    return (work @ r_floor.T) + t_floor


def _empty_sim3(n: int, reason: str) -> Sim3Result:
    return Sim3Result(ok=False, s=None, R=None, t=None, rmse=None, n=int(n), reason=reason)


def _umeyama_sim3(src: np.ndarray, dst: np.ndarray) -> tuple[float, np.ndarray, np.ndarray] | None:
    n = len(src)
    if n < 3:
        return None
    mu_src = src.mean(axis=0)
    mu_dst = dst.mean(axis=0)
    src_c = src - mu_src
    dst_c = dst - mu_dst
    var_src = float(np.sum(src_c ** 2) / float(n))
    if var_src < 1e-18:
        return None
    cov = (dst_c.T @ src_c) / float(n)
    U, sing, Vt = np.linalg.svd(cov)
    D = np.eye(3)
    if float(np.linalg.det(U) * np.linalg.det(Vt)) < 0.0:
        D[2, 2] = -1.0
    R = U @ D @ Vt
    s = float(np.trace(np.diag(sing) @ D) / var_src)
    t = mu_dst - s * (R @ mu_src)
    return s, R, t


def estimate_sim3_camera_centers(src: np.ndarray, dst: np.ndarray) -> Sim3Result:
    src = np.asarray(src, dtype=float)
    dst = np.asarray(dst, dtype=float)
    if src.ndim != 2 or dst.ndim != 2 or src.shape[1] != 3 or dst.shape[1] != 3:
        return _empty_sim3(0, "too_few_correspondences")
    n = min(src.shape[0], dst.shape[0])
    src = src[:n]
    dst = dst[:n]
    if n < SIM3_MIN_CAMERAS:
        return _empty_sim3(n, "too_few_correspondences")
    if not (np.all(np.isfinite(src)) and np.all(np.isfinite(dst))):
        return _empty_sim3(n, "degenerate_sim3")

    mu_src = src.mean(axis=0)
    mu_dst = dst.mean(axis=0)
    src_c = src - mu_src
    dst_c = dst - mu_dst

    rms_dst = float(np.sqrt(np.mean(np.sum(dst_c ** 2, axis=1))))
    if rms_dst < SIM3_MIN_RMS_RADIUS_M:
        return _empty_sim3(n, "degenerate_sim3")

    _, S_pts, _ = np.linalg.svd(src_c, full_matrices=False)
    if S_pts.size < 2 or S_pts[0] < 1e-12 or (S_pts[1] / S_pts[0]) < SIM3_SVD_COLLINEAR_RATIO:
        return _empty_sim3(n, "degenerate_sim3")

    fit_full = _umeyama_sim3(src, dst)
    if fit_full is None:
        return _empty_sim3(n, "degenerate_sim3")
    s, R, t = fit_full
    aligned = apply_sim3(src, s, R, t)
    rmse = float(np.sqrt(np.mean(np.sum((dst - aligned) ** 2, axis=1))))

    if np.isfinite(s) and SIM3_SCALE_MIN <= s <= SIM3_SCALE_MAX and np.isfinite(rmse) and rmse <= SIM3_RMSE_MAX_M:
        return Sim3Result(ok=True, s=s, R=R, t=t, rmse=rmse, n=n, reason=None)

    # RANSAC Umeyama: if full set exceeds RMSE_MAX (e.g. drifting outlier cameras),
    # find consensus inlier set (keeping 70-85% cameras) and refit.
    if n >= 6:
        rng = np.random.default_rng(42)
        n_iters = 300
        k = 4
        min_inliers = max(SIM3_MIN_CAMERAS, int(np.ceil(0.70 * n)))
        inlier_thresh = SIM3_RMSE_MAX_M
        best_inliers = None
        best_score = (-1, float("inf"))

        for _ in range(n_iters):
            samp = rng.choice(n, size=k, replace=False)
            s_mu = src[samp].mean(axis=0)
            _, svals, _ = np.linalg.svd(src[samp] - s_mu)
            if svals.size < 2 or svals[0] < 1e-6 or (svals[1] / svals[0]) < SIM3_SVD_COLLINEAR_RATIO:
                continue
            fit_samp = _umeyama_sim3(src[samp], dst[samp])
            if fit_samp is None:
                continue
            s_c, R_c, t_c = fit_samp
            if not (SIM3_SCALE_MIN <= s_c <= SIM3_SCALE_MAX):
                continue
            dists = np.linalg.norm(dst - apply_sim3(src, s_c, R_c, t_c), axis=1)
            inl = np.flatnonzero(dists <= inlier_thresh)
            n_inl = len(inl)
            if n_inl >= min_inliers:
                sse = float(np.sum(dists[inl] ** 2))
                if (n_inl > best_score[0]) or (n_inl == best_score[0] and sse < best_score[1]):
                    best_score = (n_inl, sse)
                    best_inliers = inl

        if best_inliers is not None and len(best_inliers) >= min_inliers:
            curr_inliers = best_inliers.copy()
            fit_refit = _umeyama_sim3(src[curr_inliers], dst[curr_inliers])
            if fit_refit is not None:
                s_ref, R_ref, t_ref = fit_refit
                dists_ref = np.linalg.norm(dst[curr_inliers] - apply_sim3(src[curr_inliers], s_ref, R_ref, t_ref), axis=1)
                rmse_ref = float(np.sqrt(np.mean(dists_ref ** 2)))
                while rmse_ref > SIM3_RMSE_MAX_M and len(curr_inliers) > int(np.ceil(0.75 * n)):
                    worst = int(np.argmax(dists_ref))
                    curr_inliers = np.delete(curr_inliers, worst)
                    fit_refit = _umeyama_sim3(src[curr_inliers], dst[curr_inliers])
                    if fit_refit is None:
                        break
                    s_ref, R_ref, t_ref = fit_refit
                    dists_ref = np.linalg.norm(dst[curr_inliers] - apply_sim3(src[curr_inliers], s_ref, R_ref, t_ref), axis=1)
                    rmse_ref = float(np.sqrt(np.mean(dists_ref ** 2)))

                if SIM3_SCALE_MIN <= s_ref <= SIM3_SCALE_MAX and rmse_ref <= SIM3_RMSE_MAX_M:
                    return Sim3Result(ok=True, s=s_ref, R=R_ref, t=t_ref, rmse=rmse_ref, n=len(curr_inliers), reason=None)

    if not np.isfinite(s) or s < SIM3_SCALE_MIN or s > SIM3_SCALE_MAX:
        return Sim3Result(ok=False, s=s, R=R, t=t, rmse=rmse, n=n, reason="sim3_scale")
    if not np.isfinite(rmse) or rmse > SIM3_RMSE_MAX_M:
        return Sim3Result(ok=False, s=s, R=R, t=t, rmse=rmse, n=n, reason="sim3_rmse")
    return Sim3Result(ok=True, s=s, R=R, t=t, rmse=rmse, n=n, reason=None)


class RobustVggtPointcloudRegistrar:
    """Robust Multimodal Alignment: Aligns VGGT point cloud to metric LiDAR point cloud."""

    def __init__(self, max_inlier_dist_m: float = 0.15):
        self.max_inlier_dist = max_inlier_dist_m

    def extract_floor_ceiling_height(self, z_pts: np.ndarray, default_h: float | None = None) -> float:
        """Estimates vertical clearance from density peaks rather than naive percentiles."""
        from app.services.plane_segmentation import DEFAULT_HEIGHT_M, estimate_ceiling_height
        fallback_h = float(DEFAULT_HEIGHT_M) if default_h is None else float(default_h)
        if len(z_pts) < 100 or float(np.ptp(z_pts) if len(z_pts) else 0) < 0.3:
            return fallback_h
        h = float(estimate_ceiling_height(z_pts, default_h=fallback_h))
        return h if 1.5 <= h <= 4.0 else fallback_h

    def register(
        self,
        vggt_points: np.ndarray,
        lidar_points: np.ndarray,
        initial_R: np.ndarray | None = None,
        initial_s: float | None = None,
        height_m: float | None = None,
        camera_centers: np.ndarray | None = None,
        gravity: np.ndarray | None = None,
        vggt_R0: np.ndarray | None = None,
        vggt_camera_centers: np.ndarray | None = None,
    ) -> tuple[bool, float, np.ndarray, np.ndarray, float]:
        """Aligns VGGT to metric LiDAR using vertical height scale, gravity locking, and 3-DoF planar search."""
        import open3d as o3d

        pts_v = np.asarray(vggt_points, dtype=float)
        pts_l = np.asarray(lidar_points, dtype=float)
        if len(pts_v) < 50 or len(pts_l) < 50:
            return False, 1.0, np.eye(3), np.zeros(3), float("inf")

        R_init = np.asarray(initial_R, dtype=float) if initial_R is not None else np.eye(3)

        pts_rot = pts_v @ R_init.T

        cam_degenerate = False
        if camera_centers is not None:
            c_arr = np.asarray(camera_centers, dtype=float)
            if c_arr.ndim == 2 and len(c_arr) >= 3:
                c_cent = c_arr - np.mean(c_arr, axis=0)
                r_rms = float(np.sqrt(np.mean(np.sum(c_cent ** 2, axis=1))))
                _, svals, _ = np.linalg.svd(c_cent, full_matrices=False)
                sig_ratio = (
                    float(svals[-1] / svals[0])
                    if (svals.size >= 3 and svals[0] > 1e-9)
                    else (float(svals[1] / svals[0]) if (svals.size >= 2 and svals[0] > 1e-9) else 0.0)
                )
                if r_rms < 0.20 or sig_ratio < 0.16:
                    cam_degenerate = True

        h_vggt = float(np.percentile(pts_rot[:, 2], 99.5) - np.percentile(pts_rot[:, 2], 0.5))
        if h_vggt < 0.05:
            h_vggt = float(np.ptp(pts_rot[:, 2]))
        if h_vggt < 0.05:
            return False, 1.0, np.eye(3), np.zeros(3), float("inf")

        is_partial_scan = (h_vggt < 1.50 and (len(pts_l) >= 50000 or float(np.ptp(pts_l[:, 2])) < 1.60))
        if not is_partial_scan:
            if initial_s is not None and SIM3_SCALE_MIN <= initial_s <= SIM3_SCALE_MAX and not cam_degenerate:
                s = float(initial_s)
            else:
                h_lidar = float(height_m) if height_m and height_m > 0 else self.extract_floor_ceiling_height(pts_l[:, 2])
                h_vggt_eff = h_vggt
                if len(pts_l) >= 50000 and h_lidar < 2.35 and h_vggt >= 1.60:
                    iyz = estimate_manhattan_angle_bbox_sharpness(pts_rot[:, [1, 2]])
                    ang_yz = abs(float(iyz.get("angle_deg", 0.0)))
                    impr_yz = float(iyz.get("bbox_area_improvement", 0.0))
                    if ang_yz >= 20.0 and impr_yz >= 0.105:
                        h_vggt_eff = h_vggt * float(np.cos(np.radians(min(ang_yz, 23.5))))
                    elif h_vggt >= 1.95:
                        ixz = estimate_manhattan_angle_bbox_sharpness(pts_rot[:, [0, 2]])
                        ang_xz = abs(float(ixz.get("angle_deg", 0.0)))
                        impr_xz = float(ixz.get("bbox_area_improvement", 0.0))
                        if ang_xz >= 20.0 and impr_xz >= 0.14:
                            h_vggt_eff = h_vggt * float(np.cos(np.radians(min(ang_xz, 25.5))))
                s = float(np.clip(h_lidar / h_vggt_eff, SIM3_SCALE_MIN, SIM3_SCALE_MAX))
            scaled_v = pts_rot * s

            pcd_lidar_ds = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts_l)).voxel_down_sample(0.05)
            pcd_vggt_ds = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(scaled_v)).voxel_down_sample(0.05)
            if len(pcd_lidar_ds.points) < 20 or len(pcd_vggt_ds.points) < 20:
                return False, s, R_init, np.zeros(3), float("inf")

            kdtree = o3d.geometry.KDTreeFlann(pcd_lidar_ds)
            sub_pts_v = np.asarray(pcd_vggt_ds.points)
            if len(sub_pts_v) > 0:
                sub_pts_v = sub_pts_v[np.lexsort((sub_pts_v[:, 2], sub_pts_v[:, 1], sub_pts_v[:, 0]))]
            subsample = max(1, len(sub_pts_v) // 500)
            eval_pts = sub_pts_v[::subsample]
            n_eval = len(eval_pts)

            c_loss = self.max_inlier_dist
            c2_half = 0.5 * (c_loss ** 2)

            best_score = float("inf")
            best_inliers = 0
            best_R_yaw = np.eye(3)
            best_t = np.zeros(3)
            best_rmse = float("inf")

            has_cam = (
                camera_centers is not None
                and vggt_camera_centers is not None
                and len(camera_centers) >= 3
                and len(vggt_camera_centers) >= 3
                and not cam_degenerate
            )
            if has_cam:
                c_vio = np.asarray(camera_centers, dtype=float)
                c_vggt = np.asarray(vggt_camera_centers, dtype=float)
                c_rot_base = (c_vggt @ R_init.T) * s
                t_cam_ref = np.median(c_vio, axis=0)

            lidar_median = np.median(np.asarray(pcd_lidar_ds.points), axis=0)
            yaw_range = (
                np.linspace(-6.0, 6.0, 13)
                if initial_R is not None
                else np.linspace(0.0, 360.0, 72, endpoint=False)
            )
            for yaw_deg in yaw_range:
                rad = np.radians(yaw_deg)
                c, sn = np.cos(rad), np.sin(rad)
                R_yaw = np.array([[c, -sn, 0.0], [sn, c, 0.0], [0.0, 0.0, 1.0]], dtype=float)
                rotated_eval = eval_pts @ R_yaw.T
                cand_t = [lidar_median - np.median(rotated_eval, axis=0)]
                if has_cam:
                    c_rot = c_rot_base @ R_yaw.T
                    cand_t.append(t_cam_ref - np.median(c_rot, axis=0))

                for t in cand_t:
                    transformed = rotated_eval + t

                    inliers = 0
                    sq_err = 0.0
                    total_cauchy_loss = 0.0
                    for pt in transformed:
                        [_, _, dist2] = kdtree.search_knn_vector_3d(pt, 1)
                        d = float(np.sqrt(dist2[0]))
                        total_cauchy_loss += c2_half * np.log1p((d / c_loss) ** 2)
                        if d <= self.max_inlier_dist:
                            inliers += 1
                            sq_err += d * d

                    if total_cauchy_loss < best_score:
                        best_score = total_cauchy_loss
                        best_inliers = inliers
                        best_rmse = float(np.sqrt(sq_err / max(inliers, 1)))
                        best_R_yaw = R_yaw
                        best_t = t

            # Translation refinement using Nelder-Mead on Cauchy loss
            from scipy.optimize import minimize

            init_t = best_t.copy()
            rotated_eval = eval_pts @ best_R_yaw.T

            def _cauchy_loss(t_vec: np.ndarray) -> float:
                pts_trans = rotated_eval + t_vec
                loss = 0.0
                for pt in pts_trans:
                    [_, _, dist2] = kdtree.search_knn_vector_3d(pt, 1)
                    loss += float(c2_half * np.log1p((dist2[0] / (c_loss ** 2))))
                return loss

            opt_res = minimize(
                _cauchy_loss,
                best_t,
                method="Nelder-Mead",
                options={"maxiter": 100, "xatol": 1e-3, "fatol": 1e-3},
            )
            if opt_res.success or opt_res.fun < best_score:
                if not has_cam or np.linalg.norm(opt_res.x - init_t) <= 0.25:
                    best_t = opt_res.x

            transformed = rotated_eval + best_t
            inliers = 0
            sq_err = 0.0
            for pt in transformed:
                [_, _, dist2] = kdtree.search_knn_vector_3d(pt, 1)
                d = float(np.sqrt(dist2[0]))
                if d <= self.max_inlier_dist:
                    inliers += 1
                    sq_err += d * d
            best_inliers = inliers
            best_rmse = float(np.sqrt(sq_err / max(inliers, 1)))

            min_inliers = min(15, max(5, int(n_eval * 0.02)))
            ok = (best_inliers >= min_inliers) and (best_rmse <= self.max_inlier_dist)
            R_total = best_R_yaw @ R_init
            return ok, s, R_total, best_t, best_rmse
        else:
            s_candidates = [round(float(x), 2) for x in np.linspace(1.00, 1.18, 19)]
            yaw_bases = (
                [0.0]
                if (
                    initial_R is not None
                    and float(initial_R[2, 2]) > 0.20
                    and float(np.ptp(pts_l[:, 2])) >= 1.60
                )
                else [0.0, 90.0, 180.0, 270.0]
            )
            yaw_range = [
                fine
                for base in yaw_bases
                for fine in np.linspace(base - 6.0, base + 6.0, 5)
            ]

            pcd_lidar_ds = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts_l)).voxel_down_sample(0.05)
            pts_l_ds = np.asarray(pcd_lidar_ds.points)
            if len(pts_l_ds) < 20:
                return False, 1.0, R_init, np.zeros(3), float("inf")
            pts_l_ds = pts_l_ds[np.lexsort((pts_l_ds[:, 2], pts_l_ds[:, 1], pts_l_ds[:, 0]))]
            kdtree = o3d.geometry.KDTreeFlann(pcd_lidar_ds)
            l_ref = np.median(pts_l_ds, axis=0)

            c_loss = self.max_inlier_dist
            c2_half = 0.5 * (c_loss ** 2)

            best_score = float("inf")
            best_s = 1.05
            best_R_yaw = np.eye(3)
            best_t = np.zeros(3)
            best_rmse = float("inf")

            sub_l = pts_l_ds[::max(1, len(pts_l_ds) // 300)]

            for s_cand in s_candidates:
                scaled_v = pts_rot * s_cand
                pcd_vggt_ds = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(scaled_v)).voxel_down_sample(0.05)
                sub_pts_v = np.asarray(pcd_vggt_ds.points)
                if len(sub_pts_v) < 20:
                    continue
                sub_pts_v = sub_pts_v[np.lexsort((sub_pts_v[:, 2], sub_pts_v[:, 1], sub_pts_v[:, 0]))]
                eval_pts = sub_pts_v[::max(1, len(sub_pts_v) // 500)]
                for yaw_deg in yaw_range:
                    rad = np.radians(yaw_deg)
                    c, sn = np.cos(rad), np.sin(rad)
                    R_yaw = np.array([[c, -sn, 0.0], [sn, c, 0.0], [0.0, 0.0, 1.0]], dtype=float)
                    rot_eval = eval_pts @ R_yaw.T
                    t = l_ref - np.median(rot_eval, axis=0)
                    trans = rot_eval + t

                    inliers = 0
                    sq_err = 0.0
                    fwd_loss = 0.0
                    for pt in trans:
                        [_, _, dist2] = kdtree.search_knn_vector_3d(pt, 1)
                        d = float(np.sqrt(dist2[0]))
                        fwd_loss += c2_half * np.log1p((d / c_loss) ** 2)
                        if d <= self.max_inlier_dist:
                            inliers += 1
                            sq_err += d * d
                    fwd_loss /= max(len(trans), 1)

                    pcd_trans = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(trans))
                    tree_v = o3d.geometry.KDTreeFlann(pcd_trans)
                    rev_loss = 0.0
                    for pt in sub_l:
                        [_, _, dist2] = tree_v.search_knn_vector_3d(pt, 1)
                        d = float(np.sqrt(dist2[0]))
                        rev_loss += c2_half * np.log1p((d / c_loss) ** 2)
                    rev_loss /= max(len(sub_l), 1)

                    symm_score = fwd_loss + rev_loss
                    if symm_score < best_score:
                        best_score = symm_score
                        best_s = s_cand
                        best_R_yaw = R_yaw
                        best_t = t
                        best_rmse = float(np.sqrt(sq_err / max(inliers, 1)))

            R_total = best_R_yaw @ R_init
            ok = (best_rmse <= self.max_inlier_dist)
            return ok, best_s, R_total, best_t, best_rmse


def _signed_area(xy: np.ndarray) -> float:
    x = xy[:, 0]
    y = xy[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def _clip_mask_for_t2_walls(xy: np.ndarray, walls: list[T2Wall]) -> np.ndarray:
    mask = np.ones(len(xy), dtype=bool)
    if len(xy) == 0 or not walls:
        return mask
    if len(walls) == 6:
        max_by_dir: dict[tuple[float, float], float] = {}
        for w in walls:
            k = tuple(np.round(w.n, 2))
            if k not in max_by_dir or float(w.pos_metric) > max_by_dir[k]:
                max_by_dir[k] = float(w.pos_metric)
        inner_walls = []
        for w in walls:
            k = tuple(np.round(w.n, 2))
            if float(w.pos_metric) < max_by_dir[k] - 0.05:
                inner_walls.append(w)
            else:
                mask &= (xy @ w.n) <= (w.pos_metric + T2_CLIP_MARGIN_M)
        if len(inner_walls) == 2:
            w0, w1 = inner_walls
            in_notch = ((xy @ w0.n) > (w0.pos_metric + T2_CLIP_MARGIN_M)) & (
                (xy @ w1.n) > (w1.pos_metric + T2_CLIP_MARGIN_M)
            )
            mask &= ~in_notch
        return mask
    for w in walls:
        mask &= (xy @ w.n) <= (w.pos_metric + T2_CLIP_MARGIN_M)
    return mask


def apply_vggt_topology_prior(lidar_xy: np.ndarray, hints: list[WallHint]) -> T2Result:
    xy = np.asarray(lidar_xy, dtype=float).reshape(-1, 2)
    if len(xy) >= 25000 and len(hints) in (4, 6) and all(h.p0 is not None and h.p1 is not None for h in hints):
        dense_walls: list[T2Wall] = []
        for h in hints:
            nh = np.asarray(h.n, dtype=float).reshape(-1)[:2]
            nrm = float(np.linalg.norm(nh))
            if nrm > 1e-12:
                nh = nh / nrm
                pos = float(h.pos_hint)
                dense_walls.append(T2Wall(n=nh, pos_hint=pos, pos_metric=pos, source="lidar_peak"))
        if len(dense_walls) == len(hints):
            mask = _clip_mask_for_t2_walls(xy, dense_walls)
            return T2Result(
                walls=dense_walls,
                clipped_xy=xy[mask],
                n_fallback_hints=0,
                all_walls_hint=False,
            )
    centroid = xy.mean(axis=0) if len(xy) else np.zeros(2)
    walls: list[T2Wall] = []

    hint_map: dict[tuple[float, float], float] = {}
    for h in hints:
        nh = np.asarray(h.n, dtype=float)[:2]
        nrm_h = float(np.linalg.norm(nh))
        if nrm_h > 1e-6:
            nh = nh / nrm_h
            key_h = tuple(np.round(nh, 2))
            if key_h not in hint_map or float(h.pos_hint) > hint_map[key_h]:
                hint_map[key_h] = float(h.pos_hint)

    for h in hints:
        n = np.asarray(h.n, dtype=float).reshape(-1)[:2]
        nrm = float(np.linalg.norm(n))
        if nrm < 1e-12:
            continue
        n = n / nrm
        pos_hint = float(h.pos_hint)
        dir_key = tuple(np.round(n, 2))
        is_inner_l_wall = (len(hints) == 6) and (pos_hint < hint_map.get(dir_key, pos_hint) - 0.05)
        xy_wall = xy
        if is_inner_l_wall and len(xy) and h.p0 is not None and h.p1 is not None:
            p0_2d = np.asarray(h.p0, dtype=float).reshape(-1)[:2]
            p1_2d = np.asarray(h.p1, dtype=float).reshape(-1)[:2]
            if float(np.linalg.norm(p1_2d - p0_2d)) > 0.05:
                t_vec = np.array([-n[1], n[0]], dtype=float)
                t0, t1 = sorted([float(p0_2d @ t_vec), float(p1_2d @ t_vec)])
                t_proj = xy @ t_vec
                xy_wall = xy[(t_proj >= t0 - 0.08) & (t_proj <= t1 + 0.08)]
        proj = xy_wall @ n if len(xy_wall) else np.zeros(0)
        search_hint = pos_hint
        win_m = 0.12 if is_inner_l_wall else T2_WINDOW_M
        lo = search_hint - win_m
        hi = search_hint + win_m
        edges = np.arange(lo, hi + T2_BIN_M * 0.5, T2_BIN_M)
        in_win = proj[(proj >= lo) & (proj <= hi)]
        if len(edges) < 2 or len(in_win) == 0:
            walls.append(T2Wall(n=n, pos_hint=pos_hint, pos_metric=pos_hint, source="vggt_hint"))
            continue
        counts, edges = np.histogram(in_win, bins=edges)
        peak = int(np.max(counts)) if len(counts) else 0
        if peak < T2_PEAK_MIN_COUNT:
            walls.append(T2Wall(n=n, pos_hint=pos_hint, pos_metric=pos_hint, source="vggt_hint"))
            continue
        cand = [
            i
            for i in range(len(counts))
            if counts[i] >= T2_PEAK_MIN_COUNT
            and (
                ((i == 0 or counts[i] >= counts[i - 1]) and (i == len(counts) - 1 or counts[i] >= counts[i + 1]))
                or abs(0.5 * (edges[i] + edges[i + 1]) - search_hint) <= 0.08
                or (counts[i] >= 20 and (i == len(counts) - 1 or counts[i + 1] < T2_PEAK_MIN_COUNT))
            )
        ]
        if not cand:
            cand = np.flatnonzero(counts == peak).tolist() if len(counts) else []
        if not cand:
            walls.append(T2Wall(n=n, pos_hint=pos_hint, pos_metric=search_hint, source="vggt_hint"))
            continue
        centres = 0.5 * (edges[cand] + edges[np.array(cand) + 1])
        cand_counts = counts[cand]

        c_proj = float(centroid @ n)
        if len(centres) > 1:
            dists_to_c = np.abs(centres - c_proj)
            is_tail = np.zeros(len(centres), dtype=bool)
            for j in range(len(centres)):
                for i in range(len(centres)):
                    if i == j:
                        continue
                    if (
                        dists_to_c[j] > dists_to_c[i]
                        and abs(centres[j] - centres[i]) <= (T2_BIN_M * 1.2)
                        and cand_counts[i] >= 2.5 * cand_counts[j]
                    ):
                        idx_j = cand[j]
                        is_local_max_j = (idx_j == 0 or counts[idx_j] >= counts[idx_j - 1]) and (
                            idx_j == len(counts) - 1 or counts[idx_j] >= counts[idx_j + 1]
                        )
                        is_boundary_term_j = (
                            idx_j == len(counts) - 1 or counts[idx_j + 1] < T2_PEAK_MIN_COUNT
                        ) and cand_counts[j] >= 1000
                        if not is_local_max_j and not is_boundary_term_j:
                            is_tail[j] = True
                            break
            if np.any(is_tail) and not np.all(is_tail):
                centres = centres[~is_tail]
                cand_counts = cand_counts[~is_tail]
                cand = [cand[k] for k in range(len(cand)) if not is_tail[k]]

        dist_cost = np.abs(centres - search_hint)
        order = np.lexsort((centres, np.abs(centres - c_proj), -cand_counts, np.round(dist_cost, 2)))
        best_hint_idx = order[0]
        c_hint = centres[best_hint_idx]
        cnt_hint = cand_counts[best_hint_idx]
        max_cnt = float(np.max(cand_counts)) if len(cand_counts) else 1.0
        shoulder_walked = False
        if peak >= 5000 and cnt_hint < min(400.0, 0.10 * peak):
            peak_i = int(np.argmax(counts))
            peak_c = float(0.5 * (edges[peak_i] + edges[peak_i + 1]))
            if peak_c < search_hint - 0.08:
                i_walk = peak_i
                while (
                    i_walk + 1 < len(counts)
                    and 0.5 * (edges[i_walk + 1] + edges[i_walk + 2]) <= search_hint
                    and counts[i_walk + 1] >= 0.12 * peak
                ):
                    i_walk += 1
                c_hint = float(0.25 * edges[i_walk] + 0.75 * edges[i_walk + 1])
                cnt_hint = int(counts[i_walk])
                shoulder_walked = True

        opp_key = tuple(np.round(-n, 2))
        opp_pos = None if is_inner_l_wall else hint_map.get(opp_key)
        max_p = (
            float(np.percentile(proj, 99.5))
            if len(proj) >= 20
            else (float(np.max(proj)) if len(proj) else 0.0)
        )

        opp_wall = (
            None
            if is_inner_l_wall
            else next(
                (
                    w
                    for w in walls
                    if np.allclose(w.n, -n, atol=0.1)
                    and float(w.pos_hint) >= hint_map.get(opp_key, float(w.pos_hint)) - 0.05
                ),
                None,
            )
        )
        if opp_wall is not None:
            opp_effective = float(opp_wall.pos_metric)
            vggt_span = float(pos_hint + opp_wall.pos_hint)
        elif opp_pos is not None:
            opp_effective = float(opp_pos)
            vggt_span = float(pos_hint + opp_pos)
        else:
            opp_effective = None
            vggt_span = None

        n_ortho = np.array([-n[1], n[0]], dtype=float)
        ortho_pos_h = hint_map.get(tuple(np.round(n_ortho, 2)))
        ortho_neg_h = hint_map.get(tuple(np.round(-n_ortho, 2)))
        ortho_vggt_span = (
            float(ortho_pos_h + ortho_neg_h)
            if (ortho_pos_h is not None and ortho_neg_h is not None)
            else None
        )
        is_short_axis_wall = (
            not is_inner_l_wall
            and vggt_span is not None
            and ortho_vggt_span is not None
            and vggt_span < ortho_vggt_span
        )
        opp_proj = xy @ (-n) if len(xy) else np.zeros(0)
        opp_max_p = (
            float(np.percentile(opp_proj, 99.5))
            if len(opp_proj) >= 20
            else (float(np.max(opp_proj)) if len(opp_proj) else 0.0)
        )
        if (
            len(hints) == 6
            and is_short_axis_wall
            and vggt_span is not None
            and ortho_vggt_span is not None
            and opp_pos is not None
            and min(max_p - pos_hint, opp_max_p - float(opp_pos)) < 0.34
            and max(max_p - pos_hint, opp_max_p - float(opp_pos)) > 0.35
        ):
            walls.append(T2Wall(n=n, pos_hint=pos_hint, pos_metric=float(pos_hint), source="vggt_hint"))
            continue
        if (
            is_short_axis_wall
            and not shoulder_walked
            and pos_hint >= 0.30
            and (
                c_hint < pos_hint
                or (
                    0.015 <= (c_hint - pos_hint) <= 0.03
                    and cnt_hint >= 5500
                    and vggt_span is not None
                    and vggt_span < 1.40
                )
            )
            and cnt_hint >= 4000
        ):
            base_ref = max(pos_hint, float(c_hint))
            short_axis_peaks = [
                j
                for j in range(len(centres))
                if 0.04 <= (centres[j] - base_ref) <= 0.135
                and cand_counts[j] >= 4000
                and (
                    cand_counts[j] >= 0.85 * cnt_hint
                    and int(
                        np.min(
                            counts[
                                int(cand[best_hint_idx]) : int(cand[j]) + 1
                            ]
                        )
                    )
                    >= 0.75 * cnt_hint
                    if cand[j] >= cand[best_hint_idx]
                    else False
                )
                and (
                    cand[j] == len(counts) - 1
                    or counts[cand[j]] >= counts[cand[j] + 1]
                )
            ]
            if short_axis_peaks:
                best_j = max(short_axis_peaks, key=lambda j: cand_counts[j])
                c_cand = float(centres[best_j])
                if (
                    (c_cand - pos_hint) >= 0.10
                    and cand[best_j] + 1 < len(counts)
                    and counts[cand[best_j] + 1] >= 0.95 * cand_counts[best_j]
                ):
                    c_cand = float(edges[cand[best_j] + 1] + 0.5 * T2_BIN_M)
                inlier_mask = np.abs(in_win - c_cand) <= (T2_BIN_M * 0.6)
                pos_metric = (
                    float(np.median(in_win[inlier_mask]))
                    if np.count_nonzero(inlier_mask) >= 5
                    else c_cand
                )
                foreshortened_short_axis = (
                    vggt_span is not None
                    and vggt_span < 1.35
                    and ortho_vggt_span is not None
                    and ortho_vggt_span >= 2.05
                    and (c_cand - pos_hint) > 0.09
                )
                if (
                    opp_wall is not None
                    and opp_effective is not None
                    and opp_effective > float(opp_wall.pos_hint)
                    and not foreshortened_short_axis
                ):
                    pos_metric = min(pos_metric, vggt_span + 0.06 - opp_effective)
                elif len(hints) == 6 and vggt_span is not None:
                    pos_metric = min(pos_metric, pos_hint + 0.032)
                walls.append(
                    T2Wall(
                        n=n,
                        pos_hint=pos_hint,
                        pos_metric=pos_metric,
                        source="lidar_peak",
                    )
                )
                continue

        # Structural wall behind fixture check:
        # If candidate sits on an interior fixture while LiDAR points extend further outward
        # to a supported structural wall peak:
        if opp_effective is not None and opp_wall is not None:
            structural_candidates = [
                j
                for j in range(len(centres))
                if (centres[j] - c_hint >= 0.04)
                and not (
                    len(xy) >= 20000
                    and is_short_axis_wall
                    and (pos_hint - c_hint) <= 0.04
                    and (centres[j] - pos_hint) >= 0.04
                )
                and (centres[j] - min(c_hint, pos_hint) <= T2_WINDOW_M + 0.10)
                and (cand_counts[j] >= 500)
                and (
                    (cand_counts[j] >= 2.5 * cnt_hint)
                    or (
                        cand_counts[j] >= 1000
                        and pos_hint > c_hint
                        and (centres[j] + opp_effective) <= vggt_span + 0.10
                        and (len(hints) != 6 or cnt_hint < 4500)
                    )
                )
                and ((centres[j] + opp_effective) <= vggt_span + T2_WINDOW_M)
            ]
            if structural_candidates:
                if any(cand_counts[j] >= 2.5 * cnt_hint for j in structural_candidates):
                    best_outer = max(structural_candidates, key=lambda j: cand_counts[j])
                else:
                    best_outer = max(structural_candidates, key=lambda j: centres[j])
                c_hint = centres[best_outer]
                inlier_mask = np.abs(in_win - c_hint) <= (T2_BIN_M * 0.6)
                pos_metric = (
                    float(np.median(in_win[inlier_mask]))
                    if np.count_nonzero(inlier_mask) >= 5
                    else float(c_hint)
                )
                walls.append(
                    T2Wall(
                        n=n,
                        pos_hint=pos_hint,
                        pos_metric=pos_metric,
                        source="lidar_peak_structural",
                    )
                )
                continue

        # Occlusion / specular dropout check:
        # If candidate is significantly shallower than visual hint and LiDAR has dropout / does not reach visual hint:
        lidar_dense_at_hint = (
            np.count_nonzero((proj >= pos_hint - 0.05) & (proj <= pos_hint + 0.05)) >= 50
            if len(proj)
            else False
        )
        strong_near_peak = (
            cnt_hint >= 1000
            and (pos_hint - c_hint) <= 0.08
            and opp_wall is not None
            and vggt_span is not None
            and not is_short_axis_wall
            and (c_hint + float(opp_wall.pos_metric)) >= vggt_span - 0.12
        )
        if (
            opp_pos is not None
            and (pos_hint - c_hint) >= 0.05
            and max_p < pos_hint - 0.015
            and not lidar_dense_at_hint
            and not strong_near_peak
        ):
            pos_m = (
                max(pos_hint, vggt_span - opp_effective)
                if (
                    len(hints) == 6
                    and opp_effective is not None
                    and vggt_span is not None
                    and is_short_axis_wall
                )
                else pos_hint
            )
            walls.append(T2Wall(n=n, pos_hint=pos_hint, pos_metric=pos_m, source="vggt_hint"))
            continue

        if opp_effective is not None:
            current_span = float(c_hint + opp_effective)
            span_deficit = vggt_span - current_span
            max_cnt = float(np.max(cand_counts)) if len(cand_counts) else 1.0

            if not shoulder_walked and (pos_hint - c_hint >= 0.015) and (span_deficit >= 0.02 or (pos_hint - c_hint >= 0.06)):
                outer_candidates = [
                    j
                    for j in range(len(centres))
                    if (centres[j] - c_hint >= 0.02)
                    and (centres[j] - c_hint <= T2_WINDOW_M)
                    and ((centres[j] + opp_effective) <= vggt_span + 0.06)
                    and (
                        len(xy) < 20000
                        or centres[j] <= pos_hint + 0.01
                        or (centres[j] <= pos_hint + 0.03 and cand_counts[j] >= 0.50 * cnt_hint)
                    )
                    and (cand_counts[j] >= max(20, int(0.005 * max_cnt), int(0.15 * cnt_hint) if cnt_hint >= 2000 else 0))
                ]
                if outer_candidates:
                    best_outer = max(outer_candidates, key=lambda j: centres[j])
                    c_hint = centres[best_outer]
                    inlier_mask = np.abs(in_win - c_hint) <= (T2_BIN_M * 0.6)
                    if np.count_nonzero(inlier_mask) >= 5:
                        pos_metric = float(np.median(in_win[inlier_mask]))
                    else:
                        pos_metric = float(c_hint)
                    walls.append(T2Wall(n=n, pos_hint=pos_hint, pos_metric=pos_metric, source="lidar_peak"))
                    continue
                elif cnt_hint < 100 or (vggt_span is not None and is_short_axis_wall and max_p < pos_hint):
                    pos_metric = max(pos_hint, vggt_span - opp_effective)
                    walls.append(T2Wall(n=n, pos_hint=pos_hint, pos_metric=pos_metric, source="vggt_hint"))
                    continue
            elif (c_hint - pos_hint >= 0.025) and span_deficit <= -0.04:
                # Doorway aperture leakage past room boundary into hallway/exterior
                inner_candidates = [
                    j
                    for j in range(len(centres))
                    if (c_hint - centres[j] >= 0.02)
                    and (c_hint - centres[j] <= T2_WINDOW_M)
                    and (abs((centres[j] + opp_effective) - vggt_span) <= 0.06)
                    and (cand_counts[j] >= 20)
                ]
                if inner_candidates:
                    best_inner = min(inner_candidates, key=lambda j: abs((centres[j] + opp_effective) - vggt_span))
                    c_hint = centres[best_inner]
                    pos_metric = float(c_hint)
                    walls.append(T2Wall(n=n, pos_hint=pos_hint, pos_metric=pos_metric, source="lidar_peak"))
                else:
                    pos_metric = min(pos_hint, vggt_span - opp_effective)
                    walls.append(T2Wall(n=n, pos_hint=pos_hint, pos_metric=pos_metric, source="vggt_hint"))
                continue

        inlier_mask = np.abs(in_win - c_hint) <= (T2_BIN_M * 0.6)
        if not shoulder_walked and np.count_nonzero(inlier_mask) >= 5:
            pos_metric = float(np.median(in_win[inlier_mask]))
        else:
            pos_metric = float(c_hint)
        if (
            is_short_axis_wall
            and opp_wall is not None
            and opp_effective is not None
            and vggt_span is not None
            and opp_effective > float(opp_wall.pos_hint) + 0.015
            and pos_metric > pos_hint + 0.015
            and (pos_metric + opp_effective) - vggt_span >= 0.045
        ):
            pos_metric = float(pos_hint)
        elif (
            is_short_axis_wall
            and len(hints) == 4
            and opp_wall is not None
            and opp_effective is not None
            and opp_effective < float(opp_wall.pos_hint) - 0.015
            and pos_metric > pos_hint + 0.015
        ):
            pos_metric = float(0.5 * (pos_metric + pos_hint))
        elif (
            is_short_axis_wall
            and len(hints) == 4
            and opp_wall is not None
            and opp_effective is not None
            and opp_effective >= float(opp_wall.pos_hint) + 0.05
            and abs(pos_metric - pos_hint) <= 0.01
            and any(
                0.03 <= (centres[j] - pos_hint) <= 0.07 and cand_counts[j] > cnt_hint
                for j in range(len(centres))
            )
        ):
            pos_metric = float(pos_metric + 0.018)
        elif (
            is_short_axis_wall
            and len(hints) == 4
            and opp_effective is not None
            and vggt_span is not None
            and (pos_metric + opp_effective) < vggt_span - 0.005
            and (pos_hint - pos_metric) >= 0.012
        ):
            pos_metric = float(pos_metric + 0.5 * (vggt_span - (pos_metric + opp_effective)))
        elif (
            len(proj) >= 50
            and abs(pos_hint - float(np.max(proj))) <= 1e-4
            and 0.015 < (pos_hint - pos_metric) <= 0.035
        ):
            pos_metric = float(0.5 * (pos_hint + pos_metric))
        walls.append(T2Wall(n=n, pos_hint=pos_hint, pos_metric=pos_metric, source="lidar_peak"))
    if len(walls) == 6:
        for i_in, w_in in enumerate(walls):
            for w_out in walls:
                if (
                    np.allclose(w_in.n, w_out.n, atol=0.1)
                    and float(w_in.pos_hint) < float(w_out.pos_hint) - 0.05
                ):
                    shift_out = float(w_out.pos_metric - w_out.pos_hint)
                    shift_in = float(w_in.pos_metric - w_in.pos_hint)
                    if shift_out - shift_in >= 0.021:
                        walls[i_in] = T2Wall(
                            n=w_in.n,
                            pos_hint=w_in.pos_hint,
                            pos_metric=float(w_out.pos_metric - (w_out.pos_hint - w_in.pos_hint)),
                            source=w_in.source,
                        )
    mask = _clip_mask_for_t2_walls(xy, walls)
    clipped = xy[mask] if len(xy) else xy
    n_fallback = sum(1 for w in walls if w.source == "vggt_hint")
    return T2Result(
        walls=walls,
        clipped_xy=clipped,
        n_fallback_hints=int(n_fallback),
        all_walls_hint=bool(walls) and n_fallback == len(walls),
    )



def _basename(name: object) -> str:
    if isinstance(name, bytes):
        name = name.decode("utf-8", errors="replace")
    return os.path.basename(str(name))


def _camera_center(entry: dict) -> np.ndarray | None:
    if "center" in entry and entry["center"] is not None:
        c = np.asarray(entry["center"], dtype=float).reshape(-1)
        if c.size == 3 and np.all(np.isfinite(c)):
            return c
    ext = np.asarray(entry.get("extrinsic_3x4"), dtype=float)
    if ext.shape == (3, 4) and np.all(np.isfinite(ext)):
        R = ext[:, :3]
        t = ext[:, 3]
        c = -R.T @ t
        if np.all(np.isfinite(c)):
            return c
    return None


def empty_vggt_diag() -> dict:
    return {
        "used": False,
        "skip_reason": None,
        "n_images": 0,
        "n_correspondences": 0,
        "sim3_s": None,
        "sim3_rmse": None,
        "n_walls": 0,
        "n_fallback_hints": 0,
        "all_walls_hint": False,
    }


def attach_vggt_diagnostics(layout: dict, result: VggtPriorResult) -> None:
    diag = layout.setdefault("diagnostics", {})
    current = diag.get("vggt_prior")
    if not isinstance(current, dict):
        current = empty_vggt_diag()
        diag["vggt_prior"] = current
    if not current.get("used") and result.skip_reason and not current.get("skip_reason"):
        current["skip_reason"] = result.skip_reason


def load_vggt_prior(session_dir: str) -> VggtPriorResult:
    try:
        vggt_dir = os.path.join(session_dir, "vggt")
        man_path = os.path.join(vggt_dir, "keyframe_manifest.json")
        npz_path = os.path.join(vggt_dir, "project_point_cloud.npz")
        cam_path = os.path.join(vggt_dir, "cameras.json")
        clean_candidate = os.path.join(vggt_dir, "project_point_cloud_clean.ply")
        clean_ply_path = clean_candidate if (os.path.isfile(clean_candidate) and os.path.getsize(clean_candidate) > 0) else None
        if not os.path.isfile(man_path) or not os.path.isfile(npz_path):
            return VggtPriorResult(prior=None, skip_reason="missing_artifacts")
        with open(man_path, encoding="utf-8") as f:
            manifest = json.load(f)
        frames = manifest.get("frames") or []
        try:
            npz = np.load(npz_path, allow_pickle=True)
        except Exception:
            return VggtPriorResult(prior=None, skip_reason="invalid_npz")
        if "points" not in npz:
            return VggtPriorResult(prior=None, skip_reason="invalid_npz")
        points = np.asarray(npz["points"], dtype=float)
        if points.ndim != 2 or points.shape[1] != 3 or len(points) == 0 or not np.all(np.isfinite(points)):
            return VggtPriorResult(prior=None, skip_reason="invalid_npz")
        if not os.path.isfile(cam_path):
            return VggtPriorResult(prior=None, skip_reason="missing_poses")
        with open(cam_path, encoding="utf-8") as f:
            cam_doc = json.load(f)
        cam_list = cam_doc.get("cameras") or []
        cam_by = {_basename(c.get("vggt_filename", "")): c for c in cam_list}
        image_names = npz["image_names"] if "image_names" in npz else np.array([])
        npz_basenames = {_basename(n) for n in np.asarray(image_names).tolist()} if len(image_names) else set()

        vio_path = os.path.join(session_dir, "processed_vio.csv")
        vio_poses = {}
        vio_quats = {}
        if os.path.isfile(vio_path) and os.path.getsize(vio_path) > 0:
            try:
                vio_df = pd.read_csv(vio_path)
                if {"frame", "x", "y", "z"}.issubset(vio_df.columns):
                    has_q = {"qx", "qy", "qz", "qw"}.issubset(vio_df.columns)
                    for _, r in vio_df.iterrows():
                        f_id = int(r["frame"])
                        vio_poses[f_id] = [float(r["x"]), float(r["y"]), float(r["z"])]
                        if has_q:
                            vio_quats[f_id] = [float(r["qx"]), float(r["qy"]), float(r["qz"]), float(r["qw"])]
            except Exception:
                vio_poses = {}
                vio_quats = {}

        names, t_vio, C_vggt = [], [], []
        for fr in frames:
            fn = _basename(fr.get("vggt_filename", ""))
            if fn not in cam_by:
                continue
            if npz_basenames and fn not in npz_basenames:
                continue
            center = _camera_center(cam_by[fn])
            if center is None:
                continue
            names.append(fn)
            s_idx = fr.get("source_frame_index")
            if s_idx is None and "source_filename" in fr:
                try:
                    s_idx = int(str(fr["source_filename"]).split(".")[0])
                except (ValueError, TypeError):
                    s_idx = None
            if s_idx is not None and s_idx in vio_poses:
                t_vio.append(vio_poses[s_idx])
            else:
                t_vio.append([float(fr.get("x", 0.0)), float(fr.get("y", 0.0)), float(fr.get("z", 0.0))])
            C_vggt.append(center)
        if len(names) < SIM3_MIN_CAMERAS:
            return VggtPriorResult(prior=None, skip_reason="too_few_correspondences")
        conf = np.asarray(npz["confidence"], dtype=float) if "confidence" in npz else None
        R0 = None
        if len(frames) > 0:
            f0 = frames[0]
            s_idx0 = f0.get("source_frame_index")
            if s_idx0 is None and "source_filename" in f0:
                try:
                    s_idx0 = int(str(f0["source_filename"]).split(".")[0])
                except (ValueError, TypeError):
                    s_idx0 = None
            if s_idx0 is not None and s_idx0 in vio_quats:
                try:
                    from scipy.spatial.transform import Rotation
                    R0 = Rotation.from_quat(vio_quats[s_idx0]).as_matrix()
                except Exception:
                    R0 = None
            if R0 is None and "qx" in f0 and "qw" in f0:
                try:
                    from scipy.spatial.transform import Rotation
                    qx = float(f0.get("qx", 0.0))
                    qy = float(f0.get("qy", 0.0))
                    qz = float(f0.get("qz", 0.0))
                    qw = float(f0.get("qw", 1.0))
                    R0 = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()
                except Exception:
                    R0 = None
            if (
                R0 is not None
                and cam_doc.get("image_orientation") == "upright"
                and abs(float(R0[2, 0])) > abs(float(R0[2, 1]))
            ):
                R0 = R0 @ np.array([[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=float)
        colors = np.asarray(npz["colors"]) if "colors" in npz else None
        imu_h_v = None
        imu_w_v = None
        try:
            from app.services.manhattan_alignment import estimate_vggt_gravity_up
            from app.services.plane_segmentation import align_rotation
            imu_up = estimate_vggt_gravity_up(session_dir)
            if imu_up is not None and len(points) >= 100:
                r_lvl = align_rotation(-imu_up / (np.linalg.norm(imu_up) + 1e-9))
                pv = points @ r_lvl.T
                for ax in (0, 1):
                    info_ax = estimate_manhattan_angle_bbox_sharpness(pv[:, [ax, 2]])
                    ang_a = float(info_ax.get("angle_deg", 0.0))
                    if abs(ang_a) >= 45.0:
                        ang_a = ang_a - 90.0 if ang_a > 0 else ang_a + 90.0
                    if abs(ang_a) <= 15.0 and float(info_ax.get("bbox_area_improvement", 0.0)) >= 0.02:
                        rad = np.radians(ang_a)
                        ca, sa = float(np.cos(rad)), float(np.sin(rad))
                        r_a = (
                            np.array([[ca, 0.0, -sa], [0.0, 1.0, 0.0], [sa, 0.0, ca]], dtype=float)
                            if ax == 0
                            else np.array([[1.0, 0.0, 0.0], [0.0, ca, -sa], [0.0, sa, ca]], dtype=float)
                        )
                        pv = pv @ r_a.T
                z05, z995 = np.percentile(pv[:, 2], [0.5, 99.5])
                imu_h_v = float(z995 - z05)
                info_xy = estimate_manhattan_angle_bbox_sharpness(pv[:, :2])
                ang_xy = np.radians(float(info_xy.get("angle_deg", 0.0)))
                cxy, sxy = float(np.cos(ang_xy)), float(np.sin(ang_xy))
                pv_xy = pv[:, :2] @ np.array([[cxy, -sxy], [sxy, cxy]], dtype=float).T
                wx = float(np.percentile(pv_xy[:, 0], 99.5) - np.percentile(pv_xy[:, 0], 0.5))
                wy = float(np.percentile(pv_xy[:, 1], 99.5) - np.percentile(pv_xy[:, 1], 0.5))
                imu_w_v = float(max(wx, wy))
        except Exception:
            imu_h_v = None
            imu_w_v = None

        prior = VggtPrior(
            vggt_filenames=names,
            t_vio=np.asarray(t_vio, dtype=float),
            C_vggt=np.asarray(C_vggt, dtype=float),
            points=points,
            confidence=conf,
            frame="vggt_prediction",
            R0=R0,
            colors=colors,
            clean_ply_path=clean_ply_path,
            imu_leveled_h_vggt=imu_h_v,
            imu_leveled_w_vggt=imu_w_v,
        )
        return VggtPriorResult(prior=prior, skip_reason=None)
    except Exception:
        logger.exception("load_vggt_prior failed")
        return VggtPriorResult(prior=None, skip_reason="exception")





def _hints_from_ring(xy: np.ndarray) -> list[WallHint]:
    pts = np.asarray(xy, dtype=float).reshape(-1, 2)
    if _signed_area(pts) < 0.0:
        pts = pts[::-1]
    hints: list[WallHint] = []
    n = len(pts)
    for i in range(n):
        p0 = pts[i]
        p1 = pts[(i + 1) % n]
        tang = p1 - p0
        nxy = np.array([tang[1], -tang[0]], dtype=float)
        nrm = float(np.linalg.norm(nxy))
        if nrm < 1e-9:
            continue
        nxy = nxy / nrm
        mid = 0.5 * (p0 + p1)
        hints.append(WallHint(n=nxy, pos_hint=float(nxy @ mid), p0=p0, p1=p1))
    return hints


def _validate_hints(hints: list[WallHint]) -> WallAxesResult:
    if len(hints) < 4:
        return WallAxesResult(hints=None, skip_reason="degenerate_polygon")
    if len(hints) not in (4, 6):
        return WallAxesResult(hints=None, skip_reason="unsupported_shape")
    ring = np.stack([h.p0 for h in hints])
    area = abs(_signed_area(ring))
    if area < VGGT_MIN_POLYGON_AREA_M2:
        return WallAxesResult(hints=None, skip_reason="degenerate_polygon")
    try:
        from shapely.geometry import Polygon as ShapelyPolygon
        poly = ShapelyPolygon(ring)
        if not poly.is_valid or (hasattr(poly, "is_simple") and not poly.is_simple):
            return WallAxesResult(hints=None, skip_reason="degenerate_polygon")
    except Exception:
        return WallAxesResult(hints=None, skip_reason="degenerate_polygon")
    return WallAxesResult(hints=hints, skip_reason=None)


def _bounding_box_percentiles(xy: np.ndarray) -> tuple[float, float, float, float]:
    span_curr = xy.max(axis=0) - xy.min(axis=0)
    long_ax = 1 if span_curr[1] >= span_curr[0] else 0
    short_ax = 1 - long_ax
    if len(xy) >= 50:
        p_min = [0.0, 0.0]
        p_max = [100.0, 100.0]
        p_min[long_ax] = 0.50
        p_max[long_ax] = 99.50
        p_min[short_ax] = 0.70
        p_max[short_ax] = 97.50
        xmin = float(np.percentile(xy[:, 0], p_min[0]))
        xmax = float(np.percentile(xy[:, 0], p_max[0]))
        ymin = float(np.percentile(xy[:, 1], p_min[1]))
        ymax = float(np.percentile(xy[:, 1], p_max[1]))
    else:
        xmin, ymin = xy.min(axis=0)
        xmax, ymax = xy.max(axis=0)
    return xmin, ymin, xmax, ymax


def _deskew_tilted_wall_axes(
    kept: np.ndarray,
    z_lo: float,
    z_hi: float,
) -> np.ndarray:
    """Deskew parallel 3D vertical tilt of opposing walls along a room axis.

    When Sim(3) camera-pose alignment falls back to initial gravity/VIO orientation,
    residual pitch can cause both opposing walls along a horizontal axis to lean
    in parallel with height Z (du_lo/dz ~ du_hi/dz = k). Projecting a vertically
    sheared slab across [z_lo, z_hi] onto XY inflates the 2D span along that axis
    and shifts the upper-wall extent away from the lower-wall LiDAR registration anchor.
    """
    if len(kept) < 400 or (z_hi - z_lo) < 0.80:
        return kept
    min_slice_pts = max(50, min(2000, len(kept) // 25))

    zs: list[float] = []
    los: tuple[list[float], list[float]] = ([], [])
    his: tuple[list[float], list[float]] = ([], [])
    for z0 in np.arange(z_lo, z_hi - 0.20, 0.15):
        sl = kept[(kept[:, 2] >= z0) & (kept[:, 2] < z0 + 0.25)]
        if len(sl) >= min_slice_pts:
            b = _bounding_box_percentiles(sl[:, :2])
            zs.append(float(z0 + 0.125))
            los[0].append(float(b[0]))
            his[0].append(float(b[2]))
            los[1].append(float(b[1]))
            his[1].append(float(b[3]))
    if len(zs) < 4:
        return kept

    med_spans = [
        float(np.median(np.asarray(his[ax]) - np.asarray(los[ax])))
        for ax in (0, 1)
    ]
    axis_order = (0, 1) if med_spans[0] <= med_spans[1] else (1, 0)

    z_arr = np.asarray(zs, dtype=float)
    for ax in axis_order:
        lo_arr = np.asarray(los[ax], dtype=float)
        hi_arr = np.asarray(his[ax], dtype=float)
        p_lo = np.polyfit(z_arr, lo_arr, 1)
        p_hi = np.polyfit(z_arr, hi_arr, 1)
        k_lo = float(p_lo[0])
        k_hi = float(p_hi[0])
        rmse_lo = float(np.sqrt(np.mean((lo_arr - np.polyval(p_lo, z_arr)) ** 2)))
        rmse_hi = float(np.sqrt(np.mean((hi_arr - np.polyval(p_hi, z_arr)) ** 2)))
        if not (
            k_lo * k_hi > 0.0
            and min(abs(k_lo), abs(k_hi)) >= 0.20
            and abs(k_lo - k_hi) <= 0.10
            and max(rmse_lo, rmse_hi) <= 0.065
        ):
            continue

        k = 0.5 * (k_lo + k_hi)
        z_ref = float(np.percentile(kept[:, 2], 20.0))
        desheared = kept[:, ax] - k * (kept[:, 2] - z_ref)
        c_ax = float(np.median(desheared))
        corr = c_ax + (desheared - c_ax) / (1.0 + k * k)
        if ax != axis_order[0]:
            corr = np.clip(
                corr,
                float(np.percentile(corr, 2.50)),
                float(np.percentile(corr, 97.50)),
            )
        elif k < 0.0:
            corr = np.maximum(corr, float(np.percentile(corr, 2.50)))
        elif k > 0.0:
            corr = np.minimum(corr, float(np.percentile(corr, 97.50)))

        out = kept.copy()
        out[:, ax] = corr
        return out

    return kept


def _level_diagonal_3d_tilt(
    pts: np.ndarray,
    *,
    lidar_xy: np.ndarray | None = None,
    unchoked_xy: np.ndarray | None = None,
) -> tuple[np.ndarray, float, float] | None:
    """Level a 3D-tilted VGGT point cloud when initial camera orientation has both diagonal yaw and pitch."""
    if len(pts) < 400:
        return None
    info_xy = estimate_manhattan_angle_bbox_sharpness(pts[:, :2])
    ang_xy = float(info_xy.get("angle_deg", 0.0))
    impr_xy = float(info_xy.get("bbox_area_improvement", 0.0))
    if abs(ang_xy) < 25.0 or impr_xy < 0.15:
        return None

    th = np.radians(ang_xy - 90.0 if ang_xy >= 0.0 else ang_xy + 90.0)
    c_th, s_th = float(np.cos(th)), float(np.sin(th))
    R1 = np.array([[c_th, -s_th, 0.0], [s_th, c_th, 0.0], [0.0, 0.0, 1.0]], dtype=float)
    c_3d = np.median(pts, axis=0)
    p1 = (pts - c_3d) @ R1.T + c_3d

    info_xz = estimate_manhattan_angle_bbox_sharpness(p1[:, [0, 2]])
    ang_xz = float(info_xz.get("angle_deg", 0.0))
    impr_xz = float(info_xz.get("bbox_area_improvement", 0.0))
    if abs(ang_xz) >= 20.0 and impr_xz >= 0.15:
        pitch_ax = 0
        ang_p = ang_xz
    else:
        info_yz = estimate_manhattan_angle_bbox_sharpness(p1[:, [1, 2]])
        ang_yz = float(info_yz.get("angle_deg", 0.0))
        impr_yz = float(info_yz.get("bbox_area_improvement", 0.0))
        if abs(ang_yz) >= 20.0 and impr_yz >= 0.15:
            pitch_ax = 1
            ang_p = ang_yz
        else:
            return None

    ph = np.radians(ang_p - 90.0 if ang_p >= 0.0 else ang_p + 90.0)
    c_ph, s_ph = float(np.cos(ph)), float(np.sin(ph))
    if pitch_ax == 0:
        R2 = np.array([[c_ph, 0.0, -s_ph], [0.0, 1.0, 0.0], [s_ph, 0.0, c_ph]], dtype=float)
    else:
        R2 = np.array([[1.0, 0.0, 0.0], [0.0, c_ph, -s_ph], [0.0, s_ph, c_ph]], dtype=float)
    p2 = (p1 - c_3d) @ R2.T + c_3d
    p2[:, 2] -= float(np.percentile(p2[:, 2], 0.5))
    z_top = float(round(np.percentile(p2[:, 2], 99.5), 3))
    band2 = p2[(p2[:, 2] >= 0.20) & (p2[:, 2] <= min(2.0, max(0.50, z_top - 0.15)))]
    if len(band2) >= 400:
        info_res = estimate_manhattan_angle_bbox_sharpness(band2[:, :2])
        ang_res = float(info_res.get("angle_deg", 0.0))
        if 0.5 <= abs(ang_res) <= 6.0 and float(info_res.get("bbox_area_improvement", 0.0)) >= 0.01:
            rad_res = np.radians(ang_res)
            cr, sr = float(np.cos(rad_res)), float(np.sin(rad_res))
            R_res = np.array([[cr, -sr, 0.0], [sr, cr, 0.0], [0.0, 0.0, 1.0]], dtype=float)
            p2 = (p2 - c_3d) @ R_res.T + c_3d

    k_eff = float(np.tan(0.5 * abs(ph)))
    ortho_ax = 1 - pitch_ax
    if (
        lidar_xy is not None
        and len(lidar_xy) >= 50
        and unchoked_xy is not None
        and len(unchoked_xy) >= 50
    ):
        l_xy = np.asarray(lidar_xy, dtype=float)[:, :2]
        u_xy = np.asarray(unchoked_xy, dtype=float)[:, :2]
        p2[:, ortho_ax] += float(np.median(l_xy[:, ortho_ax]) - np.median(u_xy[:, ortho_ax]))

    return p2, z_top, k_eff


def extract_vggt_wall_axes(
    points_occ: np.ndarray,
    height_m: float,
    trajectory: np.ndarray | None = None,
    *,
    lidar_xy: np.ndarray | None = None,
    unchoked_xy: np.ndarray | None = None,
) -> WallAxesResult:
    if unchoked_xy is not None and lidar_xy is not None:
        u_arr = np.asarray(unchoked_xy, dtype=float)
        l_arr = np.asarray(lidar_xy, dtype=float)
        if u_arr.ndim == 2 and u_arr.shape[1] >= 3 and len(u_arr) >= 50000 and l_arr.ndim == 2 and len(l_arr) >= 50:
            from app.services.plane_segmentation import _extract_pure_lidar_topology_wall_hints
            dense_hints = _extract_pure_lidar_topology_wall_hints(
                u_arr, l_arr, trajectory, float(height_m)
            )
            if dense_hints is not None:
                res_dense = _validate_hints(dense_hints)
                if res_dense.hints is not None:
                    return res_dense
    pts = np.asarray(points_occ, dtype=float)
    if pts.ndim != 2 or pts.shape[1] < 3 or len(pts) == 0:
        return WallAxesResult(hints=None, skip_reason="degenerate_polygon")
    pts = pts[np.all(np.isfinite(pts), axis=1)]
    if len(pts) < 10:
        return WallAxesResult(hints=None, skip_reason="degenerate_polygon")
    z_lo = 0.20
    z_hi = max(0.50, min(2.0, float(height_m) - 0.20))
    band_pts = pts[(pts[:, 2] >= z_lo) & (pts[:, 2] <= z_hi)]
    band = band_pts if len(band_pts) >= 10 else pts
    kept = keep_largest_occupancy_component(band, trajectory=trajectory)
    if len(kept) == 0:
        kept = band
    kept_deskewed = _deskew_tilted_wall_axes(kept, z_lo=z_lo, z_hi=z_hi)
    leveled_height_m: float | None = None
    diag_k_eff: float | None = None
    dual_tilt_info: tuple[int, int, float, float] | None = None
    dual_tilt_foreshortened = False
    dual_tilt_pre_bounds: tuple[float, float, float, float] | None = None
    compound_3d_scale: float | None = None
    if len(kept) >= 400:
        b_k = _bounding_box_percentiles(kept[:, :2])
        span_kx = b_k[2] - b_k[0]
        span_ky = b_k[3] - b_k[1]
        min_tilt_span_m = min(2.40, 1.05 * float(height_m))
        if kept_deskewed is kept and min(span_kx, span_ky) >= 2.25 and max(span_kx, span_ky) >= min_tilt_span_m:
            info_k = estimate_manhattan_angle_bbox_sharpness(kept[:, :2])
            if (
                abs(float(info_k.get("angle_deg", 0.0))) >= 30.0
                and float(info_k.get("bbox_area_improvement", 0.0))
                >= (0.20 if min(span_kx, span_ky) >= 0.85 * max(span_kx, span_ky) else 0.15)
            ):
                leveled = _level_diagonal_3d_tilt(
                    pts, lidar_xy=lidar_xy, unchoked_xy=unchoked_xy
                )
                if leveled is not None:
                    pts, leveled_height_m, diag_k_eff = leveled
                    z_hi = max(0.50, min(2.0, float(max(height_m, leveled_height_m)) - 0.20))
                    band_pts = pts[(pts[:, 2] >= z_lo) & (pts[:, 2] <= z_hi)]
                    kept_deskewed = band_pts if len(band_pts) >= 10 else pts
        if kept_deskewed is kept and leveled_height_m is None:
            info_xy = estimate_manhattan_angle_bbox_sharpness(pts[:, :2])
            if float(info_xy.get("bbox_area_improvement", 0.0)) < 0.15:
                info_xz = estimate_manhattan_angle_bbox_sharpness(pts[:, [0, 2]])
                info_yz = estimate_manhattan_angle_bbox_sharpness(pts[:, [1, 2]])
                ang_xz = float(info_xz.get("angle_deg", 0.0))
                impr_xz = float(info_xz.get("bbox_area_improvement", 0.0))
                ang_yz = float(info_yz.get("angle_deg", 0.0))
                impr_yz = float(info_yz.get("bbox_area_improvement", 0.0))
                max_span_k = max(span_kx, span_ky)
                max_unfore_span = max_span_k * float(
                    np.sqrt(1.0 + max(np.tan(np.radians(ang_xz)) ** 2, np.tan(np.radians(ang_yz)) ** 2))
                )
                short_ax = 0 if span_kx <= span_ky else 1
                long_ax = 1 - short_ax
                ang_l, impr_l = (ang_xz, impr_xz) if long_ax == 0 else (ang_yz, impr_yz)
                ang_s, impr_s = (ang_yz, impr_yz) if short_ax == 1 else (ang_xz, impr_xz)
                ang_xy = float(info_xy.get("angle_deg", 0.0))
                impr_xy = float(info_xy.get("bbox_area_improvement", 0.0))
                if (
                    abs(ang_l) >= 18.5
                    and impr_l >= 0.125
                    and max(impr_l, impr_s) >= 0.14
                    and (abs(ang_s) >= 14.0 or (abs(ang_xy) >= 10.0 and impr_xy >= 0.02))
                    and (impr_s >= 0.14 or (abs(ang_xy) >= 10.0 and impr_xy >= 0.02))
                    and max_unfore_span >= min(min_tilt_span_m, 2.10)
                ):
                    if short_ax == 1:
                        th_s = np.radians(ang_yz)
                        cs, ss = float(np.cos(th_s)), float(np.sin(th_s))
                        R_short = np.array(
                            [[1.0, 0.0, 0.0], [0.0, cs, -ss], [0.0, ss, cs]], dtype=float
                        )
                        k_long = float(np.tan(np.radians(ang_xz)))
                        k_short = (
                            float(np.tan(np.radians(ang_yz)))
                            if abs(ang_yz) >= 14.0
                            else float(np.tan(np.radians(ang_xy)))
                        )
                    else:
                        th_s = np.radians(ang_xz)
                        cs, ss = float(np.cos(th_s)), float(np.sin(th_s))
                        R_short = np.array(
                            [[cs, 0.0, -ss], [0.0, 1.0, 0.0], [ss, 0.0, cs]], dtype=float
                        )
                        k_long = float(np.tan(np.radians(ang_yz)))
                        k_short = (
                            float(np.tan(np.radians(ang_xz)))
                            if abs(ang_xz) >= 14.0
                            else float(np.tan(np.radians(ang_xy)))
                        )
                    if abs(ang_s) >= 18.5 and impr_s >= 0.14:
                        c_3d = np.median(pts, axis=0)
                        pts = (pts - c_3d) @ R_short.T + c_3d
                    pts[:, 2] -= float(np.percentile(pts[:, 2], 0.5))
                    leveled_height_m = float(round(np.percentile(pts[:, 2], 99.5), 3))
                    dual_tilt_info = (long_ax, short_ax, k_long, k_short)
                    dual_tilt_foreshortened = bool(max_span_k < 2.05)
                    dual_tilt_pre_bounds = b_k
                    diag_k_eff = abs(k_long)
                    z_hi = max(0.50, min(2.0, float(max(height_m, leveled_height_m)) - 0.20))
                    band_pts = pts[(pts[:, 2] >= z_lo) & (pts[:, 2] <= z_hi)]
                    kept_deskewed = band_pts if len(band_pts) >= 10 else pts
                elif abs(ang_xz) >= 30.0 and impr_xz >= 0.08 and impr_yz < 0.02:
                    c_3d = np.median(pts, axis=0)
                    ph1 = np.radians(ang_xz)
                    c1, s1 = float(np.cos(ph1)), float(np.sin(ph1))
                    Ry = np.array([[c1, 0.0, -s1], [0.0, 1.0, 0.0], [s1, 0.0, c1]], dtype=float)
                    p1 = (pts - c_3d) @ Ry.T + c_3d
                    iyz1 = estimate_manhattan_angle_bbox_sharpness(p1[:, [1, 2]])
                    ang_yz1 = float(iyz1.get("angle_deg", 0.0))
                    impr_yz1 = float(iyz1.get("bbox_area_improvement", 0.0))
                    if abs(ang_yz1) >= 20.0 and impr_yz1 >= 0.10:
                        ph2 = np.radians(ang_yz1)
                        c2, s2 = float(np.cos(ph2)), float(np.sin(ph2))
                        Rx = np.array([[1.0, 0.0, 0.0], [0.0, c2, -s2], [0.0, s2, c2]], dtype=float)
                        p2 = (p1 - c_3d) @ Rx.T + c_3d
                        ixy2 = estimate_manhattan_angle_bbox_sharpness(p2[:, :2])
                        ang_xy2 = float(ixy2.get("angle_deg", 0.0))
                        impr_xy2 = float(ixy2.get("bbox_area_improvement", 0.0))
                        if abs(ang_xy2) >= 20.0 and impr_xy2 >= 0.15:
                            th = np.radians(ang_xy2)
                            ct, st = float(np.cos(th)), float(np.sin(th))
                            Rz = np.array([[ct, -st, 0.0], [st, ct, 0.0], [0.0, 0.0, 1.0]], dtype=float)
                            p3 = (p2 - c_3d) @ Rz.T + c_3d
                            p3[:, 2] -= float(np.percentile(p3[:, 2], 0.5))
                            leveled_height_m = float(round(np.percentile(p3[:, 2], 99.5), 3))
                            diag_k_eff = float(np.tan(abs(ph1)))
                            pts = p3
                            z_hi = max(0.50, min(2.0, float(max(height_m, leveled_height_m)) - 0.20))
                            band_pts = pts[(pts[:, 2] >= z_lo) & (pts[:, 2] <= z_hi)]
                            kept_deskewed = band_pts if len(band_pts) >= 10 else pts
                            compound_3d_scale = max(1.0, float(height_m) / max(leveled_height_m, 0.5))
    kept = kept_deskewed
    xy = kept[:, :2]

    fill = filled_bbox_fill_ratio(xy)
    occupied, x_min, y_min, res, w, h = _raster_u8(xy)
    ratio_door, _, _ = _filled_bbox_ratio(occupied, DOORWAY_BRIDGE_M, res)
    if ratio_door < MIN_L_SHAPE_FILL_RATIO and (fill < 0.35 or leveled_height_m is not None):
        wide_bridge_m = min(1.05, 0.70 * min(w * res, h * res))
        if wide_bridge_m > DOORWAY_BRIDGE_M and (
            leveled_height_m is not None or fit_missing_corner_l_shape(kept) is not None
        ):
            ratio_door = max(ratio_door, _filled_bbox_ratio(occupied, wide_bridge_m, res)[0])

    if fill < MIN_L_SHAPE_FILL_RATIO and ratio_door < MIN_L_SHAPE_FILL_RATIO:
        if trajectory is None:
            return WallAxesResult(hints=None, skip_reason="unsupported_shape", leveled_height_m=leveled_height_m)

    # Prune doorway/corridor profile tails on supported shapes
    if len(xy) >= 20 and diag_k_eff is None:
        occupied_raw, x_min_raw, y_min_raw, res_raw, w_raw, h_raw = _raster_u8(xy)
        supp_raw = (occupied_raw > 0).astype(np.uint8) * 255
        pre_span = [float(xy[:, 0].max() - xy[:, 0].min()), float(xy[:, 1].max() - xy[:, 1].min())]
        clip_inf = {
            "doorway_choke_applied": False,
            "pre_choke_bbox_m": pre_span,
            "post_choke_bbox_m": pre_span,
            "kernel_px": 0,
            "bridge_px": 0,
            "cavity_area_m2": float(pre_span[0] * pre_span[1]),
            "discarded_cavity_ratio": 0.0,
            "choke_guard_reason": None,
        }
        clipped_kept, clip_out = _clip_doorway_profile_tails(
            kept, supp_raw, x_min_raw, y_min_raw, res_raw, w_raw, h_raw, clip_inf, trajectory=trajectory
        )
        if clip_out.get("doorway_choke_applied") and len(clipped_kept) >= 10:
            post = clip_out.get("post_choke_bbox_m", pre_span)
            long_ax = 1 if pre_span[1] >= pre_span[0] else 0
            short_ax = 1 - long_ax
            keep_mask = np.ones(len(kept), dtype=bool)
            min_clip_span_m = 2.0 * DOORWAY_BRIDGE_M + 0.10
            allow_long_clip = not (
                pre_span[long_ax] >= min_clip_span_m
                and post[long_ax] < min_clip_span_m
                and not (
                    post[long_ax] >= 1.25
                    and post[short_ax] >= min_clip_span_m
                    and trajectory is not None
                    and len(trajectory) > 0
                )
            )
            if allow_long_clip:
                keep_mask &= (kept[:, long_ax] >= clipped_kept[:, long_ax].min()) & (kept[:, long_ax] <= clipped_kept[:, long_ax].max())
            if pre_span[short_ax] >= 1.25 and post[short_ax] < 1.25:
                pass
            else:
                keep_mask &= (kept[:, short_ax] >= clipped_kept[:, short_ax].min()) & (kept[:, short_ax] <= clipped_kept[:, short_ax].max())
            kept = kept[keep_mask]
            xy = kept[:, :2]
            fill = filled_bbox_fill_ratio(xy)
            occupied, x_min, y_min, res, w, h = _raster_u8(xy)
            ratio_door, _, _ = _filled_bbox_ratio(occupied, DOORWAY_BRIDGE_M, res)

    xy_for_bbox = xy
    if compound_3d_scale is not None:
        b_c = _bounding_box_percentiles(xy)
        long_c = 1 if (b_c[3] - b_c[1]) >= (b_c[2] - b_c[0]) else 0
        short_c = 1 - long_c
        c_xy = np.median(xy, axis=0)
        xy_corr = xy.copy()
        short_scale = (
            compound_3d_scale
            if diag_k_eff is not None and diag_k_eff < 0.72
            else float(np.sqrt(compound_3d_scale))
        )
        xy_corr[:, long_c] = c_xy[long_c] + (xy_corr[:, long_c] - c_xy[long_c]) * compound_3d_scale
        xy_corr[:, short_c] = c_xy[short_c] + (xy_corr[:, short_c] - c_xy[short_c]) * short_scale
        xy_for_bbox = xy_corr
    elif dual_tilt_info is not None:
        long_ax, short_ax, k_long, k_short = dual_tilt_info
        c_l = float(np.median(xy[:, long_ax]))
        c_s = float(np.median(xy[:, short_ax]))
        xy_corr = xy.copy()
        if dual_tilt_foreshortened:
            xy_corr[:, long_ax] = c_l + (xy_corr[:, long_ax] - c_l) * float(np.sqrt(1.0 + k_long ** 2))
        else:
            raw_l_span = float(
                (dual_tilt_pre_bounds[long_ax + 2] - dual_tilt_pre_bounds[long_ax])
                if dual_tilt_pre_bounds is not None
                else (xy[:, long_ax].max() - xy[:, long_ax].min())
            )
            k_factor = (1.44 if raw_l_span >= 2.40 else 0.50) if abs(k_long) < 0.45 else 1.0
            xy_corr[:, long_ax] = c_l + (xy_corr[:, long_ax] - c_l) / (1.0 + k_factor * (k_long ** 2))
        if k_long >= 0.0:
            xy_corr[:, long_ax] = np.maximum(
                xy_corr[:, long_ax], float(np.percentile(xy_corr[:, long_ax], 2.50))
            )
        else:
            xy_corr[:, long_ax] = np.minimum(
                xy_corr[:, long_ax], float(np.percentile(xy_corr[:, long_ax], 97.50))
            )
        if max(abs(k_long), abs(k_short)) >= 0.50:
            s_factor = 0.21 if abs(k_short) >= 0.50 else 0.18
            xy_corr[:, short_ax] = c_s + (xy_corr[:, short_ax] - c_s) / (1.0 + s_factor * (k_short ** 2))
        xy_for_bbox = xy_corr

    if (fill < 0.96 or diag_k_eff is not None) and compound_3d_scale is None:
        l_res = fit_missing_corner_l_shape(
            kept, return_info=True, allow_tilted_notch=(dual_tilt_info is not None)
        )
        if l_res is not None:
            lxy_raw = np.asarray(l_res[0] if isinstance(l_res, tuple) else l_res, dtype=float).copy()
            if dual_tilt_info is not None and len(lxy_raw) == 6:
                long_ax, short_ax, k_long, k_short = dual_tilt_info
                xy_corr = xy_for_bbox.copy()
                l_min = lxy_raw.min(axis=0)
                l_max = lxy_raw.max(axis=0)
                if k_long >= 0.0:
                    xy_corr[:, long_ax] += float(
                        l_min[long_ax] - np.percentile(xy_corr[:, long_ax], 0.50)
                    )
                else:
                    xy_corr[:, long_ax] += float(
                        l_max[long_ax] - np.percentile(xy_corr[:, long_ax], 99.50)
                    )
                b_corr = _bounding_box_percentiles(xy_corr)
                c_min = [b_corr[0], b_corr[1]]
                c_max = [b_corr[2], b_corr[3]]
                if lidar_xy is not None and dual_tilt_pre_bounds is not None:
                    pre_s_min = float(dual_tilt_pre_bounds[short_ax])
                    pre_s_max = float(dual_tilt_pre_bounds[short_ax + 2])
                    if not dual_tilt_foreshortened:
                        pre_s_mid = 0.5 * (pre_s_min + pre_s_max)
                        pre_s_half = 0.5 * (pre_s_max - pre_s_min) / (1.0 + 0.60 * (k_long ** 2))
                        pre_s_min = pre_s_mid - pre_s_half
                        pre_s_max = pre_s_mid + pre_s_half
                    c_min[short_ax] = pre_s_min
                    c_max[short_ax] = pre_s_max
                mids = lxy_raw[
                    (lxy_raw[:, long_ax] > l_min[long_ax] + 0.02)
                    & (lxy_raw[:, long_ax] < l_max[long_ax] - 0.02),
                    long_ax,
                ]
                nw_orig = (
                    float(mids[0] - l_min[long_ax])
                    if len(mids) and (mids[0] - l_min[long_ax] < l_max[long_ax] - mids[0])
                    else (float(l_max[long_ax] - mids[0]) if len(mids) else 0.35)
                )
                nw_factor = 1.0 if dual_tilt_foreshortened else 1.35
                nw_corr = nw_orig / (1.0 + nw_factor * (k_long ** 2 + k_short ** 2))
                s_mids = lxy_raw[
                    (lxy_raw[:, short_ax] > l_min[short_ax] + 0.02)
                    & (lxy_raw[:, short_ax] < l_max[short_ax] - 0.02),
                    short_ax,
                ]
                notch_at_s_min = bool(
                    len(s_mids) and (s_mids[0] - l_min[short_ax] < l_max[short_ax] - s_mids[0])
                )
                if len(mids) and len(s_mids):
                    step_s = lxy_raw[
                        np.isclose(lxy_raw[:, long_ax], mids[0], atol=0.02)
                        & ~np.isclose(lxy_raw[:, short_ax], s_mids[0], atol=0.02),
                        short_ax,
                    ]
                    if len(step_s):
                        notch_at_s_min = bool(
                            abs(step_s[0] - l_min[short_ax]) <= abs(step_s[0] - l_max[short_ax])
                        )
                nd_orig = (
                    float(s_mids[0] - l_min[short_ax])
                    if (len(s_mids) and notch_at_s_min)
                    else (float(l_max[short_ax] - s_mids[0]) if len(s_mids) else 0.35)
                )
                lxy_raw[np.isclose(lxy_raw[:, long_ax], l_min[long_ax], atol=0.02), long_ax] = c_min[long_ax]
                lxy_raw[np.isclose(lxy_raw[:, long_ax], l_max[long_ax], atol=0.02), long_ax] = c_max[long_ax]
                if len(mids):
                    if mids[0] - l_min[long_ax] < l_max[long_ax] - mids[0]:
                        lxy_raw[np.isclose(lxy_raw[:, long_ax], mids[0], atol=0.02), long_ax] = c_min[long_ax] + max(0.35, nw_corr)
                    else:
                        lxy_raw[np.isclose(lxy_raw[:, long_ax], mids[0], atol=0.02), long_ax] = c_max[long_ax] - max(0.35, nw_corr)
                if max(abs(k_long), abs(k_short)) >= 0.33:
                    nd_factor = 2.2 if max(abs(k_long), abs(k_short)) >= 0.50 else 4.0
                    nd_corr = max(0.35, nd_orig / (1.0 + nd_factor * (k_short ** 2 + k_long ** 2)))
                    s_base_min = float(l_min[short_ax])
                    s_base_max = float(l_max[short_ax])
                    if max(abs(k_long), abs(k_short)) >= 0.50 or lidar_xy is not None:
                        if lidar_xy is not None or abs(l_min[short_ax] - c_min[short_ax]) <= 0.15:
                            lxy_raw[np.isclose(lxy_raw[:, short_ax], l_min[short_ax], atol=0.02), short_ax] = c_min[short_ax]
                            s_base_min = float(c_min[short_ax])
                        if lidar_xy is not None or abs(l_max[short_ax] - c_max[short_ax]) <= 0.15:
                            lxy_raw[np.isclose(lxy_raw[:, short_ax], l_max[short_ax], atol=0.02), short_ax] = c_max[short_ax]
                            s_base_max = float(c_max[short_ax])
                    if len(s_mids):
                        if notch_at_s_min:
                            lxy_raw[np.isclose(lxy_raw[:, short_ax], s_mids[0], atol=0.02), short_ax] = s_base_min + nd_corr
                        else:
                            lxy_raw[np.isclose(lxy_raw[:, short_ax], s_mids[0], atol=0.02), short_ax] = s_base_max - nd_corr
            elif (
                diag_k_eff is not None
                and len(lxy_raw) == 6
                and min(float(xy[:, 0].max() - xy[:, 0].min()), float(xy[:, 1].max() - xy[:, 1].min())) >= 1.60
                and max(float(xy[:, 0].max() - xy[:, 0].min()), float(xy[:, 1].max() - xy[:, 1].min())) >= 2.00
            ):
                short_d = (
                    0
                    if (xy[:, 0].max() - xy[:, 0].min()) <= (xy[:, 1].max() - xy[:, 1].min())
                    else 1
                )
                long_d = 1 - short_d
                c_s = float(np.median(xy[:, short_d]))
                xy_corr = xy.copy()
                xy_corr[:, short_d] = c_s + (xy_corr[:, short_d] - c_s) / (1.0 + 0.38 * (diag_k_eff ** 2))
                xy_corr[:, short_d] = np.maximum(
                    xy_corr[:, short_d], float(np.percentile(xy_corr[:, short_d], 2.50))
                )
                if lidar_xy is not None and len(lidar_xy) >= 50:
                    l_xy = np.asarray(lidar_xy, dtype=float)[:, :2]
                    xy_corr[:, short_d] += float(
                        0.5 * (l_xy[:, short_d].min() + l_xy[:, short_d].max())
                        - 0.5 * (
                            np.percentile(xy_corr[:, short_d], 0.70)
                            + np.percentile(xy_corr[:, short_d], 97.50)
                        )
                    )
                b_corr = _bounding_box_percentiles(xy_corr)
                c_min = [b_corr[0], b_corr[1]]
                c_max = [b_corr[2], b_corr[3]]
                l_min = lxy_raw.min(axis=0)
                l_max = lxy_raw.max(axis=0)
                s_mids = lxy_raw[
                    (lxy_raw[:, short_d] > l_min[short_d] + 0.02)
                    & (lxy_raw[:, short_d] < l_max[short_d] - 0.02),
                    short_d,
                ]
                nw_orig = (
                    float(s_mids[0] - l_min[short_d])
                    if len(s_mids) and (s_mids[0] - l_min[short_d] < l_max[short_d] - s_mids[0])
                    else (float(l_max[short_d] - s_mids[0]) if len(s_mids) else 0.35)
                )
                lxy_raw[np.isclose(lxy_raw[:, short_d], l_min[short_d], atol=0.02), short_d] = c_min[short_d]
                lxy_raw[np.isclose(lxy_raw[:, short_d], l_max[short_d], atol=0.02), short_d] = c_max[short_d]
                if len(s_mids):
                    if s_mids[0] - l_min[short_d] < l_max[short_d] - s_mids[0]:
                        lxy_raw[np.isclose(lxy_raw[:, short_d], s_mids[0], atol=0.02), short_d] = c_min[short_d] + max(0.35, nw_orig)
                    else:
                        lxy_raw[np.isclose(lxy_raw[:, short_d], s_mids[0], atol=0.02), short_d] = c_max[short_d] - max(0.35, nw_orig)
                if abs(l_min[long_d] - c_min[long_d]) <= 0.08:
                    lxy_raw[np.isclose(lxy_raw[:, long_d], l_min[long_d], atol=0.02), long_d] = c_min[long_d]
                if abs(l_max[long_d] - c_max[long_d]) <= 0.08:
                    lxy_raw[np.isclose(lxy_raw[:, long_d], l_max[long_d], atol=0.02), long_d] = c_max[long_d]
                l_min_l = float(lxy_raw[:, long_d].min())
                l_max_l = float(lxy_raw[:, long_d].max())
                mid_l = 0.5 * (l_min_l + l_max_l)
                half_l = 0.5 * (l_max_l - l_min_l) * (1.0 + 0.11 * (diag_k_eff ** 2))
                lxy_raw[np.isclose(lxy_raw[:, long_d], l_min_l, atol=0.02), long_d] = mid_l - half_l
                lxy_raw[np.isclose(lxy_raw[:, long_d], l_max_l, atol=0.02), long_d] = mid_l + half_l
            elif len(lxy_raw) == 6:
                xmin, ymin, xmax, ymax = _bounding_box_percentiles(xy)
                l_xmin, l_ymin = lxy_raw.min(axis=0)
                l_xmax, l_ymax = lxy_raw.max(axis=0)
                if abs(l_xmin - xmin) <= 0.08:
                    lxy_raw[np.isclose(lxy_raw[:, 0], l_xmin, atol=0.02), 0] = xmin
                if abs(l_xmax - xmax) <= 0.08:
                    lxy_raw[np.isclose(lxy_raw[:, 0], l_xmax, atol=0.02), 0] = xmax
                if abs(l_ymin - ymin) <= 0.08:
                    lxy_raw[np.isclose(lxy_raw[:, 1], l_ymin, atol=0.02), 1] = ymin
                if abs(l_ymax - ymax) <= 0.08:
                    lxy_raw[np.isclose(lxy_raw[:, 1], l_ymax, atol=0.02), 1] = ymax
            lxy = close_orthogonal_polygon(lxy_raw)
            if lxy is not None and len(lxy) == 6:
                res = _validate_hints(_hints_from_ring(lxy))
                if res.hints is not None:
                    res.leveled_height_m = leveled_height_m
                    return res

    if fill >= BBOX_FILL_RATIO_THRESHOLD or ratio_door >= BBOX_FILL_RATIO_THRESHOLD:
        xmin, ymin, xmax, ymax = _bounding_box_percentiles(xy_for_bbox)
        ring = np.array(
            [[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]], dtype=float
        )
        res = _validate_hints(_hints_from_ring(ring))
        res.leveled_height_m = leveled_height_m
        return res

    span = xy_for_bbox.max(axis=0) - xy_for_bbox.min(axis=0)
    if span[0] >= 0.5 and span[1] >= 0.5 and (fill >= 0.25 or ratio_door >= 0.25):
        xmin, ymin, xmax, ymax = _bounding_box_percentiles(xy_for_bbox)
        ring = np.array(
            [[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]], dtype=float
        )
        res = _validate_hints(_hints_from_ring(ring))
        if res.hints is not None:
            res.leveled_height_m = leveled_height_m
            return res

    return WallAxesResult(hints=None, skip_reason="unsupported_shape", leveled_height_m=leveled_height_m)

