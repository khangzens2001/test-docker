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
    frame_indices: np.ndarray | None = None


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
    _, S_pts, _ = np.linalg.svd(src_c, full_matrices=False)
    if S_pts.size < 2 or S_pts[0] < 1e-12 or (S_pts[1] / S_pts[0]) < SIM3_SVD_COLLINEAR_RATIO:
        return _empty_sim3(n, "degenerate_sim3")

    min_rms = 0.08 if (n >= 20 and S_pts.size >= 2 and (S_pts[1] / S_pts[0]) >= 0.25) else SIM3_MIN_RMS_RADIUS_M
    if rms_dst < min_rms:
        return _empty_sim3(n, "degenerate_sim3")

    fit_full = _umeyama_sim3(src, dst)
    if fit_full is None:
        return _empty_sim3(n, "degenerate_sim3")
    s, R, t = fit_full
    aligned = apply_sim3(src, s, R, t)
    rmse = float(np.sqrt(np.mean(np.sum((dst - aligned) ** 2, axis=1))))
    if rms_dst < SIM3_MIN_RMS_RADIUS_M and rmse > 0.025:
        return _empty_sim3(n, "degenerate_sim3")

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
                planar_ratio = float(svals[1] / svals[0]) if (svals.size >= 2 and svals[0] > 1e-9) else 0.0
                sig_ratio = (
                    float(svals[-1] / svals[0])
                    if (svals.size >= 3 and svals[0] > 1e-9)
                    else planar_ratio
                )
                if len(c_arr) >= 20 and planar_ratio >= 0.25:
                    if r_rms < 0.08 or sig_ratio < 0.05:
                        cam_degenerate = True
                elif r_rms < 0.20 or sig_ratio < 0.16:
                    cam_degenerate = True

        h_vggt = float(np.percentile(pts_rot[:, 2], 99.5) - np.percentile(pts_rot[:, 2], 0.5))
        if h_vggt < 0.05:
            h_vggt = float(np.ptp(pts_rot[:, 2]))
        if h_vggt < 0.05:
            return False, 1.0, np.eye(3), np.zeros(3), float("inf")

        if initial_s is not None and SIM3_SCALE_MIN <= initial_s <= SIM3_SCALE_MAX and not cam_degenerate:
            s = float(initial_s)
        else:
            h_lidar = float(height_m) if height_m and height_m > 0 else self.extract_floor_ceiling_height(pts_l[:, 2])
            s = float(np.clip(h_lidar / max(h_vggt, 0.05), SIM3_SCALE_MIN, SIM3_SCALE_MAX))
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
        try:
            pcd_lidar_ds.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=20))
            init_trans = np.eye(4)
            init_trans[:3, :3] = best_R_yaw
            init_trans[:3, 3] = best_t
            loss = o3d.pipelines.registration.HuberLoss(self.max_inlier_dist)
            est = o3d.pipelines.registration.TransformationEstimationPointToPlane(loss)
            criteria = o3d.pipelines.registration.ICPConvergenceCriteria(
                max_iteration=30, relative_fitness=1e-6, relative_rmse=1e-6
            )
            icp_res = o3d.pipelines.registration.registration_icp(
                pcd_vggt_ds, pcd_lidar_ds, self.max_inlier_dist * 1.5, init_trans, est, criteria
            )
            if icp_res.inlier_rmse <= self.max_inlier_dist and len(icp_res.correspondence_set) >= min_inliers:
                best_R_yaw = icp_res.transformation[:3, :3]
                best_t = icp_res.transformation[:3, 3]
                best_rmse = float(icp_res.inlier_rmse)
                best_inliers = len(icp_res.correspondence_set)
        except Exception:
            pass

        ok = (best_inliers >= min_inliers) and (best_rmse <= self.max_inlier_dist)
        R_total = best_R_yaw @ R_init
        return ok, s, R_total, best_t, best_rmse


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


def _irls_huber_1d(vals: np.ndarray, init_val: float, delta: float = 0.012, n_iter: int = 5) -> float:
    """Compute continuous 1D location estimate via Iteratively Reweighted Least Squares (Huber M-estimator)."""
    if len(vals) == 0:
        return float(init_val)
    loc = float(np.median(vals))
    for _ in range(n_iter):
        abs_r = np.abs(vals - loc)
        w = np.where(abs_r <= delta, 1.0, delta / np.maximum(abs_r, 1e-9))
        w_sum = float(np.sum(w))
        if w_sum <= 1e-9:
            break
        loc = float(np.sum(w * vals) / w_sum)
    return loc


def _continuous_kde_modes_1d(
    in_win: np.ndarray,
    lo: float,
    hi: float,
    search_hint: float,
    window_m: float = T2_BIN_M,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate continuous 1D kernel density support in [lo, hi] and extract mode centres and support."""
    if len(in_win) == 0 or hi <= lo:
        empty = np.zeros(0, dtype=float)
        return empty, empty, empty, empty
    n_grid = max(81, int(round((hi - lo) / 0.0025)) + 1)
    grid = np.linspace(lo, hi, n_grid)
    sub_edges = np.linspace(lo, hi, n_grid + 1)
    sub_cnts, _ = np.histogram(in_win, bins=sub_edges)
    sub_ctrs = 0.5 * (sub_edges[:-1] + sub_edges[1:])
    diff = np.abs(grid[:, None] - sub_ctrs[None, :])
    flat_r = 0.35 * window_m
    sigma_r = max(0.20 * window_m, 1e-4)
    kern = np.exp(-0.5 * (np.maximum(0.0, diff - flat_r) / sigma_r) ** 2)
    support = kern @ sub_cnts.astype(float)

    local_max_idx: list[int] = []
    for i in range(n_grid):
        left_ok = i == 0 or support[i] >= support[i - 1]
        right_ok = i == n_grid - 1 or support[i] >= support[i + 1]
        strict = (i > 0 and support[i] > support[i - 1]) or (i < n_grid - 1 and support[i] > support[i + 1])
        if left_ok and right_ok and strict and support[i] >= T2_PEAK_MIN_COUNT:
            local_max_idx.append(i)
    if not local_max_idx and float(np.max(support)) >= T2_PEAK_MIN_COUNT:
        local_max_idx.append(int(np.argmax(support)))

    # Only include the grid point closest to search_hint if no stronger true mode exists in the window
    hint_i = int(np.argmin(np.abs(grid - search_hint)))
    if support[hint_i] >= T2_PEAK_MIN_COUNT:
        if all(abs(grid[hint_i] - grid[m_i]) > 0.035 for m_i in local_max_idx) and all(
            float(support[m_i]) <= 1.02 * float(support[hint_i]) for m_i in local_max_idx
        ):
            local_max_idx.append(hint_i)

    # Merge modes closer than 0.030 m (keep the higher-support mode)
    local_max_idx.sort(key=lambda idx: float(support[idx]), reverse=True)
    kept_idx: list[int] = []
    for idx in local_max_idx:
        if all(abs(grid[idx] - grid[k]) > 0.030 for k in kept_idx):
            kept_idx.append(idx)
    kept_idx.sort()

    centres = np.array(
        [
            _irls_huber_1d(in_win[np.abs(in_win - grid[idx]) <= 0.030], float(grid[idx]))
            for idx in kept_idx
        ],
        dtype=float,
    )
    cand_counts = np.array([float(support[idx]) for idx in kept_idx], dtype=float)
    return centres, cand_counts, grid, support


def apply_vggt_topology_prior(lidar_xy: np.ndarray, hints: list[WallHint]) -> T2Result:
    xy = np.asarray(lidar_xy, dtype=float).reshape(-1, 2)
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
            seg_len = float(np.linalg.norm(p1_2d - p0_2d))
            if seg_len > 0.05:
                t_vec = np.array([-n[1], n[0]], dtype=float)
                t0, t1 = sorted([float(p0_2d @ t_vec), float(p1_2d @ t_vec)])
                t_proj = xy @ t_vec
                pad = 0.025 if seg_len >= 0.12 else 0.0
                xy_wall = xy[(t_proj >= t0 + pad) & (t_proj <= t1 - pad)]
        proj = xy_wall @ n if len(xy_wall) else np.zeros(0)
        search_hint = pos_hint
        win_m = 0.12 if is_inner_l_wall else T2_WINDOW_M
        lo = search_hint - win_m
        hi = search_hint + win_m
        in_win = proj[(proj >= lo) & (proj <= hi)]
        centres, cand_counts, grid, support = _continuous_kde_modes_1d(in_win, lo, hi, search_hint)
        if is_inner_l_wall and len(centres) == 0:
            win_m = T2_WINDOW_M
            lo = search_hint - win_m
            hi = search_hint + win_m
            in_win = proj[(proj >= lo) & (proj <= hi)]
            centres, cand_counts, grid, support = _continuous_kde_modes_1d(in_win, lo, hi, search_hint)
        if len(centres) == 0:
            walls.append(T2Wall(n=n, pos_hint=pos_hint, pos_metric=pos_hint, source="vggt_hint"))
            continue

        c_proj = float(centroid @ n)
        dist_cost = np.abs(centres - search_hint)
        order = np.lexsort((centres, np.abs(centres - c_proj), -cand_counts, np.round(dist_cost, 2)))
        best_hint_idx = int(order[0])
        c_hint = float(centres[best_hint_idx])
        cnt_hint = float(cand_counts[best_hint_idx])

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

        # Structural wall behind interior fixture check:
        # If the selected mode sits on an interior fixture while LiDAR points form a stronger
        # structural wall mode further outward (without jumping across an exterior doorway gap
        # when c_hint is already at the VGGT wall hint):
        if opp_effective is not None and vggt_span is not None:
            structural_candidates = []
            for j in range(len(centres)):
                if (centres[j] - c_hint) < 0.04:
                    continue
                if (centres[j] + opp_effective) > vggt_span + T2_WINDOW_M:
                    continue
                # If c_hint is already aligned with pos_hint (within 3.5 cm), do not jump across a
                # deep density valley into exterior leakage beyond pos_hint
                if abs(c_hint - pos_hint) <= 0.035 and centres[j] > pos_hint + 0.04:
                    in_between = (grid >= c_hint) & (grid <= centres[j])
                    valley_min = float(np.min(support[in_between])) if np.any(in_between) else 0.0
                    if valley_min < 0.65 * cnt_hint:
                        continue
                if cand_counts[j] >= 500 and (
                    (opp_wall is not None and cand_counts[j] >= 2.5 * cnt_hint)
                    or (
                        cand_counts[j] >= max(1000.0, 0.85 * cnt_hint)
                        and pos_hint > c_hint + 0.02
                        and (centres[j] + opp_effective) <= vggt_span + 0.12
                    )
                ):
                    structural_candidates.append(j)
            if structural_candidates:
                best_outer = max(structural_candidates, key=lambda j: float(cand_counts[j]))
                c_outer = float(centres[best_outer])
                inliers = in_win[np.abs(in_win - c_outer) <= 0.035]
                pos_metric = _irls_huber_1d(inliers, c_outer)
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
        lidar_dense_at_hint = (
            np.count_nonzero((proj >= pos_hint - 0.05) & (proj <= pos_hint + 0.05)) >= 50
            if len(proj)
            else False
        )
        if (
            opp_pos is not None
            and (pos_hint - c_hint) >= 0.05
            and max_p < pos_hint - 0.015
            and not lidar_dense_at_hint
        ):
            walls.append(T2Wall(n=n, pos_hint=pos_hint, pos_metric=pos_hint, source="vggt_hint"))
            continue

        if opp_effective is not None and vggt_span is not None:
            current_span = float(c_hint + opp_effective)
            span_deficit = vggt_span - current_span
            if (c_hint - pos_hint >= 0.025) and span_deficit <= -0.04:
                # Doorway aperture leakage past room boundary into hallway/exterior
                inner_candidates = [
                    j
                    for j in range(len(centres))
                    if (c_hint - centres[j] >= 0.02)
                    and (c_hint - centres[j] <= T2_WINDOW_M)
                    and (abs(centres[j] - pos_hint) <= 0.06)
                    and (abs((centres[j] + opp_effective) - vggt_span) <= 0.06)
                    and (cand_counts[j] >= T2_PEAK_MIN_COUNT)
                ]
                if inner_candidates:
                    best_inner = min(
                        inner_candidates,
                        key=lambda j: abs((centres[j] + opp_effective) - vggt_span),
                    )
                    c_inner = float(centres[best_inner])
                    pos_metric = _irls_huber_1d(in_win[np.abs(in_win - c_inner) <= 0.035], c_inner)
                    walls.append(T2Wall(n=n, pos_hint=pos_hint, pos_metric=pos_metric, source="lidar_peak"))
                    continue
                elif cnt_hint < 1000.0:
                    pos_metric = min(pos_hint, vggt_span - opp_effective)
                    walls.append(T2Wall(n=n, pos_hint=pos_hint, pos_metric=pos_metric, source="vggt_hint"))
                    continue

        inliers = in_win[np.abs(in_win - c_hint) <= 0.035]
        pos_metric = _irls_huber_1d(inliers, c_hint) if len(inliers) >= 5 else float(c_hint)
        walls.append(T2Wall(n=n, pos_hint=pos_hint, pos_metric=pos_metric, source="lidar_peak"))

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


def _match_frames(
    frames: list[dict],
    vio_poses: dict[int, list[float]],
    cam_by: dict[str, dict],
    npz_basenames: set[str] | None = None,
) -> list[dict]:
    """Match VGGT keyframe manifest frames strictly against VIO poses.
    Never falls back to idx - 1 or arbitrary adjacent poses.
    """
    matched = []
    for fr in frames:
        fn = _basename(fr.get("vggt_filename", ""))
        if fn not in cam_by:
            continue
        if npz_basenames and fn not in npz_basenames:
            continue
        center = _camera_center(cam_by[fn])
        if center is None:
            continue

        s_idx = fr.get("source_frame_index")
        if s_idx is None and "source_filename" in fr:
            try:
                s_idx = int(str(fr["source_filename"]).split(".")[0])
            except (ValueError, TypeError):
                s_idx = None

        if s_idx is not None and s_idx in vio_poses:
            t = vio_poses[s_idx]
        elif {"x", "y", "z"}.issubset(fr.keys()):
            t = [float(fr["x"]), float(fr["y"]), float(fr["z"])]
        else:
            # Strictly skip frame if no exact corresponding pose is found; NEVER fallback to idx - 1
            continue

        matched.append({
            "vggt_filename": fn,
            "source_frame_index": s_idx,
            "t_vio": t,
            "center": center,
            "frame": fr,
        })
    return matched


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

        matched = _match_frames(frames, vio_poses, cam_by, npz_basenames)
        if len(matched) < SIM3_MIN_CAMERAS:
            return VggtPriorResult(prior=None, skip_reason="too_few_correspondences")
        names = [m["vggt_filename"] for m in matched]
        t_vio = [m["t_vio"] for m in matched]
        C_vggt = [m["center"] for m in matched]
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
            is_z_up_vio = bool(vio_poses) or manifest.get("pose_source") != "arcore_odometry"
            vert_row = 2 if is_z_up_vio else 1
            if (
                R0 is not None
                and cam_doc.get("image_orientation") == "upright"
                and abs(float(R0[vert_row, 0])) > abs(float(R0[vert_row, 1]))
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

        fidx = np.asarray(npz["frame_indices"], dtype=np.int32) if "frame_indices" in npz else None
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
            frame_indices=fidx,
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
    if len(xy) >= 50:
        xmin, xmax = [float(v) for v in np.percentile(xy[:, 0], [0.50, 99.50])]
        ymin, ymax = [float(v) for v in np.percentile(xy[:, 1], [0.50, 99.50])]
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
            and min(abs(k_lo), abs(k_hi)) >= 0.16
            and abs(k_lo - k_hi) <= 0.12
            and max(rmse_lo, rmse_hi) <= 0.065
        ):
            continue

        k = 0.5 * (k_lo + k_hi)
        z_ref = float(np.percentile(kept[:, 2], 20.0))
        desheared = kept[:, ax] - k * (kept[:, 2] - z_ref)

        out = kept.copy()
        out[:, ax] = desheared
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
    if len(kept) >= 400:
        for ax_g in (0, 1):
            v_g = kept[:, ax_g]
            if float(np.ptp(v_g)) < 1.50:
                continue
            v_sorted = np.sort(v_g)
            gaps_g = np.diff(v_sorted)
            big_gaps = np.flatnonzero(gaps_g >= 0.30)
            if len(big_gaps) == 0:
                continue
            t_ax = None
            if trajectory is not None:
                t_arr = np.asarray(trajectory, dtype=float)
                if t_arr.ndim == 2 and len(t_arr) > 0 and t_arr.shape[1] >= 2:
                    t_ax = t_arr[np.isfinite(t_arr[:, ax_g]), ax_g]
            for g_i in big_gaps:
                cut_mid = 0.5 * float(v_sorted[g_i] + v_sorted[g_i + 1])
                n_below = int(np.count_nonzero(v_g < cut_mid))
                n_above = int(np.count_nonzero(v_g >= cut_mid))
                if n_below < 0.05 * len(v_g) and (t_ax is None or len(t_ax) == 0 or np.all(t_ax >= cut_mid)):
                    kept = kept[kept[:, ax_g] >= cut_mid]
                    v_g = kept[:, ax_g]
                elif n_above < 0.05 * len(v_g) and (t_ax is None or len(t_ax) == 0 or np.all(t_ax < cut_mid)):
                    kept = kept[kept[:, ax_g] < cut_mid]
                    v_g = kept[:, ax_g]
    kept_deskewed = _deskew_tilted_wall_axes(kept, z_lo=z_lo, z_hi=z_hi)
    has_deskew_tilt = kept_deskewed is not kept
    leveled_height_m: float | None = None
    if len(kept) >= 400:
        b_k = _bounding_box_percentiles(kept[:, :2])
        span_kx = b_k[2] - b_k[0]
        span_ky = b_k[3] - b_k[1]
        min_tilt_span_m = 0.85 * float(height_m)
        if kept_deskewed is kept and min(span_kx, span_ky) >= 0.80 * float(height_m) and max(span_kx, span_ky) >= min_tilt_span_m:
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
                    pts, leveled_height_m, _ = leveled
                    z_hi = max(0.50, min(2.0, float(max(height_m, leveled_height_m)) - 0.20))
                    band_pts = pts[(pts[:, 2] >= z_lo) & (pts[:, 2] <= z_hi)]
                    kept_deskewed = band_pts if len(band_pts) >= 10 else pts
    kept = kept_deskewed
    xy = kept[:, :2]

    fill = filled_bbox_fill_ratio(xy)
    occupied, x_min, y_min, res, w, h = _raster_u8(xy)
    ratio_door, _, _ = _filled_bbox_ratio(occupied, DOORWAY_BRIDGE_M, res)
    if ratio_door < MIN_L_SHAPE_FILL_RATIO and (fill < 0.35 or leveled_height_m is not None):
        wide_bridge_m = min(1.05, 0.70 * min(w * res, h * res))
        if wide_bridge_m > DOORWAY_BRIDGE_M and (
            leveled_height_m is not None
            or fit_missing_corner_l_shape(
                kept, allow_tilted_notch=has_deskew_tilt
            )
            is not None
        ):
            ratio_door = max(ratio_door, _filled_bbox_ratio(occupied, wide_bridge_m, res)[0])

    if fill < MIN_L_SHAPE_FILL_RATIO and ratio_door < MIN_L_SHAPE_FILL_RATIO:
        if trajectory is None:
            return WallAxesResult(hints=None, skip_reason="unsupported_shape", leveled_height_m=leveled_height_m)

    # Prune doorway/corridor profile tails on supported shapes
    if len(xy) >= 20:
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

    l_res = fit_missing_corner_l_shape(
        kept, return_info=True, allow_tilted_notch=has_deskew_tilt
    )
    if l_res is None and len(band) > len(kept):
        l_res = fit_missing_corner_l_shape(
            band, return_info=True, allow_tilted_notch=has_deskew_tilt
        )
    if l_res is not None:
        lxy_raw = np.asarray(l_res[0] if isinstance(l_res, tuple) else l_res, dtype=float).copy()
        if len(lxy_raw) == 6:
            xmin, ymin, xmax, ymax = _bounding_box_percentiles(xy_for_bbox)
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

