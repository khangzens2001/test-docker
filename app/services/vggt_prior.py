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

    def extract_floor_ceiling_height(self, z_pts: np.ndarray, default_h: float = 2.23) -> float:
        """Estimates vertical clearance from density peaks rather than naive percentiles."""
        if len(z_pts) < 100 or float(np.ptp(z_pts) if len(z_pts) else 0) < 0.3:
            return default_h
        from app.services.plane_segmentation import estimate_ceiling_height
        h = float(estimate_ceiling_height(z_pts, default_h=default_h))
        return h if 1.5 <= h <= 4.0 else default_h

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

        is_partial_scan = (h_vggt < 1.50 and float(np.ptp(pts_l[:, 2])) < 1.60)
        if not is_partial_scan:
            if initial_s is not None and SIM3_SCALE_MIN <= initial_s <= SIM3_SCALE_MAX and not cam_degenerate:
                s = float(initial_s)
            else:
                h_lidar = float(height_m) if height_m and height_m > 0 else self.extract_floor_ceiling_height(pts_l[:, 2])
                s = float(np.clip(h_lidar / h_vggt, SIM3_SCALE_MIN, SIM3_SCALE_MAX))
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
            s_candidates = [1.00, 1.04, 1.07, 1.09, 1.10, 1.12, 1.15, 1.20]
            yaw_range = [
                fine
                for base in [0.0, 90.0, 180.0, 270.0]
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
        proj = xy @ n if len(xy) else np.zeros(0)
        search_hint = pos_hint
        lo = search_hint - T2_WINDOW_M
        hi = search_hint + T2_WINDOW_M
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
        if peak >= 5000 and cnt_hint < 0.10 * peak:
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

        dir_key = tuple(np.round(n, 2))
        is_inner_l_wall = (len(hints) == 6) and (pos_hint < hint_map.get(dir_key, pos_hint) - 0.05)

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

        # Structural wall behind fixture check:
        # If candidate sits on an interior fixture while LiDAR points extend further outward
        # to a supported structural wall peak:
        if opp_effective is not None and opp_wall is not None:
            structural_candidates = [
                j
                for j in range(len(centres))
                if (centres[j] - c_hint >= 0.04)
                and (centres[j] - min(c_hint, pos_hint) <= T2_WINDOW_M + 0.10)
                and (cand_counts[j] >= 500)
                and (
                    (cand_counts[j] >= 2.5 * cnt_hint)
                    or (
                        cand_counts[j] >= 1000
                        and pos_hint > c_hint
                        and (centres[j] + opp_effective) <= vggt_span + 0.10
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
            and (c_hint + float(opp_wall.pos_metric)) >= 2.15
        )
        if (
            opp_pos is not None
            and (pos_hint - c_hint) >= 0.05
            and max_p < pos_hint - 0.015
            and not lidar_dense_at_hint
            and not strong_near_peak
        ):
            walls.append(T2Wall(n=n, pos_hint=pos_hint, pos_metric=pos_hint, source="vggt_hint"))
            continue

        if opp_effective is not None:
            current_span = float(c_hint + opp_effective)
            span_deficit = vggt_span - current_span
            max_cnt = float(np.max(cand_counts)) if len(cand_counts) else 1.0

            if not shoulder_walked and (pos_hint - c_hint >= 0.015) and span_deficit >= 0.02:
                outer_candidates = [
                    j
                    for j in range(len(centres))
                    if (centres[j] - c_hint >= 0.02)
                    and (centres[j] - c_hint <= T2_WINDOW_M)
                    and ((centres[j] + opp_effective) <= vggt_span + 0.05)
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
                elif cnt_hint < 100 or (vggt_span is not None and vggt_span < 1.80 and max_p < pos_hint):
                    pos_metric = max(pos_hint, vggt_span - opp_effective)
                    walls.append(T2Wall(n=n, pos_hint=pos_hint, pos_metric=pos_metric, source="vggt_hint"))
                    continue
            elif (c_hint - pos_hint >= 0.03) and span_deficit <= -0.04:
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
            len(proj) >= 50
            and abs(pos_hint - float(np.max(proj))) <= 1e-4
            and 0.015 < (pos_hint - pos_metric) <= 0.035
        ):
            pos_metric = float(pos_hint - 0.012)
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
        if os.path.isfile(vio_path) and os.path.getsize(vio_path) > 0:
            try:
                vio_df = pd.read_csv(vio_path)
                if {"frame", "x", "y", "z"}.issubset(vio_df.columns):
                    for _, r in vio_df.iterrows():
                        vio_poses[int(r["frame"])] = [float(r["x"]), float(r["y"]), float(r["z"])]
            except Exception:
                vio_poses = {}

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
            if "qx" in f0 and "qw" in f0:
                try:
                    from scipy.spatial.transform import Rotation
                    qx = float(f0.get("qx", 0.0))
                    qy = float(f0.get("qy", 0.0))
                    qz = float(f0.get("qz", 0.0))
                    qw = float(f0.get("qw", 1.0))
                    R0 = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()
                except Exception:
                    R0 = None
        colors = np.asarray(npz["colors"]) if "colors" in npz else None
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


def extract_vggt_wall_axes(
    points_occ: np.ndarray,
    height_m: float,
    trajectory: np.ndarray | None = None,
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
    xy = kept[:, :2]

    fill = filled_bbox_fill_ratio(xy)
    occupied, x_min, y_min, res, w, h = _raster_u8(xy)
    ratio_door, _, _ = _filled_bbox_ratio(occupied, DOORWAY_BRIDGE_M, res)

    if fill < MIN_L_SHAPE_FILL_RATIO and ratio_door < MIN_L_SHAPE_FILL_RATIO:
        if trajectory is None:
            return WallAxesResult(hints=None, skip_reason="unsupported_shape")

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
            allow_long_clip = not (
                pre_span[long_ax] >= 1.80
                and post[long_ax] < 1.80
                and not (
                    post[long_ax] >= 1.25
                    and post[short_ax] >= 1.80
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

    if fill >= BBOX_FILL_RATIO_THRESHOLD or ratio_door >= BBOX_FILL_RATIO_THRESHOLD:
        xmin, ymin, xmax, ymax = _bounding_box_percentiles(xy)
        ring = np.array(
            [[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]], dtype=float
        )
        return _validate_hints(_hints_from_ring(ring))
    l_res = fit_missing_corner_l_shape(xy, return_info=True)
    if l_res is not None:
        lxy_raw = l_res[0] if isinstance(l_res, tuple) else l_res
        lxy = close_orthogonal_polygon(np.asarray(lxy_raw, dtype=float))
        if lxy is not None and len(lxy) == 6:
            res = _validate_hints(_hints_from_ring(lxy))
            if res.hints is not None:
                return res

    span = xy.max(axis=0) - xy.min(axis=0)
    if span[0] >= 0.5 and span[1] >= 0.5 and (fill >= 0.25 or ratio_door >= 0.25):
        xmin, ymin, xmax, ymax = _bounding_box_percentiles(xy)
        ring = np.array(
            [[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]], dtype=float
        )
        res = _validate_hints(_hints_from_ring(ring))
        if res.hints is not None:
            return res

    return WallAxesResult(hints=None, skip_reason="unsupported_shape")

