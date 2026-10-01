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
T2_WINDOW_M = 0.20
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

    # Umeyama: cov = (dst_c.T @ src_c) / n ; dst ≈ s R src + t
    cov = (dst_c.T @ src_c) / float(n)
    U, sing, Vt = np.linalg.svd(cov)
    D = np.eye(3)
    if float(np.linalg.det(U) * np.linalg.det(Vt)) < 0.0:
        D[2, 2] = -1.0
    R = U @ D @ Vt
    # sum_i ||src_i - mean||^2 / n   — not np.var(src) (that flattens xyz)
    var_src = float(np.sum(src_c ** 2) / float(n))
    if var_src < 1e-18:
        return _empty_sim3(n, "degenerate_sim3")
    s = float(np.trace(np.diag(sing) @ D) / var_src)
    t = mu_dst - s * (R @ mu_src)
    aligned = apply_sim3(src, s, R, t)
    rmse = float(np.sqrt(np.mean(np.sum((dst - aligned) ** 2, axis=1))))
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
    ) -> tuple[bool, float, np.ndarray, np.ndarray, float]:
        """Aligns VGGT to metric LiDAR using vertical height scale and 3-DoF planar search."""
        import open3d as o3d

        pts_v = np.asarray(vggt_points, dtype=float)
        pts_l = np.asarray(lidar_points, dtype=float)
        if len(pts_v) < 50 or len(pts_l) < 50:
            return False, 1.0, np.eye(3), np.zeros(3), float("inf")

        R_init = np.asarray(initial_R, dtype=float) if initial_R is not None else np.eye(3)
        if abs(R_init[2, 2]) < 0.8:
            R_init = np.eye(3)

        if initial_s is not None and SIM3_SCALE_MIN <= initial_s <= SIM3_SCALE_MAX:
            s = float(initial_s)
        else:
            h_vggt = float(np.percentile(pts_v[:, 2], 99.8) - np.percentile(pts_v[:, 2], 0.2))
            if h_vggt < 0.05:
                h_vggt = float(np.ptp(pts_v[:, 2]))
            if h_vggt < 0.05:
                return False, 1.0, np.eye(3), np.zeros(3), float("inf")

            h_lidar = float(height_m) if height_m and height_m > 0 else self.extract_floor_ceiling_height(pts_l[:, 2])
            s = float(np.clip(h_lidar / h_vggt, SIM3_SCALE_MIN, SIM3_SCALE_MAX))
        scaled_v = (pts_v @ R_init.T) * s

        pcd_lidar_ds = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts_l)).voxel_down_sample(0.05)
        pcd_vggt_ds = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(scaled_v)).voxel_down_sample(0.05)
        if len(pcd_lidar_ds.points) < 20 or len(pcd_vggt_ds.points) < 20:
            return False, s, R_init, np.zeros(3), float("inf")

        kdtree = o3d.geometry.KDTreeFlann(pcd_lidar_ds)
        sub_pts_v = np.asarray(pcd_vggt_ds.points)
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

        lidar_median = np.median(np.asarray(pcd_lidar_ds.points), axis=0)
        for yaw_deg in np.linspace(0, 360, 72, endpoint=False):
            rad = np.radians(yaw_deg)
            c, sn = np.cos(rad), np.sin(rad)
            R_yaw = np.array([[c, -sn, 0.0], [sn, c, 0.0], [0.0, 0.0, 1.0]], dtype=float)
            rotated_eval = eval_pts @ R_yaw.T
            t = lidar_median - np.median(rotated_eval, axis=0)
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

        ok = (best_inliers >= int(n_eval * 0.12)) and (best_rmse <= self.max_inlier_dist)
        R_total = best_R_yaw @ R_init
        best_t = lidar_median - np.median((scaled_v @ best_R_yaw.T), axis=0)
        return ok, s, R_total, best_t, best_rmse


def apply_vggt_topology_prior(lidar_xy: np.ndarray, hints: list[WallHint]) -> T2Result:
    xy = np.asarray(lidar_xy, dtype=float).reshape(-1, 2)
    centroid = xy.mean(axis=0) if len(xy) else np.zeros(2)
    walls: list[T2Wall] = []
    for h in hints:
        n = np.asarray(h.n, dtype=float).reshape(-1)[:2]
        nrm = float(np.linalg.norm(n))
        if nrm < 1e-12:
            continue
        n = n / nrm
        pos_hint = float(h.pos_hint)
        proj = xy @ n if len(xy) else np.zeros(0)
        max_proj = float(np.max(proj)) if len(proj) else pos_hint
        search_hint = min(pos_hint, max_proj) if max_proj > 0 else pos_hint
        lo = search_hint - T2_WINDOW_M
        hi = search_hint + T2_WINDOW_M
        in_win = proj[(proj >= lo) & (proj <= hi)]
        edges = np.arange(lo, hi + T2_BIN_M * 0.5, T2_BIN_M)
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
            )
        ]
        if not cand:
            cand = np.flatnonzero(counts == peak).tolist()
        centres = 0.5 * (edges[cand] + edges[np.array(cand) + 1])
        cand_counts = counts[cand]

        dist_cost = np.abs(centres - search_hint)
        c_proj = float(centroid @ n)
        order = np.lexsort((centres, np.abs(centres - c_proj), -cand_counts, np.round(dist_cost, 2)))
        pos_metric = float(centres[order[0]])
        walls.append(T2Wall(n=n, pos_hint=pos_hint, pos_metric=pos_metric, source="lidar_peak"))
    mask = np.ones(len(xy), dtype=bool)
    for w in walls:
        mask &= (xy @ w.n) <= (w.pos_metric + T2_CLIP_MARGIN_M)
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
        prior = VggtPrior(
            vggt_filenames=names,
            t_vio=np.asarray(t_vio, dtype=float),
            C_vggt=np.asarray(C_vggt, dtype=float),
            points=points,
            confidence=conf,
            frame="vggt_prediction",
        )
        return VggtPriorResult(prior=prior, skip_reason=None)
    except Exception:
        logger.exception("load_vggt_prior failed")
        return VggtPriorResult(prior=None, skip_reason="exception")


def _signed_area(xy: np.ndarray) -> float:
    x = xy[:, 0]
    y = xy[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


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


def extract_vggt_wall_axes(points_occ: np.ndarray, height_m: float) -> WallAxesResult:
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
    kept = keep_largest_occupancy_component(band)
    if len(kept) == 0:
        kept = band
    xy = kept[:, :2]

    fill = filled_bbox_fill_ratio(xy)
    occupied, x_min, y_min, res, w, h = _raster_u8(xy)
    ratio_door, _, _ = _filled_bbox_ratio(occupied, DOORWAY_BRIDGE_M, res)

    if fill >= BBOX_FILL_RATIO_THRESHOLD or ratio_door >= BBOX_FILL_RATIO_THRESHOLD:
        xmin, ymin = xy.min(axis=0)
        xmax, ymax = xy.max(axis=0)
        ring = np.array(
            [[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]], dtype=float
        )
        return _validate_hints(_hints_from_ring(ring))

    if fill < MIN_L_SHAPE_FILL_RATIO and ratio_door < MIN_L_SHAPE_FILL_RATIO:
        return WallAxesResult(hints=None, skip_reason="unsupported_shape")

    l_res = fit_missing_corner_l_shape(xy, return_info=True)
    if l_res is None:
        span = xy.max(axis=0) - xy.min(axis=0)
        if span[0] >= 0.5 and span[1] >= 0.5 and (fill >= 0.35 or ratio_door >= 0.35):
            xmin, ymin = xy.min(axis=0)
            xmax, ymax = xy.max(axis=0)
            ring = np.array(
                [[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]], dtype=float
            )
            return _validate_hints(_hints_from_ring(ring))
        return WallAxesResult(hints=None, skip_reason="unsupported_shape")
    lxy_raw = l_res[0] if isinstance(l_res, tuple) else l_res
    lxy = close_orthogonal_polygon(np.asarray(lxy_raw, dtype=float))
    if lxy is None or len(lxy) != 6:
        return WallAxesResult(hints=None, skip_reason="unsupported_shape")
    return _validate_hints(_hints_from_ring(lxy))

