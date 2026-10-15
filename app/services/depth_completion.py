import os
import hashlib
import logging
import threading
import numpy as np
from scipy.stats import rankdata
from scipy.optimize import least_squares
from scipy.spatial import KDTree
from PIL import Image
from app.core.config import settings
from app.core.security import verify_weight_integrity
from app.core.model_utils import create_onnx_session_from_bytes


try:
    import cv2
except ImportError:
    cv2 = None

logger = logging.getLogger(__name__)

_ray_cache = {}
_ray_lock = threading.Lock()


def _resize_image(arr: np.ndarray, target_size: tuple[int, int]) -> np.ndarray:
    w_tar, h_tar = target_size
    if cv2 is not None:
        return cv2.resize(arr, (w_tar, h_tar), interpolation=cv2.INTER_LINEAR)
    img = Image.fromarray(arr)
    resized = img.resize((w_tar, h_tar), Image.Resampling.BILINEAR)
    return np.array(resized)


PLANE_INLIER_M = 0.05
PLANE_BEHIND_M = 0.15
PLANE_MIN_INLIERS = 12
PLANE_MIN_AABB_DIAG_M = 0.50
PLANE_MIN_STANDOFF_M = 0.45
PLANE_MIN_AXIS_SPAN_M = 0.30
PLANE_DENOM_EPS = 1e-6
PLANE_GRAZING_FACTOR = 1e-3
RANSAC_MIN_SAMPLE_DIST_M = 0.05
INLIER_HULL_DILATE_PX = 24
GHOST_RANGE_FLOOR_M = 2.50
CONTINUITY_SOBEL_ABS_M = 0.08
CONTINUITY_SOBEL_REL = 0.06
INPAINT_Z_MIN_M = 0.20
INPAINT_Z_MAX_M = 8.00


def analytical_z_plane(
    n: np.ndarray, D: float, K: np.ndarray, height: int, width: int
) -> np.ndarray:
    n = np.asarray(n, dtype=np.float64).reshape(3)
    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])
    u, v = np.meshgrid(np.arange(width, dtype=np.float64), np.arange(height, dtype=np.float64))
    denom = n[0] * (u - cx) / fx + n[1] * (v - cy) / fy + n[2]
    eps = max(PLANE_DENOM_EPS, PLANE_GRAZING_FACTOR * float(np.linalg.norm(n)))
    out = np.full((height, width), np.nan, dtype=np.float64)
    ok = np.abs(denom) >= eps
    out[ok] = -float(D) / denom[ok]
    out[ok & (out <= 0.0)] = np.nan
    return out


def _ransac_single_gravity_plane(
    points_xyz: np.ndarray, g_cam: np.ndarray, z_max_ghost: float | None = None
) -> dict | None:
    pts = np.asarray(points_xyz, dtype=np.float64)
    g = np.asarray(g_cam, dtype=np.float64).reshape(3)
    gn = float(np.linalg.norm(g))
    if pts.ndim != 2 or pts.shape[1] != 3 or len(pts) < PLANE_MIN_INLIERS or gn < 1e-12:
        return None
    g = g / gn
    n_pts = len(pts)
    best = None
    best_z = np.inf
    rng = np.random.default_rng(42)
    if n_pts <= 64:
        i_idx, j_idx = np.triu_indices(n_pts, k=1)
    else:
        all_count = n_pts * (n_pts - 1) // 2
        n_samples = min(2000, all_count)
        pairs = [tuple(rng.choice(n_pts, 2, replace=False)) for _ in range(n_samples)]
        pairs_arr = np.asarray(pairs, dtype=np.int64)
        i_idx = pairs_arr[:, 0]
        j_idx = pairs_arr[:, 1]

    if len(i_idx) == 0:
        return None
    pts_i = pts[i_idx]
    pts_j = pts[j_idx]
    v = pts_j - pts_i
    v_norm = np.linalg.norm(v, axis=1)
    mask_v = v_norm >= RANSAC_MIN_SAMPLE_DIST_M
    if not np.any(mask_v):
        return None

    v = v[mask_v]
    pts_i = pts_i[mask_v]

    nt = np.cross(g[None, :], v)
    nn = np.linalg.norm(nt, axis=1)
    mask_n = nn >= 1e-9
    if not np.any(mask_n):
        return None

    nt = nt[mask_n]
    nn = nn[mask_n]
    pts_i = pts_i[mask_n]

    N = nt / nn[:, None]
    D = -np.sum(N * pts_i, axis=1)

    R = np.abs(pts @ N.T + D)
    inlier_mask = R < PLANE_INLIER_M
    inlier_counts = np.sum(inlier_mask, axis=0)

    candidate_indices = np.flatnonzero(inlier_counts >= PLANE_MIN_INLIERS)
    for c_idx in candidate_indices:
        inlier_index = np.flatnonzero(inlier_mask[:, c_idx])
        xyz_i = pts[inlier_index]
        med_z = float(np.median(xyz_i[:, 2]))
        if med_z < PLANE_MIN_STANDOFF_M:
            continue
        if z_max_ghost is not None and med_z > float(z_max_ghost):
            continue
        dx = float(xyz_i[:, 0].max() - xyz_i[:, 0].min())
        dy = float(xyz_i[:, 1].max() - xyz_i[:, 1].min())
        dz = float(xyz_i[:, 2].max() - xyz_i[:, 2].min())
        if dx < PLANE_MIN_AXIS_SPAN_M and dy < PLANE_MIN_AXIS_SPAN_M:
            continue
        if (dx * dx + dy * dy + dz * dz) ** 0.5 < PLANE_MIN_AABB_DIAG_M:
            continue
        if med_z < best_z:
            best_z = med_z
            best = {"n": N[c_idx], "D": float(D[c_idx]), "inlier_index": inlier_index, "median_z": med_z}
    return best


def fit_multi_planes_gravity_constrained(
    points_xyz: np.ndarray,
    g_cam: np.ndarray,
    z_max_ghost: float | None = None,
    max_planes: int = 3,
) -> list[dict]:
    """Fit up to max_planes distinct gravity-constrained vertical planes sequentially."""
    pts = np.asarray(points_xyz, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 3 or len(pts) < PLANE_MIN_INLIERS:
        return []
    planes: list[dict] = []
    rem_mask = np.ones(len(pts), dtype=bool)
    for _ in range(max_planes):
        rem_idx = np.flatnonzero(rem_mask)
        if len(rem_idx) < PLANE_MIN_INLIERS:
            break
        cand = _ransac_single_gravity_plane(pts[rem_idx], g_cam, z_max_ghost=z_max_ghost)
        if cand is None:
            break
        n_c = np.asarray(cand["n"], dtype=np.float64)
        if any(abs(float(np.dot(n_c, np.asarray(p["n"], dtype=np.float64)))) >= 0.85 for p in planes):
            rem_mask[rem_idx[cand["inlier_index"]]] = False
            continue
        global_inliers = rem_idx[cand["inlier_index"]]
        planes.append({
            "n": n_c,
            "D": float(cand["D"]),
            "inlier_index": global_inliers,
            "median_z": float(cand["median_z"]),
        })
        rem_mask[global_inliers] = False
    return planes


def inlier_support_mask(
    height: int, width: int, uv: np.ndarray, dilate_px: int = INLIER_HULL_DILATE_PX
) -> np.ndarray:
    mask = np.zeros((height, width), dtype=np.uint8)
    if uv is None:
        return mask
    pts = np.asarray(uv, dtype=np.int32).reshape(-1, 2)
    if len(pts) == 0:
        return mask
    pts[:, 0] = np.clip(pts[:, 0], 0, width - 1)
    pts[:, 1] = np.clip(pts[:, 1], 0, height - 1)
    if cv2 is not None and len(pts) >= 3:
        hull = cv2.convexHull(pts)
        cv2.fillConvexPoly(mask, hull, 1)
    else:
        mask[pts[:, 1], pts[:, 0]] = 1
    if cv2 is None:
        return mask
    k = 2 * int(dilate_px) + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    return cv2.dilate(mask, kernel)


def pregate_sparse_tof(
    sparse_tof: np.ndarray,
    K: np.ndarray,
    plane: dict | None,
    inlier_uv: np.ndarray | None,
    z_max_ghost: float | None,
) -> tuple[np.ndarray, int]:
    out = np.array(sparse_tof, dtype=np.float32, copy=True)
    rows, cols = np.where((out > 0.05) & (out < 8.0))
    if len(rows) == 0:
        return out, 0
    if plane is None:
        if z_max_ghost is None:
            return out, 0
        ghost_mask = out[rows, cols] > float(z_max_ghost)
        dropped_rows = rows[ghost_mask]
        dropped_cols = cols[ghost_mask]
        out[dropped_rows, dropped_cols] = 0.0
        return out, int(np.count_nonzero(ghost_mask))

    h, w = out.shape
    dilate_px = max(INLIER_HULL_DILATE_PX, int(round(0.20 * max(h, w))))
    in_hull = np.zeros(len(rows), dtype=bool)
    if inlier_uv is not None and len(inlier_uv) > 0:
        pts = np.asarray(inlier_uv, dtype=np.float32).reshape(-1, 2)
        pts[:, 0] = np.clip(pts[:, 0], 0, w - 1)
        pts[:, 1] = np.clip(pts[:, 1], 0, h - 1)
        if cv2 is not None and len(pts) >= 3:
            hull_pts = cv2.convexHull(pts)
            hull_pts = np.asarray(hull_pts, dtype=np.float32)
            for idx, (c, r) in enumerate(zip(cols, rows)):
                if cv2.pointPolygonTest(hull_pts, (float(c), float(r)), True) >= -dilate_px:
                    in_hull[idx] = True
        elif len(pts) > 0:
            query_pts = np.column_stack((cols, rows))
            dists = np.min(np.linalg.norm(query_pts[:, None, :] - pts[None, :, :], axis=2), axis=1)
            in_hull = dists <= dilate_px

    if not np.any(in_hull):
        return out, 0

    valid_rows = rows[in_hull]
    valid_cols = cols[in_hull]

    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])
    n = np.asarray(plane["n"], dtype=float).reshape(3)
    D = float(plane["D"])
    A, B, C = n[0], n[1], n[2]

    denom = A * (valid_cols - cx) / fx + B * (valid_rows - cy) / fy + C
    eps = max(PLANE_DENOM_EPS, PLANE_GRAZING_FACTOR * float(np.linalg.norm(n)))
    valid_denom = np.abs(denom) >= eps

    zp = np.full(valid_rows.shape, np.nan, dtype=float)
    zp[valid_denom] = -D / denom[valid_denom]
    zp[valid_denom & (zp <= 0.0)] = np.nan

    is_finite = np.isfinite(zp)
    drop_mask = is_finite & (out[valid_rows, valid_cols] > zp + PLANE_BEHIND_M)
    out[valid_rows[drop_mask], valid_cols[drop_mask]] = 0.0
    return out, int(np.count_nonzero(drop_mask))


def flatten_mirror_cavity_depth(
    dense_metric: np.ndarray,
    sparse_status: np.ndarray | None,
    plane: np.ndarray | dict,
    K: np.ndarray,
    inlier_hull_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, int]:
    """Flatten mirror/glass/multipath phantom cavities inside a supported wall plane without touching foreground objects."""
    out = np.array(dense_metric, dtype=np.float32, copy=True)
    if isinstance(plane, dict):
        n = np.asarray(plane["n"], dtype=np.float64).reshape(3)
        D = float(plane["D"])
    else:
        p_arr = np.asarray(plane, dtype=np.float64).reshape(4)
        n, D = p_arr[:3], float(p_arr[3])
    h, w = out.shape
    zmap = analytical_z_plane(n, D, K, h, w)
    valid = np.isfinite(zmap) & (zmap > INPAINT_Z_MIN_M) & (zmap < INPAINT_Z_MAX_M)
    if inlier_hull_mask is not None:
        valid &= np.asarray(inlier_hull_mask) > 0
    if not np.any(valid):
        return out, 0

    anom_region = np.zeros((h, w), dtype=bool)
    if sparse_status is not None and sparse_status.shape == out.shape:
        st = np.asarray(sparse_status, dtype=np.int32)
        anom_pts = (st != 0) & (st != 5) & (st != 9)
        if np.any(anom_pts):
            if cv2 is not None:
                rad = max(28, int(round(0.08 * max(h, w))))
                k_anom = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * rad + 1, 2 * rad + 1))
                anom_region = cv2.dilate(anom_pts.astype(np.uint8), k_anom) > 0
            else:
                anom_region = anom_pts

    # Only flatten pixels that bulge BEHIND the wall plane (never flatten foreground obstacles where out < zmap)
    cavity_mask = valid & (
        (anom_region & (out > zmap + PLANE_INLIER_M))
        | (out > zmap + PLANE_BEHIND_M)
        | (anom_region & (out <= 0.0))
    )
    count = int(np.count_nonzero(cavity_mask))
    if count > 0:
        out[cavity_mask] = zmap[cavity_mask].astype(np.float32)
    return out, count


def inpaint_planar_depth(
    dense_metric: np.ndarray,
    pre_huber_depth: np.ndarray,
    plane: np.ndarray,
    K: np.ndarray,
    inlier_hull_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, int]:
    out = np.array(dense_metric, dtype=np.float32, copy=True)
    plane = np.asarray(plane, dtype=np.float64).reshape(4)
    n, D = plane[:3], float(plane[3])
    h, w = out.shape
    zmap = analytical_z_plane(n, D, K, h, w)
    valid = np.isfinite(zmap) & (zmap > INPAINT_Z_MIN_M) & (zmap < INPAINT_Z_MAX_M)
    if inlier_hull_mask is not None:
        valid &= np.asarray(inlier_hull_mask) > 0
    behind = valid & (out > zmap + PLANE_BEHIND_M)
    invalid = valid & (out <= 0)
    candidates = behind | invalid
    if cv2 is None or not np.any(candidates):
        return out, 0
    nlab, labels, stats, _ = cv2.connectedComponentsWithStats(candidates.astype(np.uint8), 4)
    sx = cv2.Sobel(pre_huber_depth.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3)
    sy = cv2.Sobel(pre_huber_depth.astype(np.float32), cv2.CV_32F, 0, 1, ksize=3)
    mag = np.hypot(sx, sy)
    painted = 0
    k1 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    for lab in range(1, nlab):
        if stats[lab, cv2.CC_STAT_AREA] < 16:
            continue
        cc = labels == lab
        dil = cv2.dilate(cc.astype(np.uint8), k1) > 0
        border = dil & ~cc
        if np.any(border):
            zp = float(np.nanmedian(zmap[cc]))
            thr = max(CONTINUITY_SOBEL_ABS_M, CONTINUITY_SOBEL_REL * zp)
            if float(np.median(mag[border])) > thr:
                continue
        out[cc] = zmap[cc].astype(np.float32)
        painted += int(np.count_nonzero(cc))
    return out, painted



class DepthorModelManager:
    _instance = None
    _lock = None

    @classmethod
    def _get_lock(cls):
        if cls._lock is None:
            cls._lock = threading.Lock()
        return cls._lock

    @classmethod
    def get_session(cls, model_path: str = None):
        if model_path is None:
            model_path = settings.MODEL_WEIGHTS_PATH
        if cls._instance is None:
            with cls._get_lock():
                if cls._instance is None:
                    cls._instance = cls._init_session(model_path)
        return cls._instance

    @classmethod
    def release(cls):
        with cls._get_lock():
            cls._instance = None
        import gc
        gc.collect()

    @classmethod
    def _init_session(cls, model_path: str):
        if not os.path.exists(model_path):
            env_mode = getattr(settings, "ENV_MODE", "local")
            if env_mode == "local":
                logger.warning("Weights not found at %s. Running in dummy mode.", model_path)
                return None
            raise FileNotFoundError(f"Missing weights file for inference: {model_path}")

        # Read model bytes into memory buffer and verify hash to prevent TOCTOU file swap race
        with open(model_path, "rb") as f:
            model_bytes = f.read()

        expected_sha = getattr(settings, "MODEL_WEIGHTS_SHA", None)
        if not verify_weight_integrity(model_bytes, expected_sha):
            raise ValueError(f"Weights integrity validation failed for weights at: {model_path}")

        return create_onnx_session_from_bytes(model_bytes)



class DepthCompletionService:
    def __init__(self):
        self.last_fusion_info = {}

    def construct_input_tensor(
        self,
        rgb_frame: np.ndarray,
        sparse_tof: np.ndarray,
        mask: np.ndarray,
        K: np.ndarray,
        target_size: tuple[int, int] = (518, 518),
    ) -> dict:
        h_tar, w_tar = target_size
        rgb_resized = _resize_image(rgb_frame, (w_tar, h_tar))
        rgb_tensor = np.transpose(rgb_resized.astype(np.float32) / 255.0, (2, 0, 1))

        h_orig, w_orig = sparse_tof.shape
        rows, cols = np.where(mask)
        depths = sparse_tof[mask]

        # Exact sub-pixel coordinate scaling formula
        rows_tar = np.clip(np.round((rows + 0.5) * (h_tar / h_orig) - 0.5).astype(int), 0, h_tar - 1)
        cols_tar = np.clip(np.round((cols + 0.5) * (w_tar / w_orig) - 0.5).astype(int), 0, w_tar - 1)

        tof_resized = np.zeros((h_tar, w_tar), dtype=np.float32)
        mask_resized = np.zeros((h_tar, w_tar), dtype=bool)
        tof_resized[rows_tar, cols_tar] = depths
        mask_resized[rows_tar, cols_tar] = True

        tof_tensor = tof_resized[np.newaxis, :, :]
        mask_tensor = mask_resized[np.newaxis, :, :].astype(np.float32)

        cx = K[0, 2] * (w_tar / w_orig)
        cy = K[1, 2] * (h_tar / h_orig)
        fx = K[0, 0] * (w_tar / w_orig)
        fy = K[1, 1] * (h_tar / h_orig)

        cache_key = (h_tar, w_tar, float(fx), float(fy), float(cx), float(cy))
        with _ray_lock:
            if cache_key in _ray_cache:
                ray_tensor = _ray_cache[cache_key]
            else:
                u, v = np.meshgrid(np.arange(w_tar), np.arange(h_tar))
                # Omit +0.5 pixel offset to stay aligned with OpenCV intrinsics convention
                d_norm = np.sqrt(((u - cx) / fx) ** 2 + ((v - cy) / fy) ** 2 + 1.0)
                rx = (u - cx) / (fx * d_norm)
                ry = (v - cy) / (fy * d_norm)
                rz = 1.0 / d_norm
                ray_tensor = np.stack([rx, ry, rz], axis=0).astype(np.float32)
                _ray_cache[cache_key] = ray_tensor

        return {
            "rgb": rgb_tensor,
            "tof_depth": tof_tensor,
            "active_mask": mask_tensor,
            "rays": ray_tensor,
        }

    def align_scale_shift(
        self,
        rel_depth: np.ndarray,
        sparse_tof: np.ndarray,
        mask: np.ndarray,
        sparse_status: np.ndarray | None = None,
    ) -> np.ndarray:
        if not np.any(mask):
            rel_median = np.median(rel_depth)
            scale = 1.5 / rel_median if rel_median > 0 else 1.0
            aligned = rel_depth * scale
            invalid = aligned > 8.0
            aligned = np.clip(aligned, 0.1, 8.0)
            aligned[invalid] = 0.0
            return aligned

        y = sparse_tof[mask]
        x = rel_depth[mask]
        w_pts = np.ones_like(y, dtype=np.float64)
        if sparse_status is not None and sparse_status.shape == sparse_tof.shape:
            st = sparse_status[mask]
            w_pts = np.where(st == 5, 1.0, np.where(st == 9, 0.35, 0.15))

        if len(y) < 2:
            scale = np.median(y) / np.median(x) if np.median(x) > 0 else 1.0
            aligned = rel_depth * scale
            invalid = aligned > 8.0
            aligned = np.clip(aligned, 0.1, 8.0)
            aligned[invalid] = 0.0
            return aligned

        if len(x) > 500:
            indices = np.unique(np.linspace(0, len(x) - 1, 500, dtype=int))
            x = x[indices]
            y = y[indices]
            w_pts = w_pts[indices]

        high_conf = w_pts >= 0.9
        if np.count_nonzero(high_conf) >= 4 and np.median(x[high_conf]) > 0:
            s0 = np.median(y[high_conf]) / np.median(x[high_conf])
        else:
            s0 = np.median(y) / np.median(x) if np.median(x) > 0 else 1.0

        def res(p):
            s, c = p
            return w_pts * ((s * x + c) - y)

        try:
            # Explicitly specify Huber loss f_scale=0.1 and scale bound s based on s0
            lb = max(0.3 * s0, 0.1)
            ub = min(3.0 * s0, 10.0)
            if lb < ub:
                s_init = float(np.clip(s0, lb, ub))
                res_opt = least_squares(
                    res,
                    x0=[s_init, 0.0],
                    loss="huber",
                    f_scale=0.1,
                    bounds=([max(0.3 * s0, 0.1), -np.inf], [min(3.0 * s0, 10.0), np.inf]),
                )
                s_opt, c_opt = res_opt.x
            else:
                s_opt = max(s0, 0.1)
                c_opt = 0.0
        except Exception as err:
            logger.warning("Scale optimization failed: %s. Falling back to median ratio.", err)
            s_opt = max(s0, 0.1)
            c_opt = 0.0

        aligned = s_opt * rel_depth + c_opt
        invalid = aligned > 8.0
        aligned = np.clip(aligned, 0.1, 8.0)
        aligned[invalid] = 0.0
        return aligned

    def remove_tof_anomalies(
        self, rel_depth: np.ndarray, sparse_tof: np.ndarray, mask: np.ndarray
    ) -> np.ndarray:
        filtered_mask = mask.copy()
        rows, cols = np.where(mask)
        if len(rows) > 500:
            return filtered_mask

        points = np.stack([rows, cols], axis=1)
        K = min(8, len(points) - 1)
        if K <= 2:
            return filtered_mask

        tree = KDTree(points)
        _, nearest_indices = tree.query(points, k=K + 1)

        neighbor_coords = points[nearest_indices]
        p_rel_all = rel_depth[neighbor_coords[:, :, 0], neighbor_coords[:, :, 1]]
        p_tof_all = sparse_tof[neighbor_coords[:, :, 0], neighbor_coords[:, :, 1]]

        std_rel = np.std(p_rel_all, axis=1)
        std_tof = np.std(p_tof_all, axis=1)
        valid_std = (std_rel >= 1e-6) & (std_tof >= 1e-6)

        # Use scipy.stats.rankdata to compute exact ranks with ties handled properly
        ranks_rel = rankdata(p_rel_all, axis=1)
        ranks_tof = rankdata(p_tof_all, axis=1)

        r_rel_mean = np.mean(ranks_rel, axis=1, keepdims=True)
        r_tof_mean = np.mean(ranks_tof, axis=1, keepdims=True)
        rel_diff = ranks_rel - r_rel_mean
        tof_diff = ranks_tof - r_tof_mean

        num = np.sum(rel_diff * tof_diff, axis=1)
        den = np.sqrt(np.sum(rel_diff**2, axis=1) * np.sum(tof_diff**2, axis=1))
        corr = np.divide(num, den, out=np.zeros_like(num), where=den > 1e-6)

        anomalies = valid_std & (corr < 0.3)
        filtered_mask[rows[anomalies], cols[anomalies]] = False
        return filtered_mask

    def run_depthor_plus(
        self,
        rgb: np.ndarray,
        sparse_tof: np.ndarray,
        intrinsics: np.ndarray,
        *,
        g_cam: np.ndarray | None = None,
        z_max_ghost: float | None = None,
        sparse_status: np.ndarray | None = None,
    ) -> np.ndarray:
        h, w, _ = rgb.shape
        work = np.array(sparse_tof, dtype=np.float32, copy=True)
        status_dropped = 0
        if sparse_status is not None and sparse_status.shape == work.shape:
            st = np.asarray(sparse_status, dtype=np.int32)
            valid_tof = (work > 0.05) & (work < 8.0)
            bad_st = valid_tof & (st != 0) & (st != 5) & (st != 9)
            if np.any(bad_st):
                status_dropped += int(np.count_nonzero(bad_st))
                work[bad_st] = 0.0
            s5_mask = (work > 0.05) & (work < 8.0) & (st == 5)
            s9_mask = (work > 0.05) & (work < 8.0) & (st == 9)
            if np.count_nonzero(s5_mask) >= 8 and np.any(s9_mask):
                p90_s5 = float(np.percentile(work[s5_mask], 90))
                s9_outlier = s9_mask & (work > max(1.6 * p90_s5, p90_s5 + 0.80))
                if np.any(s9_outlier):
                    status_dropped += int(np.count_nonzero(s9_outlier))
                    work[s9_outlier] = 0.0

        fusion_off = g_cam is None and z_max_ghost is None
        dropped = status_dropped
        trusted = None
        secondary = None
        secondary_uv = None
        inpainted = 0
        reason = "fusion_off"
        input_empty = not np.any((work > 0.05) & (work < 8.0))
        if not fusion_off:
            try:
                rows, cols = np.where((work > 0.05) & (work < 8.0))
                pts = None
                if g_cam is not None and len(rows) >= PLANE_MIN_INLIERS:
                    fx, fy = float(intrinsics[0, 0]), float(intrinsics[1, 1])
                    cx, cy = float(intrinsics[0, 2]), float(intrinsics[1, 2])
                    z = work[rows, cols].astype(np.float64)
                    x = (cols - cx) / fx * z
                    y = (rows - cy) / fy * z
                    pts = np.column_stack([x, y, z])
                    # Prefer status==5 points for primary plane fitting when abundant
                    if sparse_status is not None and sparse_status.shape == work.shape:
                        st_rc = sparse_status[rows, cols]
                        s5_idx = np.flatnonzero(st_rc == 5)
                        if len(s5_idx) >= PLANE_MIN_INLIERS:
                            s5_planes = fit_multi_planes_gravity_constrained(
                                pts[s5_idx], g_cam, z_max_ghost=z_max_ghost, max_planes=1
                            )
                            if s5_planes:
                                fit_s5 = s5_planes[0]
                                n_fit = np.asarray(fit_s5["n"], dtype=np.float64)
                                d_fit = float(fit_s5["D"])
                                inl_all = np.flatnonzero(np.abs(pts @ n_fit + d_fit) < PLANE_INLIER_M)
                                trusted = {
                                    "n": n_fit,
                                    "D": d_fit,
                                    "inlier_index": inl_all,
                                    "median_z": float(np.median(pts[inl_all, 2])) if len(inl_all) > 0 else fit_s5["median_z"],
                                }
                    if trusted is None:
                        multi_planes = fit_multi_planes_gravity_constrained(
                            pts, g_cam, z_max_ghost=z_max_ghost, max_planes=2
                        )
                        if multi_planes:
                            trusted = multi_planes[0]
                            if len(multi_planes) > 1:
                                secondary = multi_planes[1]
                                sec_global_idx = secondary["inlier_index"]
                                secondary_uv = np.column_stack([cols[sec_global_idx], rows[sec_global_idx]])
                    # Check for a secondary vertical wall plane (e.g. orthogonal wall in a corner)
                    if trusted is not None and secondary is None:
                        rem_mask = np.ones(len(pts), dtype=bool)
                        rem_mask[trusted["inlier_index"]] = False
                        rem_idx = np.flatnonzero(rem_mask)
                        if len(rem_idx) >= PLANE_MIN_INLIERS:
                            sec_planes = fit_multi_planes_gravity_constrained(
                                pts[rem_idx], g_cam, z_max_ghost=z_max_ghost, max_planes=1
                            )
                            if sec_planes:
                                sec_fit = sec_planes[0]
                                cos_ang = abs(float(np.dot(trusted["n"], sec_fit["n"])))
                                if cos_ang < 0.85:  # distinct wall orientation (> ~31 deg)
                                    sec_global_idx = rem_idx[sec_fit["inlier_index"]]
                                    secondary = {
                                        "n": sec_fit["n"],
                                        "D": sec_fit["D"],
                                        "inlier_index": sec_global_idx,
                                        "median_z": sec_fit["median_z"],
                                    }
                                    secondary_uv = np.column_stack([cols[sec_global_idx], rows[sec_global_idx]])
                uv = None
                if trusted is not None:
                    uv = np.column_stack(
                        [cols[trusted["inlier_index"]], rows[trusted["inlier_index"]]]
                    )
                    reason = "trusted_inpaint"
                elif g_cam is None:
                    reason = "no_gravity"
                else:
                    reason = "no_plane_ghost_cap"
                work, pg_dropped = pregate_sparse_tof(work, intrinsics, trusted, uv, z_max_ghost)
                dropped += int(pg_dropped)
            except Exception:
                logger.warning("planar pre-gate failed; using original sparse ToF", exc_info=True)
                work = np.array(sparse_tof, dtype=np.float32, copy=True)
                dropped = 0
                trusted = None
                secondary = None
                reason = "pregate_failed"
            remain = np.any((work > 0.05) & (work < 8.0))
            if (not remain) and (dropped > 0 or not input_empty):
                self.last_fusion_info = {
                    "trusted_plane": False,
                    "reason": "empty_mask",
                    "n": None,
                    "D": None,
                    "inliers": 0,
                    "median_z": None,
                    "pre_gate_dropped": int(dropped),
                    "inpainted_pixels": 0,
                    "z_max_ghost": z_max_ghost,
                }
                return np.zeros((h, w), dtype=np.float32)

        mask = (work > 0.05) & (work < 8.0)
        tensors = self.construct_input_tensor(rgb, work, mask, intrinsics, target_size=(518, 518))
        model_path = settings.MODEL_WEIGHTS_PATH
        session = DepthorModelManager.get_session(model_path)
        if session is not None:
            input_feed = {
                "rgb": np.expand_dims(tensors["rgb"], axis=0),
                "tof_depth": np.expand_dims(tensors["tof_depth"], axis=0),
                "active_mask": np.expand_dims(tensors["active_mask"], axis=0),
                "rays": np.expand_dims(tensors["rays"], axis=0),
            }
            outputs = session.run(None, input_feed)
            out_depth = np.squeeze(outputs[0])
            if out_depth.ndim == 3 and out_depth.shape[0] == 128:
                bins = np.exp(np.linspace(np.log(0.1), np.log(8.0), 128)).reshape(128, 1, 1)
                exp_logits = np.exp(out_depth - np.max(out_depth, axis=0, keepdims=True))
                probs = exp_logits / np.sum(exp_logits, axis=0, keepdims=True)
                rel_depth_518 = np.sum(probs * bins, axis=0)
            else:
                rel_depth_518 = out_depth
            rel_depth = _resize_image(rel_depth_518, (w, h))
        else:
            rel_depth = np.random.uniform(0.5, 2.0, (h, w))

        clean_mask = self.remove_tof_anomalies(rel_depth, work, mask)
        dense_metric = self.align_scale_shift(rel_depth, work, clean_mask, sparse_status=sparse_status)
        if trusted is not None and session is not None:
            try:
                plane = np.array([*trusted["n"], trusted["D"]], dtype=np.float64)
                hull_mask = None
                if secondary is not None and secondary_uv is not None and uv is not None:
                    dil_px = max(INLIER_HULL_DILATE_PX, int(round(0.18 * max(h, w))))
                    prim_hull = inlier_support_mask(h, w, uv, dilate_px=dil_px)
                    sec_hull = inlier_support_mask(h, w, secondary_uv, dilate_px=INLIER_HULL_DILATE_PX)
                    hull_mask = (prim_hull > 0) & (sec_hull == 0)
                dense_metric, inpainted = inpaint_planar_depth(
                    dense_metric, rel_depth, plane, intrinsics, inlier_hull_mask=hull_mask
                )
                if sparse_status is not None and uv is not None:
                    prim_support = hull_mask if hull_mask is not None else inlier_support_mask(
                        h, w, uv, dilate_px=INLIER_HULL_DILATE_PX
                    )
                    dense_metric, mirror_flat = flatten_mirror_cavity_depth(
                        dense_metric, sparse_status, plane, intrinsics, inlier_hull_mask=prim_support
                    )
                    inpainted += int(mirror_flat)
                if secondary is not None and secondary_uv is not None:
                    sec_plane = np.array([*secondary["n"], secondary["D"]], dtype=np.float64)
                    sec_hull_full = inlier_support_mask(
                        h, w, secondary_uv, dilate_px=max(INLIER_HULL_DILATE_PX, int(round(0.15 * max(h, w))))
                    )
                    if uv is not None:
                        prim_core = inlier_support_mask(h, w, uv, dilate_px=INLIER_HULL_DILATE_PX)
                        sec_hull_full = (sec_hull_full > 0) & (prim_core == 0)
                    dense_metric, inpainted_sec = inpaint_planar_depth(
                        dense_metric, rel_depth, sec_plane, intrinsics, inlier_hull_mask=sec_hull_full
                    )
                    inpainted += int(inpainted_sec)
                    if sparse_status is not None:
                        dense_metric, mirror_sec = flatten_mirror_cavity_depth(
                            dense_metric, sparse_status, sec_plane, intrinsics, inlier_hull_mask=sec_hull_full
                        )
                        inpainted += int(mirror_sec)
            except Exception:
                logger.warning("planar inpaint failed; keeping aligned depth", exc_info=True)
                inpainted = 0
        self.last_fusion_info = {
            "trusted_plane": trusted is not None,
            "reason": reason,
            "n": None if trusted is None else [float(x) for x in trusted["n"]],
            "D": None if trusted is None else float(trusted["D"]),
            "inliers": 0 if trusted is None else int(len(trusted["inlier_index"])),
            "median_z": None if trusted is None else float(trusted["median_z"]),
            "pre_gate_dropped": int(dropped),
            "inpainted_pixels": int(inpainted),
            "z_max_ghost": z_max_ghost,
        }
        return dense_metric



