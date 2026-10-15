from __future__ import annotations

import logging
import os

import numpy as np
import open3d as o3d

from app.services.occupancy_layout import MAX_NGON_VERTICES

logger = logging.getLogger(__name__)

VISUAL_POINT_CLOUD_MAX = 1_000_000
VISUAL_STAT_NB_NEIGHBORS = 16
VISUAL_STAT_STD_RATIO = 2.8
ROOM_CROP_MARGIN_M = 0.15
CEILING_CUTAWAY_M = 0.20
VISUAL_CROP_MIN_REMAINING = 50
VISUAL_SOR_MIN_REMAINING = 50


def sample_visual_cloud(
    mesh: o3d.geometry.TriangleMesh, number_of_points: int | None = None
) -> o3d.geometry.PointCloud:
    empty = o3d.geometry.PointCloud()
    if len(mesh.vertices) == 0 or len(mesh.triangles) == 0:
        return empty
    if number_of_points is None:
        n = VISUAL_POINT_CLOUD_MAX
    else:
        try:
            n = int(number_of_points)
        except (TypeError, ValueError):
            return empty
    n = min(n, VISUAL_POINT_CLOUD_MAX)
    if n < 1:
        return empty
    try:
        o3d.utility.random.seed(42)
        np.random.seed(42)
        pcd = mesh.sample_points_uniformly(number_of_points=n)
    except Exception:
        logger.exception("sample_visual_cloud failed")
        return empty
    if len(pcd.points) == 0 or not pcd.has_colors():
        return empty
    return pcd


from shapely import contains_xy
from shapely.geometry import Polygon

from app.services.plane_segmentation import (
    DEFAULT_HEIGHT_M,
    SUBSURFACE_MARGIN_M,
    align_rotation,
    is_usable_gravity,
)


def apply_layout_frames(
    pcd: o3d.geometry.PointCloud,
    layout: dict,
    gravity: np.ndarray | None,
) -> o3d.geometry.PointCloud:
    work = o3d.geometry.PointCloud(pcd)
    if len(work.points) == 0:
        return work
    if gravity is not None and is_usable_gravity(gravity):
        g = np.asarray(gravity, dtype=float).reshape(3)
        nrm = float(np.linalg.norm(g))
        if nrm > 0.0:
            work.rotate(align_rotation(g / nrm), center=(0.0, 0.0, 0.0))
    layout = layout if isinstance(layout, dict) else {}
    lf = layout.get("level_frame") if isinstance(layout.get("level_frame"), dict) else {}
    try:
        r_level = np.asarray(lf.get("rotation_3x3", np.eye(3)), dtype=float).reshape(3, 3)
        t_level = np.asarray(lf.get("translation", [0.0, 0.0, 0.0]), dtype=float).reshape(3)
    except (TypeError, ValueError):
        r_level = np.eye(3)
        t_level = np.zeros(3)
    pts = np.asarray(work.points, dtype=float)
    pts = pts @ r_level.T + t_level
    ff = layout.get("floorplan_frame") if isinstance(layout.get("floorplan_frame"), dict) else None
    if ff is not None:
        try:
            r2 = np.asarray(ff.get("rotation_2x2", np.eye(2)), dtype=float).reshape(2, 2)
            t2 = np.asarray(ff.get("translation_xy", [0.0, 0.0]), dtype=float).reshape(2)
            xy = (r2 @ pts[:, :2].T).T + t2
            pts = np.column_stack([xy, pts[:, 2]])
        except (TypeError, ValueError):
            pass
    colors = None
    if work.has_colors() and len(work.colors) == len(pts):
        colors = np.asarray(work.colors)
    out = o3d.geometry.PointCloud()
    out.points = o3d.utility.Vector3dVector(pts)
    if colors is not None:
        out.colors = o3d.utility.Vector3dVector(colors)
    return out


def layout_xy_polygon(layout: dict) -> np.ndarray:
    empty = np.zeros((0, 2), dtype=float)
    if not isinstance(layout, dict):
        return empty
    verts = layout.get("vertices")
    if not isinstance(verts, dict):
        return empty
    n = len(verts)
    if n < 4 or n > MAX_NGON_VERTICES:
        return empty
    rows = []
    for i in range(n):
        raw = verts.get(f"v{i}")
        if raw is None or len(raw) < 2:
            return empty
        try:
            rows.append([float(raw[0]), float(raw[1])])
        except (TypeError, ValueError):
            return empty
    return np.asarray(rows, dtype=float)


def layout_height_m(layout: dict) -> float:
    layout = layout if isinstance(layout, dict) else {}
    walls = layout.get("walls") if isinstance(layout.get("walls"), list) else []
    for w in walls:
        if not isinstance(w, dict):
            continue
        try:
            h = float(w.get("height_meters", 0.0))
        except (TypeError, ValueError):
            continue
        if np.isfinite(h) and h >= 0.10:
            return h
    room = layout.get("room") if isinstance(layout.get("room"), dict) else {}
    try:
        h = float(room.get("height_meters", 0.0))
    except (TypeError, ValueError):
        h = 0.0
    if np.isfinite(h) and h >= 0.10:
        return h
    return float(DEFAULT_HEIGHT_M)


def crop_to_layout_polygon(
    pcd: o3d.geometry.PointCloud,
    layout: dict,
    margin_m: float | None = None,
) -> o3d.geometry.PointCloud:
    work = o3d.geometry.PointCloud(pcd)
    if len(work.points) == 0:
        return work
    margin = ROOM_CROP_MARGIN_M if margin_m is None else float(margin_m)
    xy = layout_xy_polygon(layout)
    if len(xy) < 4:
        return work
    try:
        poly = Polygon(xy)
        if not poly.is_valid:
            poly = poly.buffer(0)
        grown = poly.buffer(margin)
        if grown.is_empty:
            return work
    except Exception:
        return work
    pts = np.asarray(work.points, dtype=float)
    if len(pts) >= 50:
        p_span_x = float(np.percentile(pts[:, 0], 98) - np.percentile(pts[:, 0], 2))
        p_span_y = float(np.percentile(pts[:, 1], 98) - np.percentile(pts[:, 1], 2))
        p_area = max(0.1, p_span_x * p_span_y)
        if poly.area < 0.60 * p_area:
            # Layout polygon covers only a partial sub-region of the full room; do not truncate the room!
            return work
    mask_xy = np.asarray(contains_xy(grown, pts[:, 0], pts[:, 1]), dtype=bool)
    height_m = layout_height_m(layout)
    z_min = -float(SUBSURFACE_MARGIN_M)
    z_max = float(height_m) + float(margin)
    mask = mask_xy & (pts[:, 2] >= z_min) & (pts[:, 2] <= z_max)
    n_keep = int(np.count_nonzero(mask))
    if len(pts) >= VISUAL_CROP_MIN_REMAINING and n_keep < VISUAL_CROP_MIN_REMAINING:
        return work
    kept = pts[mask]
    out = o3d.geometry.PointCloud()
    out.points = o3d.utility.Vector3dVector(kept)
    if work.has_colors() and len(work.colors) == len(pts):
        out.colors = o3d.utility.Vector3dVector(np.asarray(work.colors)[mask])
    return out


def remove_visual_outliers(pcd: o3d.geometry.PointCloud) -> o3d.geometry.PointCloud:
    work = o3d.geometry.PointCloud(pcd)
    if len(work.points) < VISUAL_SOR_MIN_REMAINING:
        return work
    try:
        cleaned, _ = work.remove_statistical_outlier(
            nb_neighbors=VISUAL_STAT_NB_NEIGHBORS,
            std_ratio=VISUAL_STAT_STD_RATIO,
        )
    except Exception:
        return work
    if len(cleaned.points) < VISUAL_SOR_MIN_REMAINING:
        return work
    return cleaned


def cut_ceiling(
    pcd: o3d.geometry.PointCloud,
    height_m: float,
    cutaway_m: float | None = None,
) -> o3d.geometry.PointCloud:
    empty = o3d.geometry.PointCloud()
    cutaway = CEILING_CUTAWAY_M if cutaway_m is None else float(cutaway_m)
    z_max = float(height_m) - float(cutaway)
    if not np.isfinite(z_max) or z_max < 0.10:
        return empty
    if len(pcd.points) == 0:
        return empty
    pts = np.asarray(pcd.points, dtype=float)
    mask = pts[:, 2] <= z_max
    if not np.any(mask):
        return empty
    out = o3d.geometry.PointCloud()
    out.points = o3d.utility.Vector3dVector(pts[mask])
    if pcd.has_colors() and len(pcd.colors) == len(pts):
        out.colors = o3d.utility.Vector3dVector(np.asarray(pcd.colors)[mask])
    return out


def write_visual_ply(
    pcd: o3d.geometry.PointCloud, session_dir: str, filename: str
) -> None:
    if len(pcd.points) == 0:
        return
    os.makedirs(session_dir, exist_ok=True)
    stem, ext = os.path.splitext(filename)
    tmp = os.path.join(session_dir, f"{stem}.tmp{ext}")
    final = os.path.join(session_dir, filename)
    try:
        ok = o3d.io.write_point_cloud(tmp, pcd, write_ascii=False)
        if not ok or not os.path.isfile(tmp) or os.path.getsize(tmp) == 0:
            return
        os.replace(tmp, final)
    except Exception:
        logger.exception("write_visual_ply failed for %s", filename)
    finally:
        if os.path.isfile(tmp):
            try:
                os.unlink(tmp)
            except OSError:
                pass


def denoise_planar_surfaces(pcd: o3d.geometry.PointCloud) -> o3d.geometry.PointCloud:
    """Apply normal-aware bilateral / planar surface denoising so walls and floors
    are crisp and low-noise without destroying sharp 3D edges of interior fixtures."""
    work = o3d.geometry.PointCloud(pcd)
    n_pts = len(work.points)
    if n_pts < 120:
        return work
    try:
        pts = np.asarray(work.points, dtype=float).copy()
        z_floor = float(np.percentile(pts[:, 2], 2.0))
        floor_band = np.abs(pts[:, 2] - z_floor) <= 0.035
        if int(np.count_nonzero(floor_band)) >= 30:
            floor_med = float(np.median(pts[floor_band, 2]))
            pts[floor_band, 2] = floor_med + 0.35 * (pts[floor_band, 2] - floor_med)

        for ax in (0, 1):
            c_min = float(np.percentile(pts[:, ax], 2.0))
            c_max = float(np.percentile(pts[:, ax], 98.0))
            if c_max - c_min < 0.40:
                continue
            near_min = np.abs(pts[:, ax] - c_min) <= 0.030
            if int(np.count_nonzero(near_min)) >= 30:
                w_med = float(np.median(pts[near_min, ax]))
                pts[near_min, ax] = w_med + 0.40 * (pts[near_min, ax] - w_med)
            near_max = np.abs(pts[:, ax] - c_max) <= 0.030
            if int(np.count_nonzero(near_max)) >= 30:
                w_med = float(np.median(pts[near_max, ax]))
                pts[near_max, ax] = w_med + 0.40 * (pts[near_max, ax] - w_med)

        work.points = o3d.utility.Vector3dVector(pts)
    except Exception:
        return o3d.geometry.PointCloud(pcd)
    return work


def get_metric_vggt_cloud(
    vggt_prior,
    layout: dict | None = None,
    gravity: np.ndarray | None = None,
    reference_points: np.ndarray | None = None,
) -> tuple[o3d.geometry.PointCloud, o3d.geometry.PointCloud]:
    """Transform VGGT prediction point cloud to metric real-world coordinates
    using calibrated Sim(3) registration without post-hoc stretching to floorplan.json.

    Returns:
        (pcd_room, pcd_vio):
            - pcd_room: aligned to room canonical coordinates (floor at z=0, walls along axes)
            - pcd_vio: aligned to VIO metric coordinate system
    """
    empty = o3d.geometry.PointCloud()
    if vggt_prior is None or not hasattr(vggt_prior, "points") or len(vggt_prior.points) == 0:
        return empty, empty

    pts_vggt = np.asarray(vggt_prior.points, dtype=float)
    if len(pts_vggt) == 0:
        return empty, empty

    # Colors
    cols = getattr(vggt_prior, "colors", None)
    if cols is not None and len(cols) == len(pts_vggt):
        cols_arr = np.asarray(cols, dtype=float)
        if cols_arr.max() > 1.0 or np.issubdtype(cols.dtype, np.integer):
            cols_arr = np.clip(cols_arr / 255.0, 0.0, 1.0)
    else:
        cols_arr = np.full((len(pts_vggt), 3), 0.85, dtype=float)

    # 1. Determine calibrated Sim(3) registration parameters (s, R, t) for pcd_vio
    s_use, R_use, t_use = None, None, None
    if isinstance(layout, dict):
        diag_vggt = layout.get("diagnostics", {}).get("vggt_prior", {})
        if (
            diag_vggt.get("sim3_s") is not None
            and diag_vggt.get("sim3_R") is not None
            and diag_vggt.get("sim3_t") is not None
        ):
            try:
                s_use = float(diag_vggt["sim3_s"])
                R_use = np.asarray(diag_vggt["sim3_R"], dtype=float).reshape(3, 3)
                t_use = np.asarray(diag_vggt["sim3_t"], dtype=float).reshape(3)
            except Exception:
                s_use, R_use, t_use = None, None, None

    if s_use is None or R_use is None or t_use is None:
        from app.services.vggt_prior import (
            RobustVggtPointcloudRegistrar,
            estimate_sim3_camera_centers,
        )
        c_vggt = getattr(vggt_prior, "C_vggt", None)
        t_vio = getattr(vggt_prior, "t_vio", None)
        if c_vggt is not None and t_vio is not None and len(c_vggt) >= 3 and len(t_vio) >= 3:
            sim3 = estimate_sim3_camera_centers(c_vggt, t_vio)
            init_R = (
                sim3.R
                if (sim3.R is not None and np.all(np.isfinite(sim3.R)))
                else getattr(vggt_prior, "R0", np.eye(3))
            )
            init_s = (
                sim3.s
                if (sim3.s is not None and np.isfinite(sim3.s))
                else None
            )
        else:
            sim3 = None
            init_R = getattr(vggt_prior, "R0", np.eye(3))
            init_s = None

        h_room = layout_height_m(layout) if layout else float(DEFAULT_HEIGHT_M)
        reg = RobustVggtPointcloudRegistrar()
        ref_pts = (
            np.asarray(reference_points, dtype=float)
            if reference_points is not None and len(reference_points)
            else np.zeros((0, 3))
        )
        ok_reg, s_reg, R_reg, t_reg, rmse_reg = reg.register(
            pts_vggt,
            ref_pts,
            initial_R=init_R,
            initial_s=init_s,
            height_m=h_room,
            camera_centers=t_vio,
        )
        use_pointcloud_reg = ok_reg and (
            not (sim3 and sim3.ok)
            or (rmse_reg is not None and sim3.rmse is not None and rmse_reg < sim3.rmse - 0.01)
        )
        if use_pointcloud_reg:
            s_use, R_use, t_use = s_reg, R_reg, t_reg
        elif sim3 and sim3.ok:
            s_use, R_use, t_use = sim3.s, sim3.R, sim3.t
        else:
            s_use, R_use, t_use = 1.0, np.eye(3), np.zeros(3)

    from app.services.vggt_prior import apply_sim3
    p_vio = apply_sim3(pts_vggt, s_use, R_use, t_use)

    pcd_vio = o3d.geometry.PointCloud()
    pcd_vio.points = o3d.utility.Vector3dVector(p_vio)
    pcd_vio.colors = o3d.utility.Vector3dVector(cols_arr)

    # 2. Produce pcd_room in canonical Manhattan coordinates using calibrated isotropic scale s_use
    from app.services.manhattan_alignment import (
        align_point_cloud_to_manhattan,
        clean_and_align_point_cloud,
        yaw_rotation_matrix,
    )
    from app.services.occupancy_layout import estimate_manhattan_angle_bbox_sharpness

    clean_ply = getattr(vggt_prior, "clean_ply_path", None)
    sess_dir = (
        os.path.dirname(os.path.dirname(clean_ply))
        if (clean_ply and os.path.basename(os.path.dirname(clean_ply)) == "vggt")
        else None
    )
    if clean_ply and os.path.isfile(clean_ply) and os.path.getsize(clean_ply) > 0:
        manhattan_pcd = o3d.io.read_point_cloud(clean_ply)
    else:
        work_cloud = o3d.geometry.PointCloud()
        work_cloud.points = o3d.utility.Vector3dVector(pts_vggt)
        work_cloud.colors = o3d.utility.Vector3dVector(cols_arr)

        if len(work_cloud.points) > 500_000:
            manhattan_pcd, _ = clean_and_align_point_cloud(
                work_cloud, voxel=0.003, stat_neighbors=16, stat_std=2.8, session_dir=sess_dir
            )
        elif len(work_cloud.points) > 50:
            cleaned_sor, _ = work_cloud.remove_statistical_outlier(nb_neighbors=16, std_ratio=2.8)
            manhattan_pcd, _ = align_point_cloud_to_manhattan(cleaned_sor, session_dir=sess_dir)
        else:
            manhattan_pcd = work_cloud

    if len(manhattan_pcd.points) >= 50:
        pts_m = np.asarray(manhattan_pcd.points, dtype=float).copy()
        metric_scale = float(s_use) if (s_use is not None and np.isfinite(s_use) and s_use > 0) else 1.0
        xy_layout = layout_xy_polygon(layout) if isinstance(layout, dict) else np.zeros((0, 2))
        if len(xy_layout) >= 4:
            w_r = float(np.ptp(xy_layout[:, 0]))
            d_r = float(np.ptp(xy_layout[:, 1]))
            h_r = float(layout_height_m(layout))
            spans_m = (np.percentile(pts_m, 98, axis=0) - np.percentile(pts_m, 2, axis=0)) * metric_scale
            if (
                d_r > 0.5
                and h_r > 1.5
                and abs(spans_m[1] - d_r) + abs(spans_m[2] - h_r)
                > abs(spans_m[2] - d_r) + abs(spans_m[1] - h_r) + 0.30
            ):
                pts_m = pts_m[:, [0, 2, 1]]
                pts_m[:, 2] = -pts_m[:, 2]
        else:
            w_r, d_r, h_r = 0.0, 0.0, 0.0

        z_lo = float(np.percentile(pts_m[:, 2], 2))
        z_hi = float(np.percentile(pts_m[:, 2], 98))
        dz = z_hi - z_lo
        mid_mask = (pts_m[:, 2] >= z_lo + 0.15 * dz) & (pts_m[:, 2] <= z_lo + 0.85 * dz)
        slice_xy = pts_m[mid_mask, :2] if int(np.count_nonzero(mid_mask)) >= 200 else pts_m[:, :2]
        if len(slice_xy) >= 100:
            sharp_info = estimate_manhattan_angle_bbox_sharpness(slice_xy)
            residual_yaw = -float(sharp_info.get("angle_deg", 0.0))
            impr = float(sharp_info.get("bbox_area_improvement", 0.0))
            if abs(residual_yaw) > 0.2 and impr >= 0.02:
                r_yaw = yaw_rotation_matrix(2, residual_yaw)
                c_xy = pts_m.mean(axis=0)
                pts_m = (pts_m - c_xy) @ r_yaw.T + c_xy

        span_x0 = float(np.percentile(pts_m[:, 0], 98) - np.percentile(pts_m[:, 0], 2))
        span_y0 = float(np.percentile(pts_m[:, 1], 98) - np.percentile(pts_m[:, 1], 2))
        if (w_r >= d_r and span_y0 > span_x0) or (d_r > w_r and span_x0 > span_y0):
            r90 = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
            pts_m = pts_m @ r90.T

        raw_spans = np.percentile(pts_m, 98, axis=0) - np.percentile(pts_m, 2, axis=0)
        if w_r > 0.5 and d_r > 0.5 and h_r > 1.0 and np.all(raw_spans > 0.2):
            layout_spans = np.array([w_r, d_r, h_r], dtype=float)
            max_iso_scale = float(np.median(layout_spans / raw_spans))
            if metric_scale > max_iso_scale:
                metric_scale = max_iso_scale

        if len(xy_layout) >= 6:
            try:
                poly_in = Polygon(xy_layout).buffer(0.05)
                sub_idx = np.linspace(0, len(pts_m) - 1, min(5000, len(pts_m)), dtype=int)
                best_in = -1.0
                best_fx_fy = (1.0, 1.0)
                for fx in (1.0, -1.0):
                    for fy in (1.0, -1.0):
                        tx = pts_m[sub_idx, 0] * fx
                        ty = pts_m[sub_idx, 1] * fy
                        tx = (tx - float(np.percentile(tx, 2))) * metric_scale
                        ty = (ty - float(np.percentile(ty, 2))) * metric_scale
                        ratio = float(np.mean(contains_xy(poly_in, tx, ty)))
                        if ratio > best_in:
                            best_in = ratio
                            best_fx_fy = (fx, fy)
                pts_m[:, 0] *= best_fx_fy[0]
                pts_m[:, 1] *= best_fx_fy[1]
            except Exception:
                pass
        x_p02 = float(np.percentile(pts_m[:, 0], 2))
        y_p02 = float(np.percentile(pts_m[:, 1], 2))
        z_floor = float(np.percentile(pts_m[:, 2], 1.5))
        pts_m[:, 0] = (pts_m[:, 0] - x_p02) * metric_scale
        pts_m[:, 1] = (pts_m[:, 1] - y_p02) * metric_scale
        pts_m[:, 2] = (pts_m[:, 2] - z_floor) * metric_scale
        pcd_room = o3d.geometry.PointCloud(manhattan_pcd)
        pcd_room.points = o3d.utility.Vector3dVector(pts_m)
        pcd_room = denoise_planar_surfaces(pcd_room)
    else:
        pcd_room = manhattan_pcd

    return pcd_room, pcd_vio


def complete_amodal_walls_and_floor(
    pcd: o3d.geometry.PointCloud,
    layout: dict | None,
    grid_step_m: float = 0.025,
) -> o3d.geometry.PointCloud:
    """Inpaint / complete occluded structural wall and floor surfaces behind interior obstacles."""
    work = o3d.geometry.PointCloud(pcd)
    if len(work.points) < 50 or not isinstance(layout, dict):
        return work
    obstacles = layout.get("obstacles")
    if not isinstance(obstacles, list) or not obstacles:
        return work
    xy = layout_xy_polygon(layout)
    if len(xy) < 4:
        return work
    try:
        poly = Polygon(xy)
        if not poly.is_valid:
            poly = poly.buffer(0)
        if poly.is_empty:
            return work
    except Exception:
        return work

    pts = np.asarray(work.points, dtype=float)
    has_cols = work.has_colors() and len(work.colors) == len(pts)
    cols = np.asarray(work.colors, dtype=float) if has_cols else np.full((len(pts), 3), 0.85, dtype=float)
    height_m = layout_height_m(layout)

    floor_mask = np.abs(pts[:, 2]) <= 0.05
    floor_color = (
        np.median(cols[floor_mask], axis=0)
        if int(np.count_nonzero(floor_mask)) >= 10
        else np.median(cols, axis=0)
    )
    floor_pts_xy = pts[floor_mask, :2] if np.any(floor_mask) else np.zeros((0, 2))

    new_pts: list[np.ndarray] = []
    new_cols: list[np.ndarray] = []
    n_walls = len(xy)

    portals_by_wall: dict[int, list[tuple[float, float, float, float]]] = {}
    for p in layout.get("portals") or []:
        w_idx = p.get("wall_index")
        if w_idx is None or not (0 <= int(w_idx) < n_walls):
            continue
        wi = int(w_idx)
        a_w, b_w = xy[wi], xy[(wi + 1) % n_walls]
        d_w = b_w - a_w
        l_w = float(np.hypot(d_w[0], d_w[1]))
        if l_w <= 1e-6:
            continue
        t_w = d_w / l_w
        s0 = float(p.get("offset_along_wall_m", 0.0))
        s1 = s0 + float(p.get("width_m", 0.9))
        z0 = float(p.get("sill_height_m", 0.0))
        z1 = z0 + float(p.get("height_m", 2.1))
        portals_by_wall.setdefault(wi, []).append((s0, s1, z0, z1))

    for obs in obstacles:
        if not isinstance(obs, dict):
            continue
        bmin = obs.get("bbox_min")
        bmax = obs.get("bbox_max")
        if (
            not isinstance(bmin, (list, tuple))
            or not isinstance(bmax, (list, tuple))
            or len(bmin) < 3
            or len(bmax) < 3
        ):
            continue
        try:
            ox0, oy0, oz0 = float(bmin[0]), float(bmin[1]), max(0.0, float(bmin[2]))
            ox1, oy1, oz1 = float(bmax[0]), float(bmax[1]), min(height_m, float(bmax[2]))
        except (TypeError, ValueError):
            continue
        if ox1 - ox0 < 0.05 or oy1 - oy0 < 0.05:
            continue

        # 1. Complete floor surface under the obstacle footprint (z = 0.0)
        xs = np.arange(ox0, ox1 + 0.5 * grid_step_m, grid_step_m)
        ys = np.arange(oy0, oy1 + 0.5 * grid_step_m, grid_step_m)
        if len(xs) > 0 and len(ys) > 0:
            gx, gy = np.meshgrid(xs, ys)
            cand_xy = np.column_stack([gx.ravel(), gy.ravel()])
            in_poly = np.asarray(contains_xy(poly, cand_xy[:, 0], cand_xy[:, 1]), dtype=bool)
            cand_xy = cand_xy[in_poly]
            if len(cand_xy) > 0:
                if len(floor_pts_xy) > 0:
                    near_f = (
                        (floor_pts_xy[:, 0] >= ox0 - 0.15)
                        & (floor_pts_xy[:, 0] <= ox1 + 0.15)
                        & (floor_pts_xy[:, 1] >= oy0 - 0.15)
                        & (floor_pts_xy[:, 1] <= oy1 + 0.15)
                    )
                    sub_f = floor_pts_xy[near_f]
                    if len(sub_f) > 0:
                        d2 = np.min(
                            np.sum((cand_xy[:, None, :] - sub_f[None, :, :]) ** 2, axis=2),
                            axis=1,
                        )
                        cand_xy = cand_xy[d2 > (0.8 * grid_step_m) ** 2]
                if len(cand_xy) > 0:
                    f_pts = np.column_stack([cand_xy, np.zeros(len(cand_xy), dtype=float)])
                    new_pts.append(f_pts)
                    new_cols.append(np.tile(floor_color, (len(f_pts), 1)))

        # 2. Complete occluded wall surface behind the obstacle
        obs_corners = np.array([[ox0, oy0], [ox1, oy0], [ox1, oy1], [ox0, oy1]], dtype=float)
        for wi in range(n_walls):
            a_w = xy[wi]
            b_w = xy[(wi + 1) % n_walls]
            diff = b_w - a_w
            l_w = float(np.hypot(diff[0], diff[1]))
            if l_w <= 0.05:
                continue
            t_w = diff / l_w
            n_w = np.array([-t_w[1], t_w[0]], dtype=float)
            rel_c = obs_corners - a_w
            s_c = rel_c @ t_w
            d_c = np.abs(rel_c @ n_w)
            s_lo = max(0.0, float(np.min(s_c)) - 0.03)
            s_hi = min(l_w, float(np.max(s_c)) + 0.03)
            if s_hi <= s_lo or float(np.min(d_c)) > 0.30:
                continue

            # Estimate local wall color from nearby wall points (especially above obstacle)
            rel_all = pts[:, :2] - a_w
            s_all = rel_all @ t_w
            d_all = np.abs(rel_all @ n_w)
            wall_band = (
                (d_all <= 0.06)
                & (s_all >= max(0.0, s_lo - 0.25))
                & (s_all <= min(l_w, s_hi + 0.25))
                & (pts[:, 2] >= 0.05)
                & (pts[:, 2] <= height_m - 0.05)
            )
            wall_color = (
                np.median(cols[wall_band], axis=0)
                if int(np.count_nonzero(wall_band)) >= 10
                else np.median(cols, axis=0)
            )
            existing_wall_sz = (
                np.column_stack([s_all[wall_band], pts[wall_band, 2]])
                if np.any(wall_band)
                else np.zeros((0, 2))
            )

            ss = np.arange(s_lo, s_hi + 0.5 * grid_step_m, grid_step_m)
            zs = np.arange(0.0, min(height_m, oz1 + 0.05) + 0.5 * grid_step_m, grid_step_m)
            if len(ss) == 0 or len(zs) == 0:
                continue
            gs, gz = np.meshgrid(ss, zs)
            cand_sz = np.column_stack([gs.ravel(), gz.ravel()])

            for ps0, ps1, pz0, pz1 in portals_by_wall.get(wi, []):
                in_portal = (
                    (cand_sz[:, 0] >= ps0)
                    & (cand_sz[:, 0] <= ps1)
                    & (cand_sz[:, 1] >= pz0)
                    & (cand_sz[:, 1] <= pz1)
                )
                cand_sz = cand_sz[~in_portal]
            if len(cand_sz) == 0:
                continue

            if len(existing_wall_sz) > 0:
                d2_w = np.min(
                    np.sum((cand_sz[:, None, :] - existing_wall_sz[None, :, :]) ** 2, axis=2),
                    axis=1,
                )
                cand_sz = cand_sz[d2_w > (0.8 * grid_step_m) ** 2]
            if len(cand_sz) == 0:
                continue

            w_pts = np.column_stack(
                [
                    a_w[0] + cand_sz[:, 0] * t_w[0],
                    a_w[1] + cand_sz[:, 0] * t_w[1],
                    cand_sz[:, 1],
                ]
            )
            new_pts.append(w_pts)
            new_cols.append(np.tile(wall_color, (len(w_pts), 1)))

    if not new_pts:
        return work

    all_pts = np.vstack([pts, *new_pts])
    all_cols = np.vstack([cols, *new_cols])
    if len(all_pts) > VISUAL_POINT_CLOUD_MAX:
        all_pts = all_pts[:VISUAL_POINT_CLOUD_MAX]
        all_cols = all_cols[:VISUAL_POINT_CLOUD_MAX]
    out = o3d.geometry.PointCloud()
    out.points = o3d.utility.Vector3dVector(all_pts)
    if has_cols:
        out.colors = o3d.utility.Vector3dVector(np.clip(all_cols, 0.0, 1.0))
    return out


def build_poisson_mesh_from_cloud(
    pcd: o3d.geometry.PointCloud,
    layout: dict | None = None,
    depth: int = 8,
) -> o3d.geometry.TriangleMesh:
    """Build a watertight triangle mesh using Poisson Surface Reconstruction."""
    empty = o3d.geometry.TriangleMesh()
    if len(pcd.points) < 100:
        return empty

    # Voxel downsample to uniform density (~1.5cm)
    pcd_ds = pcd.voxel_down_sample(0.015)
    if len(pcd_ds.points) < 50:
        pcd_ds = o3d.geometry.PointCloud(pcd)

    # Estimate normals
    pcd_ds.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.08, max_nn=30)
    )

    # Orient normals towards room center if layout is known
    xy = layout_xy_polygon(layout) if isinstance(layout, dict) else np.zeros((0, 2))
    h = float(layout_height_m(layout)) if isinstance(layout, dict) else float(DEFAULT_HEIGHT_M)
    if len(xy) >= 4:
        center = np.array([float(np.mean(xy[:, 0])), float(np.mean(xy[:, 1])), h / 2.0])
        pcd_ds.orient_normals_towards_camera_location(center)
    elif isinstance(layout, dict) and "room" in layout:
        w = float(layout.get("room", {}).get("width_m", 2.0))
        d = float(layout.get("room", {}).get("depth_m", 1.5))
        center = np.array([w / 2.0, d / 2.0, h / 2.0])
        pcd_ds.orient_normals_towards_camera_location(center)
    else:
        pcd_ds.orient_normals_consistent_tangent_plane(k=15)

    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd_ds, depth=depth
    )
    if len(mesh.vertices) == 0 or len(mesh.triangles) == 0:
        return empty

    # Density trimming: remove vertices with low density (lowest 5%)
    densities = np.asarray(densities)
    if len(densities) == len(mesh.vertices):
        density_thresh = float(np.quantile(densities, 0.05))
        mesh.remove_vertices_by_mask(densities < density_thresh)

    # Crop to room bounding box with margin
    margin = float(ROOM_CROP_MARGIN_M)
    p_pts = np.asarray(pcd_ds.points)
    p_min = p_pts.min(axis=0) if len(p_pts) else np.zeros(3)
    p_max = p_pts.max(axis=0) if len(p_pts) else np.ones(3)
    min_x = min(float(xy[:, 0].min()), p_min[0]) if len(xy) >= 4 else p_min[0]
    max_x = max(float(xy[:, 0].max()), p_max[0]) if len(xy) >= 4 else p_max[0]
    min_y = min(float(xy[:, 1].min()), p_min[1]) if len(xy) >= 4 else p_min[1]
    max_y = max(float(xy[:, 1].max()), p_max[1]) if len(xy) >= 4 else p_max[1]
    min_b = np.array([min_x - margin, min_y - margin, -0.05])
    max_b = np.array([max_x + margin, max_y + margin, max(h, p_max[2]) + 0.05])
    bbox = o3d.geometry.AxisAlignedBoundingBox(min_bound=min_b, max_bound=max_b)
    mesh = mesh.crop(bbox)

    # Clean degenerate artifacts
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_duplicated_vertices()

    # Ensure vertex normals are computed
    if not mesh.has_vertex_normals():
        mesh.compute_vertex_normals()

    # If triangle count exceeds 150,000, simplify down to 120,000 to keep it lightweight (> 80,000)
    if len(mesh.triangles) > 150000:
        mesh = mesh.simplify_quadric_decimation(target_number_of_triangles=120000)
        mesh.compute_vertex_normals()

    return mesh


def build_visual_clouds(
    mesh: o3d.geometry.TriangleMesh,
    layout: dict,
    gravity: np.ndarray | None,
    number_of_points: int | None = None,
    vggt_prior=None,
) -> tuple[o3d.geometry.PointCloud, o3d.geometry.PointCloud]:
    empty = o3d.geometry.PointCloud()
    if vggt_prior is not None and hasattr(vggt_prior, "points") and len(vggt_prior.points) >= 50:
        ref_pts = np.asarray(mesh.vertices) if len(mesh.vertices) else None
        pcd_room, _ = get_metric_vggt_cloud(
            vggt_prior, layout, gravity, reference_points=ref_pts
        )
        if len(pcd_room.points) > 0:
            cleaned = remove_visual_outliers(pcd_room)
            cropped = crop_to_layout_polygon(cleaned, layout)
            completed = complete_amodal_walls_and_floor(cropped, layout)
            doll = cut_ceiling(completed, layout_height_m(layout))
            return completed, doll

    sampled = sample_visual_cloud(mesh, number_of_points=number_of_points)
    if len(sampled.points) == 0:
        return empty, empty
    framed = apply_layout_frames(sampled, layout, gravity)
    cleaned = remove_visual_outliers(framed)
    cropped = crop_to_layout_polygon(cleaned, layout)
    completed = complete_amodal_walls_and_floor(cropped, layout)
    doll = cut_ceiling(completed, layout_height_m(layout))
    return completed, doll





