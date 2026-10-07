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


def get_metric_vggt_cloud(
    vggt_prior,
    layout: dict | None = None,
    gravity: np.ndarray | None = None,
    reference_points: np.ndarray | None = None,
) -> tuple[o3d.geometry.PointCloud, o3d.geometry.PointCloud]:
    """Transform VGGT prediction point cloud to metric real-world coordinates.

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

    # 1. Determine Sim(3) registration parameters (s, R, t)
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

        h_room = layout_height_m(layout) if layout else 2.23
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
        use_pointcloud_reg = ok_reg and (not (sim3 and sim3.ok) or (rmse_reg is not None and sim3.rmse is not None and rmse_reg < sim3.rmse - 0.01))
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

    # 2. Transform to room canonical coordinates
    if isinstance(layout, dict):
        pcd_room = apply_layout_frames(pcd_vio, layout, gravity)
        pts_r = np.asarray(pcd_room.points).copy()
        if len(pts_r) > 100:
            z_floor = float(np.percentile(pts_r[:, 2], 0.5))
            pts_r[:, 2] -= z_floor
            pcd_room.points = o3d.utility.Vector3dVector(pts_r)
    else:
        pcd_room = o3d.geometry.PointCloud(pcd_vio)

    return pcd_room, pcd_vio


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
    h = float(layout_height_m(layout)) if isinstance(layout, dict) else 2.23
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
    if len(xy) >= 4:
        min_b = np.array([float(xy[:, 0].min()) - margin, float(xy[:, 1].min()) - margin, -0.05])
        max_b = np.array([float(xy[:, 0].max()) + margin, float(xy[:, 1].max()) + margin, h + 0.05])
        bbox = o3d.geometry.AxisAlignedBoundingBox(min_bound=min_b, max_bound=max_b)
        mesh = mesh.crop(bbox)
    elif isinstance(layout, dict) and "room" in layout:
        w = float(layout.get("room", {}).get("width_m", 2.0))
        d = float(layout.get("room", {}).get("depth_m", 1.5))
        min_b = np.array([-margin, -margin, -0.05])
        max_b = np.array([w + margin, d + margin, h + 0.05])
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
            doll = cut_ceiling(cropped, layout_height_m(layout))
            return cropped, doll

    sampled = sample_visual_cloud(mesh, number_of_points=number_of_points)
    if len(sampled.points) == 0:
        return empty, empty
    framed = apply_layout_frames(sampled, layout, gravity)
    cleaned = remove_visual_outliers(framed)
    cropped = crop_to_layout_polygon(cleaned, layout)
    doll = cut_ceiling(cropped, layout_height_m(layout))
    return cropped, doll




