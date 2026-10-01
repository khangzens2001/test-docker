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


def build_visual_clouds(
    mesh: o3d.geometry.TriangleMesh,
    layout: dict,
    gravity: np.ndarray | None,
    number_of_points: int | None = None,
) -> tuple[o3d.geometry.PointCloud, o3d.geometry.PointCloud]:
    empty = o3d.geometry.PointCloud()
    sampled = sample_visual_cloud(mesh, number_of_points=number_of_points)
    if len(sampled.points) == 0:
        return empty, empty
    framed = apply_layout_frames(sampled, layout, gravity)
    cleaned = remove_visual_outliers(framed)
    cropped = crop_to_layout_polygon(cleaned, layout)
    doll = cut_ceiling(cropped, layout_height_m(layout))
    return cropped, doll




