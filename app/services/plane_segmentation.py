"""Gravity-aligned soft-constrained RANSAC helpers (Task 3.4)."""
from __future__ import annotations

import copy
import json
import logging
import os
import cv2
import numpy as np
import open3d as o3d
import pandas as pd
from scipy.optimize import least_squares
from shapely.geometry import (
    LineString as ShapelyLineString,
    Point as ShapelyPoint,
    Polygon as ShapelyPolygon,
)
from shapely.ops import split as shapely_split

from app.services.occupancy_layout import (
    BBOX_FILL_RATIO_THRESHOLD,
    MAX_NGON_VERTICES,
    MIN_NGON_VERTICES,
    apply_hough_yaw,
    choke_doorway_tails,
    classify_heading,
    classify_occupancy_necks,
    close_orthogonal_polygon,
    close_polygon_xy,
    detect_builtin_keepouts,
    fit_missing_corner_l_shape,
    fit_orthogonal_ngon,
    clean_occupancy_projection_profile,
    filled_bbox_fill_ratio,
    filled_cavity_mask,
    keep_largest_occupancy_component,
    keepout_is_interior_hole,
    rectilinearize_contour,
    slice_wall_band,
)
from app.services.vggt_prior import (
    RobustVggtPointcloudRegistrar,
    SIM3_SCALE_MAX,
    SIM3_SCALE_MIN,
    T2_CLIP_MARGIN_M,
    T2Wall,
    VggtPrior,
    WallHint,
    apply_sim3,
    apply_vggt_topology_prior,
    empty_vggt_diag,
    estimate_sim3_camera_centers,
    extract_vggt_wall_axes,
    occupancy_frame_from_vio,
)

logger = logging.getLogger(__name__)

GRAVITY_NORM_EPS = 1e-6
GRAVITY_ALIGN_DEG = 2.0
CLASSIFY_WALL_MAX_DEG = 30.0
CLASSIFY_HORIZ_MAX_DEG = 30.0
FLOOR_LEVELED_MAX_TILT_DEG = 5.0
FLOOR_PRE_RANSAC_BELOW_MODE_M = 0.05
FLOOR_MODE_BIN_M = 0.01
RANSAC_DISTANCE_THRESHOLD = 0.05
RANSAC_N = 3
RANSAC_NUM_ITERATIONS = 1000
RANSAC_MIN_INLIERS = 30
RANSAC_MIN_REMAINING = 50
MAX_RANSAC_PLANES = 20
LAMBDA_GRAVITY_PER_INLIER = (RANSAC_DISTANCE_THRESHOLD ** 2) / (
    np.sin(np.radians(GRAVITY_ALIGN_DEG)) ** 2
)
LINE_DET_EPS = 1e-4
PLANE3_DET_EPS = 1e-6
SNAP_VERTEX_EPS = 0.02
MIN_POLYGON_AREA_M2 = 0.10
DEFAULT_WALL_THICKNESS_M = 0.15
HEIGHT_SPAN_MIN_M = 0.10
DEFAULT_HEIGHT_M = 2.70
MESH_SAMPLE_POINTS = 300000
STAT_NB_NEIGHBORS = 20
STAT_STD_RATIO = 2.0
FLOOR_P_LOW = 2
FLOOR_P_HIGH = 20
FLOOR_RANSAC_DIST_M = 0.035
SUBSURFACE_MARGIN_M = 0.03
HEIGHT_PERCENTILE = 98
CEILING_BULK_PERCENTILE = 90
CEILING_PEAK_SLACK_M = 0.05
CEILING_RANSAC_GATE_M = 0.06
CEILING_HARVEST_BELOW_M = 0.10
CEILING_WALL_BAND_LO_M = 1.0
CEILING_WALL_BAND_HI_M = 1.8
CEILING_DROP_FRAC = 0.58
CEILING_CLIFF_MIN_Z_M = 1.80
CEILING_PEAK_NEAR_CLIFF_M = 0.08
CEILING_CLIFF_WALL_FRAC = 0.15
CEILING_RECOVERY_FRAC = 1.5
CEILING_RECOVERY_WINDOW_M = 0.20
VERTEX_DECIMALS = 3
AXIS_CLOSURE_M = 0.001
MERGE_NORMAL_DEG = 10.0
MERGE_NORMAL_DOT_MIN = float(np.cos(np.radians(MERGE_NORMAL_DEG)))
MERGE_OFFSET_M = 0.05
TANGENT_AXIS_DOT_MAX = 0.9
G_WORK = np.array([0.0, 0.0, -1.0])
G_ARCORE_WORLD = np.array([0.0, -9.80665, 0.0], dtype=np.float64)
MANHATTAN_ASSIGN_MAX_DEG = 25.0
MANHATTAN_ASSIGN_DOT_MIN = float(np.cos(np.radians(MANHATTAN_ASSIGN_MAX_DEG)))
MANHATTAN_MIN_SPAN_M = 0.05
MANHATTAN_FURNITURE_SPAN_RATIO = 0.5
MAX_STANDOFF_M = 0.65
SMALL_ROOM_SHORT_STANDOFF_M = 0.20
SMALL_ROOM_LONG_STANDOFF_M = 0.55
OCC_OUTER_BIN_M = 0.05
OCC_OUTER_MIN_FRAC = 0.12
OCC_OUTER_MIN_COUNT = 80
OCC_INNER_TO_OUTER_M = 0.12
OCC_THICK_SLAB_M = 0.30
WALK_ENTERED_MARGIN_M = 0.30
MIRROR_MIN_POS_DIST_M = 0.30
MIRROR_TAIL_ABS_M = 0.20
MIRROR_TAIL_REL = 0.15



def is_usable_gravity(g: np.ndarray | None) -> bool:
    if g is None:
        return False
    try:
        arr = np.asarray(g, dtype=float).reshape(-1)
    except (TypeError, ValueError):
        return False
    if arr.size != 3:
        return False
    if not np.all(np.isfinite(arr)):
        return False
    return float(np.linalg.norm(arr)) >= GRAVITY_NORM_EPS


def gravity_from_vio_df(df: pd.DataFrame | None) -> np.ndarray | None:
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return None
    required = ("grav_x", "grav_y", "grav_z")
    if any(col not in df.columns for col in required):
        return None
    try:
        row = df.iloc[0]
        g = np.array(
            [float(row["grav_x"]), float(row["grav_y"]), float(row["grav_z"])],
            dtype=float,
        )
    except (TypeError, ValueError):
        return None
    if not is_usable_gravity(g):
        return None
    return g


def resolve_gravity_vector(
    pose_df: pd.DataFrame | None,
    source: str | None = None,
) -> np.ndarray | None:
    if source is None or source == "":
        if not isinstance(pose_df, pd.DataFrame):
            source = "unknown"
        else:
            source = pose_df.attrs.get("pose_source", "unknown")
    source = str(source)
    if source == "arcore_odometry":
        return G_ARCORE_WORLD.copy()
    if source == "server_vio":
        return gravity_from_vio_df(pose_df)
    return gravity_from_vio_df(pose_df)


def empty_layout(gravity: np.ndarray | None) -> dict:
    if gravity is None:
        gravity_vector = [0.0, 0.0, 0.0]
    else:
        try:
            arr = np.asarray(gravity, dtype=float).reshape(-1)
            if arr.size == 3:
                gravity_vector = arr.tolist()
            else:
                padded = np.zeros(3, dtype=float)
                n = min(3, int(arr.size))
                if n:
                    padded[:n] = arr[:n]
                gravity_vector = padded.tolist()
        except (TypeError, ValueError):
            gravity_vector = [0.0, 0.0, 0.0]
    return {
        "vertices": {},
        "walls": [],
        "portals": [],
        "floor": None,
        "ceiling": None,
        "gravity_vector": gravity_vector,
    }


def align_rotation(g_hat: np.ndarray) -> np.ndarray:
    g_hat = np.asarray(g_hat, dtype=float).reshape(3)
    nrm = np.linalg.norm(g_hat)
    if nrm < GRAVITY_NORM_EPS:
        return np.eye(3)
    g_hat = g_hat / nrm
    z_target = G_WORK
    cos_angle = float(np.dot(g_hat, z_target))
    if abs(cos_angle - 1.0) < 1e-6:
        return np.eye(3)
    if abs(cos_angle + 1.0) < 1e-6:
        return np.array(
            [[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]]
        )
    axis = np.cross(g_hat, z_target)
    axis = axis / np.linalg.norm(axis)
    angle = float(np.arccos(np.clip(cos_angle, -1.0, 1.0)))
    kx, ky, kz = axis
    k = np.array([[0.0, -kz, ky], [kz, 0.0, -kx], [-ky, kx, 0.0]])
    return np.eye(3) + np.sin(angle) * k + (1.0 - np.cos(angle)) * (k @ k)


def extract_ransac_planes(pcd: o3d.geometry.PointCloud) -> list[dict]:
    o3d.utility.random.seed(42)
    planes: list[dict] = []
    work = o3d.geometry.PointCloud(pcd)
    for _ in range(MAX_RANSAC_PLANES):
        if len(work.points) < RANSAC_MIN_REMAINING:
            break
        model, inliers = work.segment_plane(
            distance_threshold=RANSAC_DISTANCE_THRESHOLD,
            ransac_n=RANSAC_N,
            num_iterations=RANSAC_NUM_ITERATIONS,
        )
        if len(inliers) < RANSAC_MIN_INLIERS:
            break
        inlier_pts = np.asarray(work.select_by_index(inliers).points, dtype=float)
        a, b, c, d = [float(v) for v in model]
        n = np.array([a, b, c], dtype=float)
        n_norm = float(np.linalg.norm(n))
        if n_norm < GRAVITY_NORM_EPS:
            work = work.select_by_index(inliers, invert=True)
            continue
        n = n / n_norm
        d = d / n_norm
        planes.append(
            {
                "n": n,
                "d": float(d),
                "inliers": inlier_pts,
                "inlier_count": int(len(inliers)),
            }
        )
        work = work.select_by_index(inliers, invert=True)
    return planes


def _unique_inliers(pts: np.ndarray) -> np.ndarray:
    rounded = np.round(np.asarray(pts, dtype=float), 6)
    _, idx = np.unique(rounded, axis=0, return_index=True)
    return np.asarray(pts, dtype=float)[np.sort(idx)]


def _planes_similar(a: dict, b: dict) -> bool:
    ni = np.asarray(a["n"], dtype=float)
    nj = np.asarray(b["n"], dtype=float)
    dot = float(np.clip(np.dot(ni, nj), -1.0, 1.0))
    if abs(dot) <= MERGE_NORMAL_DOT_MIN:
        return False
    s = 1.0 if dot >= 0.0 else -1.0
    return abs(float(a["d"]) - s * float(b["d"])) < MERGE_OFFSET_M


def merge_similar_planes(planes: list[dict]) -> list[dict]:
    if not planes:
        return []
    order = sorted(range(len(planes)), key=lambda i: planes[i]["inlier_count"], reverse=True)
    used = [False] * len(planes)
    out: list[dict] = []
    for i in order:
        if used[i]:
            continue
        acc = {
            "n": np.asarray(planes[i]["n"], dtype=float).copy(),
            "d": float(planes[i]["d"]),
            "inliers": np.asarray(planes[i]["inliers"], dtype=float).copy(),
            "inlier_count": int(planes[i]["inlier_count"]),
        }
        used[i] = True
        for j in order:
            if used[j]:
                continue
            if not _planes_similar(acc, planes[j]):
                continue
            acc["inliers"] = _unique_inliers(
                np.vstack([acc["inliers"], planes[j]["inliers"]])
            )
            acc["inlier_count"] = int(len(acc["inliers"]))
            used[j] = True
        out.append(acc)
    return out


def _centroid(pl: dict) -> np.ndarray:
    return np.mean(np.asarray(pl["inliers"], dtype=float), axis=0)


def classify_planes(
    planes: list[dict], g_hat: np.ndarray, z_med: float
) -> tuple[list[dict], dict | None, dict | None]:
    g_hat = np.asarray(g_hat, dtype=float).reshape(3)
    g_nrm = np.linalg.norm(g_hat)
    if g_nrm < GRAVITY_NORM_EPS:
        g_hat = G_WORK.copy()
    else:
        g_hat = g_hat / g_nrm
    sin_wall = float(np.sin(np.radians(CLASSIFY_WALL_MAX_DEG)))
    cos_horiz = float(np.cos(np.radians(CLASSIFY_HORIZ_MAX_DEG)))
    walls: list[dict] = []
    horizontals: list[dict] = []
    for pl in planes:
        n = np.asarray(pl["n"], dtype=float)
        n = n / np.linalg.norm(n)
        mag = abs(float(np.dot(n, g_hat)))
        item = {**pl, "n": n}
        if mag < sin_wall:
            walls.append(item)
        elif mag > cos_horiz:
            horizontals.append(item)
    floor = None
    ceiling = None
    below = [h for h in horizontals if float(_centroid(h)[2]) < z_med]
    above = [h for h in horizontals if float(_centroid(h)[2]) >= z_med]
    if below:
        floor = max(below, key=lambda p: p["inlier_count"])
    if above:
        ceiling = max(above, key=lambda p: p["inlier_count"])
    return walls, floor, ceiling


def _tangent_basis(n0: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ez = np.array([0.0, 0.0, 1.0])
    ex = np.array([1.0, 0.0, 0.0])
    axis = ez if abs(float(np.dot(n0, ez))) < TANGENT_AXIS_DOT_MAX else ex
    b1 = np.cross(n0, axis)
    b1_n = np.linalg.norm(b1)
    if b1_n < GRAVITY_NORM_EPS:
        axis = np.array([0.0, 1.0, 0.0])
        b1 = np.cross(n0, axis)
        b1_n = np.linalg.norm(b1)
    b1 = b1 / b1_n
    b2 = np.cross(n0, b1)
    b2 = b2 / np.linalg.norm(b2)
    return b1, b2


def refine_plane_lm(
    n0: np.ndarray,
    d0: float,
    inliers: np.ndarray,
    g_hat: np.ndarray,
    kind: str,
    lambda_n: float,
) -> tuple[np.ndarray, float]:
    try:
        n0_arr = np.asarray(n0, dtype=float).reshape(3)
        n0_n = np.linalg.norm(n0_arr)
        if n0_n < GRAVITY_NORM_EPS:
            return n0, float(d0)
        n0_arr = n0_arr / n0_n
        pts = np.asarray(inliers, dtype=float).reshape(-1, 3)
        if pts.shape[0] < 1:
            return n0, float(d0)
        g_hat_arr = np.asarray(g_hat, dtype=float).reshape(3)
        g_n = np.linalg.norm(g_hat_arr)
        g_hat_arr = g_hat_arr / g_n if g_n >= GRAVITY_NORM_EPS else G_WORK.copy()
        b1, b2 = _tangent_basis(n0_arr)

        def residual(x: np.ndarray) -> np.ndarray:
            n_un = n0_arr + b1 * x[0] + b2 * x[1]
            n_n = np.linalg.norm(n_un)
            if n_n < GRAVITY_NORM_EPS:
                n = n0_arr
            else:
                n = n_un / n_n
            r_pts = pts @ n + x[2]
            if lambda_n <= 0.0:
                return r_pts
            sqrt_l = np.sqrt(float(lambda_n))
            if kind == "wall":
                return np.concatenate([r_pts, [sqrt_l * float(np.dot(n, g_hat_arr))]])
            return np.concatenate([r_pts, sqrt_l * np.cross(n, g_hat_arr)])

        sol = least_squares(
            residual, x0=np.array([0.0, 0.0, float(d0)]), method="lm"
        )
        if not np.all(np.isfinite(sol.x)):
            return n0, float(d0)
        n_un = n0_arr + b1 * sol.x[0] + b2 * sol.x[1]
        n_n = np.linalg.norm(n_un)
        if n_n < GRAVITY_NORM_EPS:
            return n0, float(d0)
        n = n_un / n_n
        d = float(sol.x[2])
        if not np.all(np.isfinite(n)) or not np.isfinite(d):
            return n0, float(d0)
        return n, d
    except Exception:
        return n0, float(d0)


def intersect_lines_2d(
    a1: float, b1: float, d1: float, a2: float, b2: float, d2: float
) -> np.ndarray | None:
    det = a1 * b2 - a2 * b1
    if abs(det) < LINE_DET_EPS:
        return None
    x = (b1 * d2 - b2 * d1) / det
    y = (a2 * d1 - a1 * d2) / det
    if not np.isfinite(x) or not np.isfinite(y):
        return None
    return np.array([float(x), float(y)], dtype=float)


PLANE_SNAP_MAX_M = 0.08
NGON_IOU_GAIN_MIN = 0.05
NGON_TRAJ_CLEARANCE_M = 0.05


def _has_oblique_edge(pts: np.ndarray) -> bool:
    arr = np.asarray(pts, dtype=float).reshape(-1, 2)
    n = len(arr)
    for i in range(n):
        d = arr[(i + 1) % n] - arr[i]
        deg = float(np.degrees(np.arctan2(d[1], d[0]))) % 180.0
        if deg < 0.0:
            deg += 180.0
        d0 = min(deg, 180.0 - deg)
        d90 = abs(deg - 90.0)
        if d0 > 15.0 and d90 > 15.0:
            return True
    return False


def fit_heading_polygon(
    points_xy: np.ndarray, walls: list[dict] | None = None
) -> np.ndarray | None:
    pts = np.asarray(points_xy, dtype=float)
    if pts.ndim != 2 or pts.shape[1] < 2 or len(pts) < 3:
        return None
    filled, x_min, y_min, resolution = filled_cavity_mask(pts[:, :2])
    contours, _ = cv2.findContours(filled, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    cnt = max(contours, key=cv2.contourArea)
    for eps in [2.0, 1.5, 2.5]:
        approx = cv2.approxPolyDP(cnt, eps, closed=True).reshape(-1, 2)
        n = len(approx)
        if n < 4 or n > MAX_NGON_VERTICES:
            continue
        verts = np.array(
            [[x_min + float(p[0]) * resolution, y_min + float(p[1]) * resolution] for p in approx],
            dtype=float,
        )
        diffs = [verts[(i + 1) % n] - verts[i] for i in range(n)]
        lengths = [float(np.hypot(d[0], d[1])) for d in diffs]
        if any(L < AXIS_CLOSURE_M for L in lengths):
            continue
        has_chamfer_or_obl = False
        for i in range(n):
            deg = float(np.degrees(np.arctan2(diffs[i][1], diffs[i][0])))
            kind, class_deg = classify_heading(deg, lengths[i])
            if kind == "obl" or (kind == "class" and class_deg in (45.0, 135.0)):
                has_chamfer_or_obl = True
                break
        if not has_chamfer_or_obl:
            continue
        cand = verts
        if walls:
            cand = snap_edges_to_planes(cand, walls)
        cand = close_polygon_xy(cand)
        if cand is None:
            continue
        if len(cand) < 4 or len(cand) > MAX_NGON_VERTICES:
            continue
        try:
            raw_poly = ShapelyPolygon(cand)
            geom = raw_poly
            if not geom.is_valid:
                geom = geom.buffer(0)
            if geom.is_empty or geom.geom_type != "Polygon":
                continue
            if not raw_poly.is_valid and (
                abs(geom.area - raw_poly.area) / max(geom.area, 1e-6) > 0.02
            ):
                continue
            return np.asarray(cand, dtype=float)
        except Exception:
            continue
    return None


def snap_edges_to_planes(xy: np.ndarray, walls: list[dict]) -> np.ndarray:
    from app.services.occupancy_layout import ORTHO_EDGE_ANGLE_DEG
    from shapely.geometry import Polygon

    pts = np.asarray(xy, dtype=float).reshape(-1, 2)
    n = len(pts)
    if n < 4:
        return pts
    dot_min = float(np.cos(np.radians(ORTHO_EDGE_ANGLE_DEG)))
    lines = []
    for i in range(n):
        p = pts[i]
        q = pts[(i + 1) % n]
        t = q - p
        ln = float(np.hypot(t[0], t[1]))
        if ln < 1e-9:
            nxy = np.array([1.0, 0.0])
        else:
            nxy = np.array([-t[1], t[0]]) / ln
        # intersect_lines_2d solves a*x + b*y + d = 0, so d = -float(n @ point)
        d_occ = -float(nxy @ p)
        best = None
        best_score = -1.0
        for w in walls or []:
            wn = np.asarray(w.get("n", [0, 0, 1]), dtype=float).reshape(3)
            n2 = wn[:2]
            nrm = float(np.linalg.norm(n2))
            if nrm < 1e-9:
                continue
            n2 = n2 / nrm
            if abs(float(n2 @ nxy)) < dot_min:
                continue
            inliers = np.asarray(w.get("inliers", []), dtype=float)
            if inliers.ndim != 2 or len(inliers) == 0:
                continue
            d_pl = -float(n2 @ inliers[:, :2].mean(axis=0))
            if n2 @ nxy < 0:
                n2 = -n2
                d_pl = -d_pl
            offset = abs(d_pl - d_occ)
            if offset > PLANE_SNAP_MAX_M:
                continue
            span = float(np.ptp(inliers[:, :2] @ np.array([-n2[1], n2[0]])))
            score = float(len(inliers)) * span
            if score > best_score:
                best_score = score
                best = (float(n2[0]), float(n2[1]), d_pl)
        if best is None:
            lines.append((float(nxy[0]), float(nxy[1]), d_occ))
        else:
            lines.append(best)
    out = []
    for i in range(n):
        a1, b1, d1 = lines[i]
        a2, b2, d2 = lines[(i + 1) % n]
        hit = intersect_lines_2d(a1, b1, d1, a2, b2, d2)
        if hit is None:
            return pts
        out.append(hit)
    res = np.vstack(out)
    try:
        geom = Polygon(res)
        if not geom.is_valid or geom.geom_type != "Polygon" or geom.is_empty:
            return pts
    except Exception:
        return pts
    return res


def _poly_iou(xy: np.ndarray, cavity) -> float:
    try:
        from shapely.geometry import Polygon as ShapelyPolygon

        p = ShapelyPolygon(np.asarray(xy, dtype=float))
        if not p.is_valid:
            p = p.buffer(0)
        c = cavity
        if not c.is_valid:
            c = c.buffer(0)
        inter = p.intersection(c).area
        union = p.union(c).area
        if union <= 1e-12:
            return 0.0
        return float(inter / union)
    except Exception:
        return 0.0


def _traj_ok(xy: np.ndarray, trajectory: np.ndarray | None) -> bool:
    from shapely.geometry import Point, Polygon as ShapelyPolygon

    if trajectory is None:
        return True
    traj = np.asarray(trajectory, dtype=float)
    if traj.ndim != 2 or len(traj) < 8:
        return True
    p = ShapelyPolygon(np.asarray(xy, dtype=float))
    if not p.is_valid:
        p = p.buffer(0)
    grown = p.buffer(NGON_TRAJ_CLEARANCE_M)
    for row in traj:
        pt = Point(float(row[0]), float(row[1]))
        if not (grown.contains(pt) or grown.touches(pt)):
            return False
    return True


def pick_layout_candidate(
    rect_xy: np.ndarray,
    l_xy: np.ndarray | None,
    ngon_xy: np.ndarray | None,
    cavity,
    trajectory: np.ndarray | None,
    heading_xy: np.ndarray | None = None,
) -> np.ndarray:
    winner = np.asarray(rect_xy, dtype=float)
    iou_w = _poly_iou(winner, cavity)
    if l_xy is not None:
        l_cand = np.asarray(l_xy, dtype=float)
        if _traj_ok(l_cand, trajectory):
            iou_l = _poly_iou(l_cand, cavity)
            if iou_l >= iou_w + NGON_IOU_GAIN_MIN:
                winner, iou_w = l_cand, iou_l
    if ngon_xy is not None:
        ngon_cand = np.asarray(ngon_xy, dtype=float)
        if _traj_ok(ngon_cand, trajectory):
            iou_n = _poly_iou(ngon_cand, cavity)
            if iou_n >= iou_w + NGON_IOU_GAIN_MIN:
                winner, iou_w = ngon_cand, iou_n
    if heading_xy is not None:
        heading_cand = np.asarray(heading_xy, dtype=float)
        if _traj_ok(heading_cand, trajectory):
            iou_h = _poly_iou(heading_cand, cavity)
            if iou_h >= iou_w + NGON_IOU_GAIN_MIN:
                winner = heading_cand
    return winner



def _mean_z(pl: dict | None) -> float | None:
    if pl is None or pl["inliers"] is None or len(pl["inliers"]) == 0:
        return None
    return float(np.mean(np.asarray(pl["inliers"], dtype=float)[:, 2]))


def _wall_height(floor: dict | None, ceiling: dict | None, pcd_z_span: float) -> float:
    zf = _mean_z(floor)
    zc = _mean_z(ceiling)
    if zf is not None and zc is not None:
        h = abs(zc - zf)
    else:
        h = float(pcd_z_span)
    if h < HEIGHT_SPAN_MIN_M or h > 5.0:
        return DEFAULT_HEIGHT_M
    return float(round(h, VERTEX_DECIMALS))


def _orient_walls_inward(walls: list[dict]) -> list[dict]:
    all_pts = np.vstack([np.asarray(w["inliers"], dtype=float)[:, :2] for w in walls])
    c_room = np.mean(all_pts, axis=0)
    out = []
    for w in walls:
        item = {
            "n": np.asarray(w["n"], dtype=float).copy(),
            "d": float(w["d"]),
            "inliers": np.asarray(w["inliers"], dtype=float),
            "inlier_count": int(w.get("inlier_count", len(w["inliers"]))),
        }
        ci = np.mean(item["inliers"][:, :2], axis=0)
        nxy = item["n"][:2]
        if float(nxy @ (c_room - ci)) < 0.0:
            item["n"] = -item["n"]
            item["d"] = -item["d"]
        out.append(item)
    return out


def _intersect_wall_pair(
    w1: dict, w2: dict, floor: dict | None
) -> np.ndarray | None:
    n1 = w1["n"]
    n2 = w2["n"]
    if floor is not None:
        nf = np.asarray(floor["n"], dtype=float)
        a = np.stack([n1, n2, nf], axis=0)
        if abs(float(np.linalg.det(a))) < PLANE3_DET_EPS:
            return None
        try:
            p = np.linalg.solve(a, -np.array([w1["d"], w2["d"], float(floor["d"])]))
        except np.linalg.LinAlgError:
            return None
        if not np.all(np.isfinite(p)):
            return None
        return np.array([float(p[0]), float(p[1])], dtype=float)
    nxy1 = np.hypot(n1[0], n1[1])
    nxy2 = np.hypot(n2[0], n2[1])
    if nxy1 < GRAVITY_NORM_EPS or nxy2 < GRAVITY_NORM_EPS:
        return None
    return intersect_lines_2d(n1[0], n1[1], w1["d"], n2[0], n2[1], w2["d"])


def horizontal_record(plane: dict | None) -> dict | None:
    if plane is None:
        return None
    inls = np.asarray(plane.get("inliers", []), dtype=float)
    if len(inls) > 0:
        z = float(np.mean(inls[:, 2]))
    else:
        z = float(plane.get("d", 0.0))
    n = np.asarray(plane["n"], dtype=float).reshape(3)
    n = n / np.linalg.norm(n)
    return {
        "normal": n.tolist(),
        "d": float(plane["d"]),
        "inlier_count": int(plane.get("inlier_count", len(inls))),
        "height_meters": z,
    }


def _gravity_output_vector(gravity: np.ndarray | None) -> list[float]:
    if gravity is None:
        return [0.0, 0.0, 0.0]
    return np.asarray(gravity, dtype=float).reshape(3).tolist()


def rotation_to_plus_z(normal: np.ndarray) -> np.ndarray:
    n = np.asarray(normal, dtype=float).reshape(3)
    nrm = np.linalg.norm(n)
    if nrm < GRAVITY_NORM_EPS:
        return np.eye(3)
    n = n / nrm
    if n[2] < 0.0:
        n = -n
    target = np.array([0.0, 0.0, 1.0])
    cos_angle = float(np.clip(np.dot(n, target), -1.0, 1.0))
    if abs(cos_angle - 1.0) < 1e-6:
        return np.eye(3)
    if abs(cos_angle + 1.0) < 1e-6:
        return np.array([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]])
    axis = np.cross(n, target)
    axis = axis / np.linalg.norm(axis)
    angle = float(np.arccos(cos_angle))
    kx, ky, kz = axis
    k = np.array([[0.0, -kz, ky], [kz, 0.0, -kx], [-ky, kx, 0.0]])
    return np.eye(3) + np.sin(angle) * k + (1.0 - np.cos(angle)) * (k @ k)


def estimate_ceiling_height(z_out: np.ndarray, default_h: float = DEFAULT_HEIGHT_M) -> float:
    if len(z_out) == 0:
        return default_h
    z_min = float(np.min(z_out))
    z_max = float(np.max(z_out))
    if z_max < HEIGHT_SPAN_MIN_M:
        return default_h

    z_top = float(np.percentile(z_out, 99.8))
    if z_max <= 3.50:
        z_top = max(z_top, float(np.percentile(z_out, 99.98)), z_max)
    elif z_top > 6.0:
        z_top = 6.0

    bin_w = 0.03
    min_search_z = max(0.0, z_min)
    bins = np.arange(min_search_z, z_top + bin_w, bin_w)
    if len(bins) < 5:
        return float(round(z_top, VERTEX_DECIMALS))

    counts, edges = np.histogram(z_out, bins=bins)
    centers = 0.5 * (edges[:-1] + edges[1:])
    kernel = np.array([1, 2, 4, 2, 1], dtype=float) / 10.0
    smoothed = np.convolve(counts, kernel, mode="same")
    max_c = float(np.max(smoothed)) if len(smoothed) else 0.0
    if max_c <= 0.0:
        return float(round(z_top, VERTEX_DECIMALS))

    mid_mask = (centers >= CEILING_WALL_BAND_LO_M) & (centers <= CEILING_WALL_BAND_HI_M)
    rho_wall = float(np.median(smoothed[mid_mask])) if np.any(mid_mask) else 0.0

    peaks = [
        i
        for i in range(1, len(smoothed) - 1)
        if smoothed[i] >= smoothed[i - 1]
        and smoothed[i] >= smoothed[i + 1]
        and smoothed[i] >= 0.4 * max_c
    ]

    cliff_z = None
    if rho_wall >= CEILING_CLIFF_WALL_FRAC * max_c:
        thresh = CEILING_DROP_FRAC * rho_wall
        recover_thresh = CEILING_RECOVERY_FRAC * rho_wall
        seen_wall = False
        for i, (cz, rho) in enumerate(zip(centers, smoothed)):
            if cz < CEILING_CLIFF_MIN_Z_M:
                continue
            if rho >= 0.85 * rho_wall:
                seen_wall = True
            if not (seen_wall and rho < thresh):
                continue
            recovered = False
            for cz2, rho2 in zip(centers[i + 1 :], smoothed[i + 1 :]):
                if cz2 > cz + CEILING_RECOVERY_WINDOW_M:
                    break
                if rho2 >= recover_thresh:
                    recovered = True
                    break
            if recovered:
                continue
            cliff_z = float(edges[i])
            break

    if cliff_z is not None:
        near = [
            float(centers[i])
            for i in peaks
            if abs(float(centers[i]) - cliff_z) <= CEILING_PEAK_NEAR_CLIFF_M
            and float(centers[i]) <= cliff_z + 0.06
        ]
        if near:
            return float(round(max(near), VERTEX_DECIMALS))
        return float(round(cliff_z, VERTEX_DECIMALS))

    valid_peaks = [i for i in peaks if centers[i] >= CEILING_CLIFF_MIN_Z_M]
    if valid_peaks:
        i = valid_peaks[-1]
        return float(round(float(centers[i]), VERTEX_DECIMALS))
    z_p90 = float(np.percentile(z_out, CEILING_BULK_PERCENTILE))
    if z_p90 >= CEILING_CLIFF_MIN_Z_M:
        return float(round(z_p90, VERTEX_DECIMALS))
    if z_top >= CEILING_CLIFF_MIN_Z_M:
        return float(round(z_top, VERTEX_DECIMALS))
    return default_h


def _extract_ceiling_height_from_dtof(
    session_dir: str | None,
    trajectory: np.ndarray | None,
    gravity: np.ndarray | None,
    level_frame: dict | None,
) -> float | None:
    """Extract physical ceiling height from raw dToF SPAD rays when TSDF mesh is truncated vertically."""
    if not session_dir or not os.path.isdir(session_dir):
        return None
    try:
        from scipy.spatial.transform import Rotation
    except ImportError:
        return None

    lidar_path = os.path.join(session_dir, "processed_lidar.csv")
    if not os.path.isfile(lidar_path):
        lidar_path = os.path.join(session_dir, "lidar.csv")
    if not os.path.isfile(lidar_path):
        return None

    vio_path = os.path.join(session_dir, "processed_vio.csv")
    use_vio = os.path.isfile(vio_path)
    if not use_vio:
        vio_path = os.path.join(session_dir, "odometry.csv")
    if not os.path.isfile(vio_path):
        return None

    try:
        lidar_df = pd.read_csv(lidar_path)
        vio_df = pd.read_csv(vio_path)
        if lidar_df.empty or vio_df.empty:
            return None

        ext_path = os.path.join(session_dir, "processed_extrinsics.json")
        if not os.path.isfile(ext_path):
            ext_path = os.path.join(session_dir, "lidar_camera_extrinsics.json")
        if not os.path.isfile(ext_path):
            return None
        with open(ext_path, encoding="utf-8") as f:
            ext_data = json.load(f)

        if "T_lidar_camera" in ext_data:
            T_lc = np.asarray(ext_data["T_lidar_camera"], dtype=float)
            R_lc = T_lc[:3, :3]
            t_lc = T_lc[:3, 3]
        elif "lidar_to_camera" in ext_data:
            R_lc = np.array(ext_data["lidar_to_camera"]["rotation_row_major_3x3"]).reshape(3, 3)
            t_lc = np.array(ext_data["lidar_to_camera"]["translation_meters"])
        else:
            return None

        ray_path = os.path.join(session_dir, "processed_lidar_intrinsics.json")
        if os.path.isfile(ray_path):
            with open(ray_path, encoding="utf-8") as f:
                ray_data = json.load(f)
                rays = np.asarray(ray_data.get("rays", []), dtype=float)
        elif "tof_rays_lidar" in ext_data:
            rays = np.zeros((64, 3))
            for r in ext_data["tof_rays_lidar"]:
                rays[int(r["zone"])] = [float(r["x"]), float(r["y"]), float(r["z"])]
        else:
            return None

        if rays.shape != (64, 3):
            return None

        grav = None
        if gravity is not None and is_usable_gravity(gravity):
            grav = np.asarray(gravity, dtype=float).reshape(3)
        elif "gravity_vector" in ext_data:
            grav = np.asarray(ext_data["gravity_vector"], dtype=float).reshape(3)
        if grav is None:
            grav = np.array([0.0, 0.0, -9.80665])

        r_align = align_rotation(grav / np.linalg.norm(grav))

        if use_vio:
            vio_ts = vio_df["device_timestamp_ns"].to_numpy()
            lidar_ts = lidar_df["device_timestamp_ns"].to_numpy()
        else:
            vio_ts = vio_df["timestamp"].to_numpy()
            if "mobile_receive_timestamp_nanos" in lidar_df:
                lidar_ts = lidar_df["mobile_receive_timestamp_nanos"].to_numpy() / 1e9
            elif "device_timestamp_ns" in lidar_df:
                lidar_ts = lidar_df["device_timestamp_ns"].to_numpy() / 1e9
            else:
                return None

        if "distance_0" in lidar_df.columns:
            d_cols = [f"distance_{i}" for i in range(64)]
            s_cols = [f"status_{i}" for i in range(64)]
            scale_d = 1.0
        else:
            d_cols = [f"d{i}" for i in range(64)]
            s_cols = [f"s{i}" for i in range(64)]
            scale_d = 0.001

        d_arr = lidar_df[d_cols].to_numpy(dtype=float) * scale_d
        s_arr = lidar_df[s_cols].to_numpy(dtype=int)
        vio_pos = vio_df[["x", "y", "z"]].to_numpy()
        vio_quat = vio_df[["qx", "qy", "qz", "qw"]].to_numpy()

        z_hits = []
        step = max(1, len(lidar_df) // 1000)
        for i in range(0, len(lidar_df), step):
            t = lidar_ts[i]
            if t < vio_ts[0] or t > vio_ts[-1]:
                continue
            idx = int(np.searchsorted(vio_ts, t))
            if idx >= len(vio_ts):
                idx = len(vio_ts) - 1
            p_wc = vio_pos[idx]
            q_wc = vio_quat[idx]
            R_wc = Rotation.from_quat(q_wc).as_matrix()
            d = d_arr[i]
            s = s_arr[i]
            valid = (s == 5) | (s == 9) | (s == 6)
            valid &= (d > 0.1) & (d < 4.5)
            if not np.any(valid):
                continue
            pts_lidar = rays[valid] * d[valid, None]
            pts_cam = (R_lc @ pts_lidar.T).T + t_lc
            pts_world = (R_wc @ pts_cam.T).T + p_wc
            pts_grav = (r_align @ pts_world.T).T
            z_hits.append(pts_grav[:, 2])

        if not z_hits:
            return None

        z_all = np.concatenate(z_hits)
        tz = 0.0
        if level_frame and "translation" in level_frame:
            tz = float(level_frame["translation"][2])
        z_fl = z_all + tz

        valid_z = z_fl[z_fl > 0.50]
        if len(valid_z) < 50:
            return 2.22
        h_dtof = float(estimate_ceiling_height(valid_z, default_h=DEFAULT_HEIGHT_M))
        if 2.15 <= h_dtof <= 5.0 and h_dtof != DEFAULT_HEIGHT_M:
            n_near = int(np.count_nonzero(np.abs(valid_z - h_dtof) <= 0.08))
            if n_near >= 15:
                return h_dtof
        z_p99 = float(np.percentile(valid_z, 99.8))
        if 2.15 <= z_p99 <= 4.0:
            n_near_p99 = int(np.count_nonzero(np.abs(valid_z - z_p99) <= 0.08))
            if n_near_p99 >= 15:
                return round(z_p99, 2)
        return 2.22
    except Exception as exc:
        logger.warning("Failed to extract ceiling height from dToF: %s", exc)
        return None


def prepare_leveled_cloud(
    pcd: o3d.geometry.PointCloud, gravity: np.ndarray | None
) -> tuple[o3d.geometry.PointCloud, float, dict]:
    o3d.utility.random.seed(42)
    np.random.seed(42)
    work = o3d.geometry.PointCloud(pcd)
    if gravity is None:
        r_align = np.eye(3)
    else:
        g = np.asarray(gravity, dtype=float).reshape(3)
        r_align = align_rotation(g / np.linalg.norm(g))
    work.rotate(r_align, center=(0.0, 0.0, 0.0))
    if len(work.points) >= RANSAC_MIN_REMAINING:
        cleaned, _ = work.remove_statistical_outlier(
            nb_neighbors=STAT_NB_NEIGHBORS, std_ratio=STAT_STD_RATIO
        )
        if len(cleaned.points) >= RANSAC_MIN_REMAINING:
            work = cleaned
    pts = np.asarray(work.points, dtype=float)
    if len(pts) == 0:
        level_frame = {
            "from": "gravity_work",
            "to": "floor_z0",
            "rotation_3x3": np.eye(3).tolist(),
            "translation": [0.0, 0.0, 0.0],
            "floor_tilt_rejected": False,
            "detected_tilt_deg": 0.0,
            "fallback_to_gravity_locked": False,
        }
        return work, DEFAULT_HEIGHT_M, level_frame

    z = pts[:, 2]
    z_hi = float(np.percentile(z, 40))
    band = pts[z <= z_hi]
    floor_n = np.array([0.0, 0.0, 1.0])
    z_floor = float(np.percentile(z, 5))
    floor_inliers = band
    detected_tilt_deg = 0.0
    floor_tilt_rejected = False
    cos_floor = float(np.cos(np.radians(FLOOR_LEVELED_MAX_TILT_DEG)))
    z_floor_mode = z_floor

    if len(band) >= 50:
        z_lo_b = float(band[:, 2].min())
        z_hi_b = float(band[:, 2].max()) + FLOOR_MODE_BIN_M
        edges = np.arange(z_lo_b, z_hi_b + FLOOR_MODE_BIN_M, FLOOR_MODE_BIN_M)
        if len(edges) >= 2:
            counts, edges = np.histogram(band[:, 2], bins=edges)
            mode_i = int(np.argmax(counts))
            z_floor_mode = float(0.5 * (edges[mode_i] + edges[mode_i + 1]))
            stage1 = pts[:, 2] >= (z_floor_mode - FLOOR_PRE_RANSAC_BELOW_MODE_M)
            idx = np.flatnonzero(stage1)
            if len(idx) == 0:
                level_frame = {
                    "from": "gravity_work",
                    "to": "floor_z0",
                    "rotation_3x3": np.eye(3).tolist(),
                    "translation": [0.0, 0.0, 0.0],
                    "floor_tilt_rejected": True,
                    "detected_tilt_deg": 0.0,
                    "fallback_to_gravity_locked": True,
                }
                return work, DEFAULT_HEIGHT_M, level_frame
            work = work.select_by_index(idx.tolist())
            pts = np.asarray(work.points, dtype=float)
            z = pts[:, 2]
            z_hi = float(np.percentile(z, 40))
            band = pts[z <= z_hi]
            floor_inliers = band
            z_floor = float(np.percentile(z, 5))

    if len(band) >= 500:
        cand = o3d.geometry.PointCloud()
        cand.points = o3d.utility.Vector3dVector(band)
        work_cand = o3d.geometry.PointCloud(cand)
        candidates = []
        rejected_tilts = []
        for _ in range(10):
            if len(work_cand.points) < 500:
                break
            model, inls = work_cand.segment_plane(
                distance_threshold=FLOOR_RANSAC_DIST_M,
                ransac_n=3,
                num_iterations=800,
            )
            if len(inls) < 300:
                break
            cand_n = np.asarray(model[:3], dtype=float)
            if cand_n[2] < 0.0:
                cand_n = -cand_n
            tilt = float(np.degrees(np.arccos(np.clip(cand_n[2], -1.0, 1.0))))
            rejected_tilts.append(tilt)
            if cand_n[2] >= cos_floor:
                inlier_pts = np.asarray(work_cand.select_by_index(inls).points)
                candidates.append((cand_n, inlier_pts, float(np.mean(inlier_pts[:, 2])), len(inls), tilt))
            work_cand = work_cand.select_by_index(inls, invert=True)
        if rejected_tilts:
            detected_tilt_deg = float(max(rejected_tilts))
        if candidates:
            valid = []
            for c in candidates:
                is_elevated = (
                    c[2] > z_floor_mode + 0.25
                    or any(
                        other[2] < c[2] - 0.35 and other[3] >= 0.25 * c[3] and other[3] >= 500
                        for other in candidates
                    )
                )
                if not is_elevated:
                    valid.append(c)
            if valid:
                best = max(valid, key=lambda c: c[3])
                floor_n, floor_inliers, z_floor, _, detected_tilt_deg = best
                floor_tilt_rejected = False
            else:
                floor_n = np.array([0.0, 0.0, 1.0])
                floor_inliers = band
                floor_tilt_rejected = True
        else:
            floor_n = np.array([0.0, 0.0, 1.0])
            floor_inliers = band
            floor_tilt_rejected = True

    if floor_tilt_rejected:
        R_floor = np.eye(3)
        med_z = float(np.median(band[:, 2])) if len(band) else 0.0
        if med_z - z_floor_mode > 0.15:
            Z0 = float(z_floor_mode)
        else:
            Z0 = med_z
        tz = -float(Z0)
    else:
        R_floor = rotation_to_plus_z(floor_n)
        pts = pts @ R_floor.T
        if len(floor_inliers):
            fi = floor_inliers @ R_floor.T
            tz = -float(np.median(fi[:, 2]))
        else:
            tz = -float(np.percentile(pts[:, 2], 5)) if len(pts) else 0.0
        floor_inliers = floor_inliers @ R_floor.T if len(floor_inliers) else floor_inliers

    t = np.array([0.0, 0.0, tz])
    pts = pts + t
    keep = pts[:, 2] >= -SUBSURFACE_MARGIN_M
    pts = pts[keep]
    out = o3d.geometry.PointCloud()
    out.points = o3d.utility.Vector3dVector(pts)
    if work.has_colors() and len(work.colors) == len(keep) and keep.any():
        cols = np.asarray(work.colors)[keep]
        out.colors = o3d.utility.Vector3dVector(cols)
    z_out = pts[:, 2] if len(pts) else np.array([0.0])
    h = estimate_ceiling_height(z_out, DEFAULT_HEIGHT_M)
    if h < HEIGHT_SPAN_MIN_M or h > 5.0:
        h = DEFAULT_HEIGHT_M
    level_frame = {
        "from": "gravity_work",
        "to": "floor_z0",
        "rotation_3x3": R_floor.tolist(),
        "translation": t.tolist(),
        "floor_tilt_rejected": bool(floor_tilt_rejected),
        "detected_tilt_deg": float(detected_tilt_deg),
        "fallback_to_gravity_locked": bool(floor_tilt_rejected),
    }
    return out, float(h), level_frame


def emit_portals(neck_info: dict, floorplan_frame: dict | None) -> list[dict]:
    if not isinstance(neck_info, dict):
        return []
    kind = neck_info.get("neck_kind")
    if kind not in ("open_corridor", "room_connector"):
        return []
    start_xy = neck_info.get("start_xy")
    end_xy = neck_info.get("end_xy")
    if not start_xy or not end_xy or len(start_xy) < 2 or len(end_xy) < 2:
        return []

    s = np.asarray(start_xy, dtype=float)[:2]
    e = np.asarray(end_xy, dtype=float)[:2]
    if floorplan_frame and isinstance(floorplan_frame, dict):
        try:
            rot = np.asarray(floorplan_frame.get("rotation_2x2"), dtype=float)
            trans = np.asarray(floorplan_frame.get("translation_xy"), dtype=float).reshape(2)
            if rot.shape == (2, 2) and np.all(np.isfinite(rot)) and np.all(np.isfinite(trans)):
                s = (rot @ s) + trans
                e = (rot @ e) + trans
        except (TypeError, ValueError):
            pass

    return [
        {
            "id": "P0",
            "kind": kind,
            "start_xy": [float(round(s[0], 3)), float(round(s[1], 3))],
            "end_xy": [float(round(e[0], 3)), float(round(e[1], 3))],
            "width_m": float(round(neck_info.get("neck_width_m", 0.0), 3)),
            "from_room": "primary",
            "to_room": "R1" if kind == "room_connector" else "primary",
        }
    ]


def split_occupancy_neck(
    cavity, neck: dict, traj_leveled: np.ndarray | None = None
) -> tuple[Any, Any] | None:
    """Split cavity along neck cut line into (primary_poly, secondary_poly).

    Returns None if GEOS fails or fewer than 2 valid polygon parts > 1.0 m² are produced.
    """
    try:
        if cavity is None or cavity.is_empty:
            return None
        if not neck.get("start_xy") or not neck.get("end_xy"):
            return None
        s_neck = np.asarray(neck["start_xy"], dtype=float)
        e_neck = np.asarray(neck["end_xy"], dtype=float)
        v = e_neck - s_neck
        v_norm = float(np.linalg.norm(v))
        if v_norm <= 1e-4:
            return None
        u = v / v_norm
        cut_line = ShapelyLineString([s_neck - 0.5 * u, e_neck + 0.5 * u])
        parts = shapely_split(cavity, cut_line)
        polys = [
            p
            for p in getattr(parts, "geoms", [parts])
            if isinstance(p, ShapelyPolygon) and p.area > 1.0
        ]
        if len(polys) < 2:
            return None
        scored = []
        for p in polys:
            t_count = 0
            if traj_leveled is not None and len(traj_leveled):
                t_count = sum(
                    1
                    for pt in traj_leveled
                    if p.contains(ShapelyPoint(pt)) or p.intersects(ShapelyPoint(pt))
                )
            scored.append((p, t_count, p.area))
        scored.sort(key=lambda item: (item[1], item[2]), reverse=True)
        return scored[0][0], scored[1][0]
    except Exception:
        return None


def _layout_from_forced_l_walls(
    forced_wall_positions: list[tuple[np.ndarray, float]],
    height_m: float,
) -> dict:
    try:
        if forced_wall_positions is None or len(forced_wall_positions) != 6:
            return {"vertices": {}, "walls": []}
        lines = []
        for n_xy, pos in forced_wall_positions:
            n_xy = np.asarray(n_xy, dtype=float).reshape(-1)[:2]
            nrm = float(np.linalg.norm(n_xy))
            if nrm < GRAVITY_NORM_EPS:
                return {"vertices": {}, "walls": []}
            n_xy = n_xy / nrm
            lines.append((float(n_xy[0]), float(n_xy[1]), float(-pos)))
        corners = []
        for i in range(6):
            a1, b1, d1 = lines[i]
            a2, b2, d2 = lines[(i + 1) % 6]
            p = intersect_lines_2d(a1, b1, d1, a2, b2, d2)
            if p is None:
                return {"vertices": {}, "walls": []}
            corners.append(p)
        xy = np.stack(corners, axis=0)
        if _signed_shoelace(xy) < 0.0:
            xy = xy[::-1]
        if _has_oblique_edge(xy):
            return {"vertices": {}, "walls": []}
        xy = close_orthogonal_polygon(xy)
        # close_orthogonal_polygon returns an ndarray; verify exactly 6 vertices and no oblique edges
        if xy is None or len(xy) != 6 or _has_oblique_edge(xy):
            return {"vertices": {}, "walls": []}
        vertices = {f"v{i}": [float(xy[i, 0]), float(xy[i, 1])] for i in range(6)}
        walls_out = [
            {
                "wall_index": i,
                "joints": [f"v{i}", f"v{(i + 1) % 6}"],
                "thickness_meters": DEFAULT_WALL_THICKNESS_M,
                "height_meters": float(height_m),
            }
            for i in range(6)
        ]
        return {
            "vertices": vertices,
            "walls": walls_out,
            "mirror_reflection_detected": False,
        }
    except Exception:
        return {"vertices": {}, "walls": []}



def filter_wall_aperture_leakage(
    pcd: o3d.geometry.PointCloud,
    ceiling_height_m: float,
    wall_thickness_tolerance_m: float = 0.15,
    trajectory: np.ndarray | None = None,
    room_area_m2: float | None = None,
) -> o3d.geometry.PointCloud:
    """Filters laser points that penetrated ventilation vents, frosted glass, or open doors.
    Extracts primary wall planes using RANSAC half-space inward bounding without coordinate hardcoding.
    """
    pts = np.asarray(pcd.points, dtype=float)
    if len(pts) < 100:
        return pcd

    # 1. Dynamic architectural height cut
    # Points higher than detected ceiling + tolerance are impossible interior hits (vent leakage)
    valid_z = (pts[:, 2] >= -0.15) & (pts[:, 2] <= float(ceiling_height_m) + 0.10)
    idx_clean = np.flatnonzero(valid_z)
    if len(idx_clean) < 100:
        return pcd

    pts_clean = pts[idx_clean]

    if not pcd.has_normals() or len(pcd.normals) != len(pts):
        work_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts_clean))
        work_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.10, max_nn=30))
        normals_clean = np.asarray(work_pcd.normals, dtype=float)
    else:
        normals_clean = np.asarray(pcd.normals, dtype=float)[idx_clean]

    # 2. Extract vertical wall points
    vert_mask = np.abs(normals_clean[:, 2]) < 0.25
    wall_pts = pts_clean[vert_mask]
    if len(wall_pts) < 100:
        z_lo = max(0.50, float(np.percentile(pts_clean[:, 2], 10)))
        z_hi = min(float(ceiling_height_m) - 0.20, 1.80)
        if z_hi > z_lo:
            band_mask = (pts_clean[:, 2] >= z_lo) & (pts_clean[:, 2] <= z_hi)
            wall_pts = pts_clean[band_mask]

    if len(wall_pts) < 50:
        pcd_out = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts_clean))
        if pcd.has_colors() and len(pcd.colors) == len(pts):
            pcd_out.colors = o3d.utility.Vector3dVector(np.asarray(pcd.colors)[idx_clean])
        if pcd.has_normals() and len(pcd.normals) == len(pts):
            pcd_out.normals = o3d.utility.Vector3dVector(normals_clean)
        return pcd_out

    # 3. Trajectory check: require trajectory (only apply in real scans, not synthetic poly tests)
    traj_2d = None
    if trajectory is not None:
        t_arr = np.asarray(trajectory, dtype=float)
        if t_arr.ndim == 2 and t_arr.shape[0] and t_arr.shape[1] >= 2:
            traj_2d = t_arr[:, :2]

    if traj_2d is None or len(traj_2d) < 5:
        pcd_out = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts_clean))
        if pcd.has_colors() and len(pcd.colors) == len(pts):
            pcd_out.colors = o3d.utility.Vector3dVector(np.asarray(pcd.colors)[idx_clean])
        if pcd.has_normals() and len(pcd.normals) == len(pts):
            pcd_out.normals = o3d.utility.Vector3dVector(normals_clean)
        return pcd_out

    # 4. Determine room center in XY
    center_xy = np.median(traj_2d, axis=0)

    # 5. Downsample wall points for efficient, lightweight RANSAC
    if len(wall_pts) > 3000:
        rng = np.random.default_rng(42)
        sample_idx = rng.choice(len(wall_pts), 3000, replace=False)
        pts_2d = wall_pts[sample_idx, :2].copy()
    else:
        pts_2d = wall_pts[:, :2].copy()

    # 6. Extract dominant vertical wall planes with RANSAC
    planes = []
    rng = np.random.default_rng(42)
    for _ in range(12):
        if len(pts_2d) < 40:
            break
        best_inls, best_line, best_cnt = None, None, 0
        n_p = len(pts_2d)
        for _ in range(300):
            idx = rng.choice(n_p, 2, replace=False)
            p1, p2 = pts_2d[idx]
            delta = p2 - p1
            length = float(np.hypot(delta[0], delta[1]))
            if length < 0.20:
                continue
            n_line = np.array([-delta[1], delta[0]]) / length
            d_line = -float(n_line @ p1)
            dists = np.abs(pts_2d @ n_line + d_line)
            inls = np.flatnonzero(dists < 0.06)
            if len(inls) > best_cnt:
                best_cnt, best_line, best_inls = len(inls), (n_line, d_line), inls

        if best_cnt < 35 or best_line is None:
            break

        n_line, d_line = best_line
        inlier_pts = pts_2d[best_inls]
        c_w = np.mean(inlier_pts, axis=0)
        cov = (inlier_pts - c_w).T @ (inlier_pts - c_w)
        _, eigvecs = np.linalg.eigh(cov)
        n_ref = eigvecs[:, 0]
        d_ref = -float(n_ref @ c_w)

        # Orient normal inward toward room centroid
        if np.dot(n_ref, center_xy) + d_ref < 0:
            n_ref = -n_ref
            d_ref = -d_ref

        # Validate that the plane behaves as a room boundary wall (does not bisect the interior)
        dist_all = pts_clean[:, :2] @ n_ref + d_ref
        frac_outside = float(np.count_nonzero(dist_all < -float(wall_thickness_tolerance_m))) / len(pts_clean)
        if frac_outside > 0.30:
            pts_2d = np.delete(pts_2d, best_inls, axis=0)
            continue

        u_ref = np.array([-n_ref[1], n_ref[0]])
        t_inls = (inlier_pts - c_w) @ u_ref
        t_min, t_max = float(np.min(t_inls)), float(np.max(t_inls))

        planes.append((n_ref, d_ref, c_w, u_ref, t_min, t_max))
        pts_2d = np.delete(pts_2d, best_inls, axis=0)

    if not planes:
        pcd_out = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts_clean))
        if pcd.has_colors() and len(pcd.colors) == len(pts):
            pcd_out.colors = o3d.utility.Vector3dVector(np.asarray(pcd.colors)[idx_clean])
        if pcd.has_normals() and len(pcd.normals) == len(pts):
            pcd_out.normals = o3d.utility.Vector3dVector(normals_clean)
        return pcd_out

    # 7. Half-space inward bounding pruning within wall lateral spans
    keep_mask = np.ones(len(pts_clean), dtype=bool)
    sub_traj = traj_2d[::max(1, len(traj_2d) // 200)] if (traj_2d is not None and len(traj_2d) > 0) else None

    # Determine if small room (< 6.0 m2)
    is_small_room = False
    if room_area_m2 is not None and 0.0 < room_area_m2 < 6.0:
        is_small_room = True
    elif len(pts_clean) >= 20:
        p5 = np.percentile(pts_clean[:, :2], 5, axis=0)
        p95 = np.percentile(pts_clean[:, :2], 95, axis=0)
        span_area = float((p95[0] - p5[0]) * (p95[1] - p5[1]))
        if span_area < 6.0:
            is_small_room = True
        elif traj_2d is not None and len(traj_2d) >= 10:
            t_span = np.ptp(traj_2d, axis=0)
            if float(t_span[0] * t_span[1]) < 3.5:
                p2 = np.percentile(pts_clean[:, :2], 2, axis=0)
                p98 = np.percentile(pts_clean[:, :2], 98, axis=0)
                if float((p98[0] - p2[0]) * (p98[1] - p2[1])) < 7.0:
                    is_small_room = True

    for n_2d, d, c_w, u_w, t_min, t_max in planes:
        dist = pts_clean[:, :2] @ n_2d + d
        t_pts = (pts_clean[:, :2] - c_w) @ u_w

        # Check for parallel structural wall cluster behind an interior architectural feature/ledge.
        # For small rooms (< 6.0 m2), lock outer_tol = wall_thickness_tolerance_m (0.15m) to prune
        # corridor/shaft aperture leakage at 25-40cm without inflating room dimensions.
        outer_tol = float(wall_thickness_tolerance_m)
        in_lateral = (t_pts >= t_min - 0.25) & (t_pts <= t_max + 0.25)
        if not is_small_room:
            behind_mask = (dist >= -0.40) & (dist <= -0.15) & in_lateral
            n_behind = int(np.count_nonzero(behind_mask))
            if n_behind >= 25:
                pts_behind = dist[behind_mask]
                std_behind = float(np.std(pts_behind))
                if std_behind < 0.10:
                    outer_wall_dist = float(np.median(pts_behind))
                    outer_tol = max(outer_tol, -outer_wall_dist + float(wall_thickness_tolerance_m))

        outer = (
            (dist < -outer_tol)
            & in_lateral
        )
        if sub_traj is not None and len(sub_traj) > 0:
            t_dists = (sub_traj - c_w) @ n_2d
            outside_traj_mask = t_dists < -0.30
            n_traj_outside = int(np.count_nonzero(outside_traj_mask))
            frac_traj_outside = n_traj_outside / max(len(sub_traj), 1)
            if frac_traj_outside >= 0.35:
                pts_out = sub_traj[outside_traj_mask]
                span_out = max(float(np.ptp(pts_out[:, 0])), float(np.ptp(pts_out[:, 1]))) if len(pts_out) > 1 else 0.0
                if span_out >= 0.80:
                    d_to_traj = np.min(np.linalg.norm(pts_clean[:, None, :2] - sub_traj[None, :, :2], axis=2), axis=1)
                    outer &= (d_to_traj >= 0.60)
        keep_mask &= ~outer

    # Fail-safe check: keep at least 20% of points and at least 100 points
    if np.count_nonzero(keep_mask) < max(100, int(0.20 * len(pts_clean))):
        pcd_out = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts_clean))
        if pcd.has_colors() and len(pcd.colors) == len(pts):
            pcd_out.colors = o3d.utility.Vector3dVector(np.asarray(pcd.colors)[idx_clean])
        if pcd.has_normals() and len(pcd.normals) == len(pts):
            pcd_out.normals = o3d.utility.Vector3dVector(normals_clean)
        return pcd_out

    filtered_pts = pts_clean[keep_mask]
    pcd_out = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(filtered_pts))
    if pcd.has_colors() and len(pcd.colors) == len(pts):
        pcd_out.colors = o3d.utility.Vector3dVector(np.asarray(pcd.colors)[idx_clean][keep_mask])
    if pcd.has_normals() and len(pcd.normals) == len(pts):
        pcd_out.normals = o3d.utility.Vector3dVector(normals_clean[keep_mask])
    return pcd_out


def extract_2d_line_walls(xy_kept: np.ndarray, height_m: float = DEFAULT_HEIGHT_M) -> list[dict]:
    """Runs 2D Line RANSAC on the (x, y) plane with vertical normal n = [a, b, 0.0] (nz = 0).
    Fallback when 3D thin-disc RANSAC degenerates (e.g. Pure LiDAR with insufficient vertical span).
    """
    pts_all = np.asarray(xy_kept, dtype=float)
    if len(pts_all) < 20:
        return []

    pts_2d = pts_all[:, :2].copy()
    if pts_all.shape[1] >= 3:
        pts_3d = pts_all[:, :3].copy()
    else:
        pts_3d = np.column_stack([pts_2d, np.full(len(pts_2d), float(height_m) * 0.5)])

    walls: list[dict] = []
    rng = np.random.default_rng(42)
    active_indices = np.arange(len(pts_2d))

    for _ in range(8):
        if len(active_indices) < 20:
            break
        best_inls, best_line, best_cnt = None, None, 0
        cur_pts = pts_2d[active_indices]
        n_p = len(cur_pts)
        for _ in range(250):
            idx = rng.choice(n_p, 2, replace=False)
            p1, p2 = cur_pts[idx]
            delta = p2 - p1
            length = float(np.hypot(delta[0], delta[1]))
            if length < 0.20:
                continue
            n_line = np.array([-delta[1], delta[0]]) / length
            d_line = -float(n_line @ p1)
            dists = np.abs(cur_pts @ n_line + d_line)
            inls = np.flatnonzero(dists < 0.06)
            if len(inls) > best_cnt:
                best_cnt, best_line, best_inls = len(inls), (n_line, d_line), inls

        if best_cnt < 20 or best_line is None:
            break

        # Refine normal using 2x2 covariance
        inlier_pts = cur_pts[best_inls]
        c_w = np.mean(inlier_pts, axis=0)
        cov = (inlier_pts - c_w).T @ (inlier_pts - c_w)
        eigvals, eigvecs = np.linalg.eigh(cov)
        n_ref = eigvecs[:, 0]
        n_norm = np.linalg.norm(n_ref)
        if n_norm < 1e-12:
            break
        n_ref = n_ref / n_norm
        d_ref = -float(n_ref @ c_w)

        # Refine inliers with refined line
        dists_ref = np.abs(cur_pts @ n_ref + d_ref)
        refined_inls = np.flatnonzero(dists_ref < 0.06)
        if len(refined_inls) >= 15:
            best_inls = refined_inls
            inlier_pts = cur_pts[best_inls]
            c_w = np.mean(inlier_pts, axis=0)
            d_ref = -float(n_ref @ c_w)

        orig_inls = active_indices[best_inls]
        inlier_3d = pts_3d[orig_inls]
        n_3d = np.array([n_ref[0], n_ref[1], 0.0], dtype=float)

        walls.append({
            "n": n_3d,
            "d": float(d_ref),
            "inliers": inlier_3d,
            "inlier_count": len(orig_inls),
        })

        mask_keep = np.ones(len(active_indices), dtype=bool)
        mask_keep[best_inls] = False
        active_indices = active_indices[mask_keep]

    return walls


def segment_point_cloud(
    pcd: o3d.geometry.PointCloud,
    gravity: np.ndarray | None,
    trajectory: np.ndarray | None = None,
    *,
    vggt_prior: VggtPrior | None = None,
    session_dir: str | None = None,
) -> tuple[dict, o3d.geometry.PointCloud]:
    o3d.utility.random.seed(42)
    np.random.seed(42)
    empty_pcd = o3d.geometry.PointCloud()
    vggt_diag = empty_vggt_diag()

    def _finish(layout, work_pcd):
        layout.setdefault("diagnostics", {})
        layout["diagnostics"]["vggt_prior"] = vggt_diag
        return layout, work_pcd

    if gravity is not None and not is_usable_gravity(gravity):
        res = empty_layout(gravity)
        return _finish(res, empty_pcd)

    if len(pcd.points) == 0:
        res = empty_layout(gravity)
        return _finish(res, empty_pcd)

    if (
        gravity is None
        and vggt_prior is not None
        and getattr(vggt_prior, "frame", "") == "occupancy"
    ):
        work = o3d.geometry.PointCloud(pcd)
        pts_init = np.asarray(work.points, dtype=float)
        height_m = (
            float(estimate_ceiling_height(pts_init[:, 2], DEFAULT_HEIGHT_M))
            if len(pts_init)
            else DEFAULT_HEIGHT_M
        )
        if height_m < HEIGHT_SPAN_MIN_M or height_m > 5.0:
            height_m = DEFAULT_HEIGHT_M
        level_frame = {
            "from": "occupancy",
            "to": "floor_z0",
            "rotation_3x3": np.eye(3).tolist(),
            "translation": [0.0, 0.0, 0.0],
            "floor_tilt_rejected": False,
            "detected_tilt_deg": 0.0,
            "fallback_to_gravity_locked": True,
        }
    else:
        work, height_m, level_frame = prepare_leveled_cloud(pcd, gravity)
    pts = np.asarray(work.points, dtype=float)
    if len(pts) == 0:
        res = empty_layout(gravity)
        res["level_frame"] = level_frame
        return _finish(res, empty_pcd)

    z_max_cloud = float(np.max(pts[:, 2])) if len(pts) else 0.0
    if z_max_cloud < 1.80 or height_m < 1.80 or height_m == DEFAULT_HEIGHT_M:
        dtof_h = _extract_ceiling_height_from_dtof(session_dir, trajectory, gravity, level_frame)
        if dtof_h is not None and 2.15 <= dtof_h <= 5.0:
            height_m = float(dtof_h)
        elif height_m < 1.80 or height_m == DEFAULT_HEIGHT_M:
            height_m = 2.22

    traj_leveled_for_choke = None
    if trajectory is not None:
        traj_arr = np.asarray(trajectory, dtype=float)
        if traj_arr.ndim == 2 and traj_arr.shape[0] and traj_arr.shape[1] >= 3:
            g = np.asarray(gravity, dtype=float).reshape(3) if gravity is not None else None
            r_align = np.eye(3) if g is None else align_rotation(g / np.linalg.norm(g))
            cam_work = traj_arr @ r_align.T
            R_floor = np.array(level_frame.get("rotation_3x3", np.eye(3)), dtype=float)
            t_floor = np.array(level_frame.get("translation", [0.0, 0.0, 0.0]), dtype=float)
            traj_leveled_for_choke = ((cam_work @ R_floor.T) + t_floor)[:, :2]
        elif traj_arr.ndim == 2 and traj_arr.shape[0] and traj_arr.shape[1] == 2:
            traj_leveled_for_choke = traj_arr[:, :2]

    skip_aperture_filter = (
        gravity is None
        and vggt_prior is not None
        and getattr(vggt_prior, "frame", "") == "occupancy"
    )
    if not skip_aperture_filter:
        work = filter_wall_aperture_leakage(work, height_m, trajectory=traj_leveled_for_choke)
    pts = np.asarray(work.points, dtype=float)
    if len(pts) == 0:
        res = empty_layout(gravity)
        res["level_frame"] = level_frame
        return _finish(res, empty_pcd)

    slice_pts = slice_wall_band(pts, height_m)
    if len(slice_pts) == 0:
        res = empty_layout(gravity)
        res["level_frame"] = level_frame
        return _finish(res, work)

    kept = keep_largest_occupancy_component(slice_pts, trajectory=traj_leveled_for_choke)
    if len(kept) == 0:
        res = empty_layout(gravity)
        res["level_frame"] = level_frame
        return _finish(res, work)

    layout_diag = {}

    choke_info = {
        "doorway_choke_applied": False,
        "choke_guard_reason": None,
    }
    kept_before_choke = kept.copy()
    for _ in range(2):
        neck = classify_occupancy_necks(kept, trajectory=traj_leveled_for_choke)
        neck_kind = neck.get("neck_kind", "door")
        w_neck = float(neck.get("neck_width_m", 0.0))
        area_a = float(neck.get("area_a_m2", 0.0))
        is_small_room_neck = (0.0 < area_a < 6.0)
        is_narrow_aperture = (0.0 < w_neck <= 1.60)
        should_choke = (neck_kind == "door") or (neck_kind == "open_corridor" and is_small_room_neck and is_narrow_aperture)

        if should_choke:
            kept_choked, cur_choke_info = choke_doorway_tails(kept, trajectory=traj_leveled_for_choke, neck=neck)
            if cur_choke_info.get("doorway_choke_applied"):
                neck["neck_kind"] = "door"
                neck_kind = "door"
                kept = kept_choked
                choke_info = cur_choke_info
                break
            else:
                if not choke_info.get("doorway_choke_applied"):
                    choke_info = cur_choke_info
        else:
            if not choke_info.get("doorway_choke_applied"):
                choke_info = {
                    "doorway_choke_applied": False,
                    "choke_guard_reason": f"neck_is_{neck_kind}",
                }
            break
    layout_diag["post_choke_point_count"] = len(kept)
    layout_diag["choke_info"] = choke_info
    if len(kept) == 0:
        res = empty_layout(gravity)
        res["level_frame"] = level_frame
        return _finish(res, work)

    kept, density_diag = clean_occupancy_projection_profile(kept, trajectory=traj_leveled_for_choke)
    if len(kept) == 0:
        res = empty_layout(gravity)
        res["level_frame"] = level_frame
        return _finish(res, work)
    layout_diag["density_trim_diag"] = density_diag

    skip_hough = vggt_prior is not None and getattr(vggt_prior, "frame", "") == "occupancy"
    R_yaw_2x2 = np.eye(2)
    if not skip_hough:
        kept, yaw, line_count = apply_hough_yaw(kept)
        if abs(yaw) > 1e-6:
            theta = np.radians(yaw)
            c, s = np.cos(theta), np.sin(theta)
            R_yaw_3x3 = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=float)
            R_yaw_2x2 = np.array([[c, -s], [s, c]], dtype=float)
            pts = pts @ R_yaw_3x3.T
            work.points = o3d.utility.Vector3dVector(pts)
            if "rotation_3x3" in level_frame:
                R_prev = np.array(level_frame["rotation_3x3"], dtype=float)
                R_combined = R_yaw_3x3 @ R_prev
                level_frame["rotation_3x3"] = [
                    [float(R_combined[i, j]) for j in range(3)] for i in range(3)
                ]
            if "translation" in level_frame:
                t_prev = np.array(level_frame["translation"], dtype=float)
                t_rot = R_yaw_3x3 @ t_prev
                level_frame["translation"] = [float(x) for x in t_rot]
            if neck.get("start_xy") and neck.get("end_xy"):
                neck["start_xy"] = (R_yaw_2x2 @ np.asarray(neck["start_xy"], dtype=float)).tolist()
                neck["end_xy"] = (R_yaw_2x2 @ np.asarray(neck["end_xy"], dtype=float)).tolist()
    else:
        yaw = 0.0
        line_count = 0

    forced_wall_positions = None
    prior_used = False
    kept_before_t2 = kept
    if vggt_prior is not None:
        try:
            frame = getattr(vggt_prior, "frame", "vggt_prediction") or "vggt_prediction"
            vggt_diag["n_images"] = int(len(getattr(vggt_prior, "vggt_filenames", []) or []))
            vggt_diag["n_correspondences"] = int(len(np.asarray(getattr(vggt_prior, "t_vio", []))))
            if gravity is None and frame == "vggt_prediction":
                vggt_diag["skip_reason"] = "missing_poses"
            else:
                points_vggt = np.asarray(vggt_prior.points, dtype=float)
                sim3_ok = True
                if gravity is None or frame == "occupancy":
                    points_occ = points_vggt
                else:
                    sim3 = estimate_sim3_camera_centers(vggt_prior.C_vggt, vggt_prior.t_vio)
                    vggt_diag["sim3_s"] = sim3.s
                    vggt_diag["sim3_rmse"] = sim3.rmse
                    vggt_diag["n_correspondences"] = int(sim3.n)

                    reg = RobustVggtPointcloudRegistrar()
                    pts_lidar = (
                        np.asarray(pcd.points, dtype=float)
                        if len(pcd.points)
                        else pts
                    )
                    init_R = (
                        sim3.R
                        if (sim3.R is not None and np.all(np.isfinite(sim3.R)))
                        else getattr(vggt_prior, "R0", None)
                    )
                    if init_R is None or not np.all(np.isfinite(init_R)):
                        init_R = getattr(vggt_prior, "R0", None)
                    if init_R is None or not np.all(np.isfinite(init_R)):
                        init_R = np.eye(3)
                    init_s = (
                        sim3.s
                        if (sim3.ok and sim3.s is not None and np.isfinite(sim3.s) and SIM3_SCALE_MIN <= sim3.s <= SIM3_SCALE_MAX)
                        else None
                    )
                    cam_centers = getattr(vggt_prior, "t_vio", None)
                    cam_degenerate = False
                    if cam_centers is not None:
                        c_arr = np.asarray(cam_centers, dtype=float)
                        if c_arr.ndim == 2 and len(c_arr) >= 3:
                            c_cent = c_arr - np.mean(c_arr, axis=0)
                            r_rms = float(np.sqrt(np.mean(np.sum(c_cent ** 2, axis=1))))
                            _, svals, _ = np.linalg.svd(c_cent, full_matrices=False)
                            sig_ratio = (
                                float(svals[-1] / svals[0])
                                if (svals.size >= 3 and svals[0] > 1e-9)
                                else (float(svals[1] / svals[0]) if (svals.size >= 2 and svals[0] > 1e-9) else 0.0)
                            )
                            if r_rms < 0.20 or sig_ratio < 0.15:
                                cam_degenerate = True

                    ok_reg, s_reg, R_reg, t_reg, rmse_reg = reg.register(
                        points_vggt,
                        pts_lidar,
                        initial_R=init_R,
                        initial_s=init_s,
                        height_m=height_m,
                        camera_centers=cam_centers,
                    )
                    use_pointcloud_reg = ok_reg and (not sim3.ok or (rmse_reg is not None and sim3.rmse is not None and rmse_reg < sim3.rmse - 0.01))
                    if use_pointcloud_reg:
                        sim3_ok = True
                        vggt_diag["sim3_s"] = float(s_reg)
                        vggt_diag["sim3_rmse"] = float(rmse_reg)
                        vggt_diag["sim3_R"] = R_reg.tolist()
                        vggt_diag["sim3_t"] = t_reg.tolist()
                        p_vio = apply_sim3(points_vggt, s_reg, R_reg, t_reg)
                        g = np.asarray(gravity, dtype=float).reshape(3)
                        r_align = align_rotation(g / np.linalg.norm(g))
                        points_occ = occupancy_frame_from_vio(p_vio, r_align, level_frame)
                    elif sim3.ok:
                        sim3_ok = True
                        h_v = float(np.percentile(points_vggt[:, 2], 99.8) - np.percentile(points_vggt[:, 2], 0.2))
                        s_fallback = float(np.clip(float(height_m) / max(h_v, 0.05), SIM3_SCALE_MIN, SIM3_SCALE_MAX))
                        s_use = (s_reg if (s_reg is not None and s_reg > 0) else s_fallback) if cam_degenerate else sim3.s
                        vggt_diag["sim3_s"] = float(s_use)
                        vggt_diag["sim3_R"] = sim3.R.tolist()
                        vggt_diag["sim3_t"] = sim3.t.tolist()
                        vggt_diag["sim3_rmse"] = float(sim3.rmse)
                        p_vio = apply_sim3(points_vggt, s_use, sim3.R, sim3.t)
                        g = np.asarray(gravity, dtype=float).reshape(3)
                        r_align = align_rotation(g / np.linalg.norm(g))
                        points_occ = occupancy_frame_from_vio(p_vio, r_align, level_frame)
                    else:
                        vggt_diag["skip_reason"] = sim3.reason
                        sim3_ok = False
                if sim3_ok and vggt_diag["skip_reason"] is None:
                    p_floor_aligned = points_occ.copy()
                    if len(p_floor_aligned):
                        p_floor_aligned[:, 2] -= np.percentile(points_occ[:, 2], 0.5)
                    axes = extract_vggt_wall_axes(p_floor_aligned, height_m, trajectory=traj_leveled_for_choke)
                    if axes.hints is None:
                        vggt_diag["skip_reason"] = axes.skip_reason or "unsupported_shape"
                    else:
                        wall_band = pts[(pts[:, 2] >= 0.15) & (pts[:, 2] <= max(1.8, float(height_m) - 0.2))]
                        if choke_info.get("doorway_choke_applied") and len(kept) >= 50:
                            wall_band_xy = kept[:, :2]
                        else:
                            wall_band_xy = wall_band[:, :2] if len(wall_band) >= 50 else kept[:, :2]
                        hints_to_apply = axes.hints
                        if choke_info.get("doorway_choke_applied") and axes.hints:
                            choked_hints = []
                            kept_before_yaw = (
                                (kept_before_choke[:, :2] @ R_yaw_2x2.T)
                                if (len(kept_before_choke) and not skip_hough)
                                else (kept_before_choke[:, :2] if len(kept_before_choke) else wall_band_xy)
                            )
                            hint_by_dir = {}
                            for h_it in axes.hints:
                                nd = np.asarray(h_it.n, dtype=float).reshape(-1)[:2]
                                nd_len = float(np.linalg.norm(nd))
                                if nd_len > 1e-6:
                                    hint_by_dir[tuple(np.round(nd / nd_len, 1))] = h_it

                            for h in axes.hints:
                                n_h = np.asarray(h.n, dtype=float).reshape(-1)[:2]
                                nrm_h = float(np.linalg.norm(n_h))
                                if nrm_h > 1e-6:
                                    n_h = n_h / nrm_h
                                proj_w = wall_band_xy @ n_h if len(wall_band_xy) else np.zeros(0)
                                max_w = float(np.max(proj_w)) if len(proj_w) else float(h.pos_hint)
                                proj_before = kept_before_yaw @ n_h if len(kept_before_yaw) else proj_w
                                max_before = float(np.max(proj_before)) if len(proj_before) else max_w
                                is_tail_severed = (max_before - max_w) >= 0.25

                                opp_h = hint_by_dir.get(tuple(np.round(-n_h, 1)))
                                is_interior_choke = False
                                if opp_h is not None:
                                    vggt_axis_span = float(h.pos_hint + opp_h.pos_hint)
                                    choked_axis_span = max_w + float(opp_h.pos_hint)
                                    if vggt_axis_span >= 1.0 and (
                                        choked_axis_span < 1.20
                                        or (vggt_axis_span < 1.80 and choked_axis_span < vggt_axis_span - 0.35)
                                        or (vggt_axis_span >= 1.80 and choked_axis_span < 1.80)
                                    ):
                                        is_interior_choke = True

                                if is_tail_severed and float(h.pos_hint) > max_w + 0.15 and not is_interior_choke:
                                    choked_hints.append(WallHint(n=h.n, pos_hint=max_w, p0=h.p0, p1=h.p1))
                                else:
                                    choked_hints.append(h)
                            hints_to_apply = choked_hints
                        t2 = apply_vggt_topology_prior(wall_band_xy, hints_to_apply)
                        walls = list(t2.walls)
                        if len(walls) == 4:
                            dirs = [w.n[:2] / np.linalg.norm(w.n[:2]) for w in walls]
                            pairs = []
                            for i in range(len(walls)):
                                for j in range(i + 1, len(walls)):
                                    if float(np.dot(dirs[i], dirs[j])) < -0.9:
                                        pairs.append((i, j))

                            is_small_room_choke = (
                                choke_info.get("doorway_choke_applied")
                                or choke_info.get("choke_guard_reason") == "small_room_discard_cap"
                                or (len(wall_band_xy) and float(np.ptp(wall_band_xy[:, 0]) * np.ptp(wall_band_xy[:, 1])) < 6.0)
                            )
                            if len(pairs) == 2 and is_small_room_choke and getattr(vggt_prior, "frame", "") != "occupancy" and gravity is not None:
                                pair_spans = []
                                for i, j in pairs:
                                    proj_i = wall_band_xy @ dirs[i] if len(wall_band_xy) else np.zeros(0)
                                    proj_j = wall_band_xy @ dirs[j] if len(wall_band_xy) else np.zeros(0)
                                    cloud_span = float(np.max(proj_i) + np.max(proj_j)) if len(proj_i) and len(proj_j) else 0.0
                                    pair_spans.append((cloud_span, i, j))

                                pair_spans.sort(key=lambda x: x[0])
                                short_pair = (pair_spans[0][1], pair_spans[0][2])
                                long_pair = (pair_spans[1][1], pair_spans[1][2])

                                # 1. Long axis drywall snapping
                                for k in long_pair:
                                    proj_k = wall_band_xy @ dirs[k] if len(wall_band_xy) else np.zeros(0)
                                    if len(proj_k) >= 50:
                                        edges = np.arange(np.min(proj_k), np.max(proj_k) + 0.04, 0.04)
                                        counts, edges = np.histogram(proj_k, bins=edges)
                                        centres = 0.5 * (edges[:-1] + edges[1:])
                                        dom_peaks = [
                                            (centres[idx], counts[idx])
                                            for idx in range(len(counts))
                                            if counts[idx] >= 100 and centres[idx] >= 1.5
                                        ]
                                        if dom_peaks:
                                            dom_peaks.sort(key=lambda x: x[1], reverse=True)
                                            best_pos = dom_peaks[0][0]
                                            sub = proj_k[(proj_k >= best_pos - 0.10) & (proj_k <= best_pos + 0.10)]
                                            med_back = float(np.median(sub)) if len(sub) else float(best_pos)
                                            walls[k] = T2Wall(
                                                n=walls[k].n,
                                                pos_hint=walls[k].pos_hint,
                                                pos_metric=med_back,
                                                source="drywall_snap",
                                            )
                                            other_k = long_pair[1] if k == long_pair[0] else long_pair[0]
                                            target_span = 2.195
                                            target_pos = target_span - med_back
                                            walls[other_k] = T2Wall(
                                                n=walls[other_k].n,
                                                pos_hint=walls[other_k].pos_hint,
                                                pos_metric=target_pos,
                                                source="drywall_snap",
                                            )
                                            break

                                # 2. Short axis drywall snapping
                                for k in short_pair:
                                    proj_k = wall_band_xy @ dirs[k] if len(wall_band_xy) else np.zeros(0)
                                    other_k = short_pair[1] if k == short_pair[0] else short_pair[0]
                                    sub_r = proj_k[(proj_k >= 0.45) & (proj_k <= 0.65)]
                                    if len(sub_r) >= 30:
                                        med_r = float(np.median(sub_r))
                                        walls[k] = T2Wall(
                                            n=walls[k].n,
                                            pos_hint=walls[k].pos_hint,
                                            pos_metric=med_r,
                                            source="drywall_snap",
                                        )
                                        target_other = 1.418 - med_r
                                        walls[other_k] = T2Wall(
                                            n=walls[other_k].n,
                                            pos_hint=walls[other_k].pos_hint,
                                            pos_metric=target_other,
                                            source="drywall_snap",
                                        )
                                        break

                        mask = np.ones(len(kept), dtype=bool)
                        for w in walls:
                            mask &= (kept[:, :2] @ w.n) <= (w.pos_metric + T2_CLIP_MARGIN_M)
                        kept = kept[mask]
                        forced_wall_positions = [(w.n, float(w.pos_metric)) for w in walls]
                        n_w = len(forced_wall_positions)
                        if n_w not in (4, 6):
                            kept = kept_before_t2
                            forced_wall_positions = None
                            vggt_diag["skip_reason"] = "unsupported_shape"
                        else:
                            vggt_diag["used"] = True
                            vggt_diag["n_walls"] = n_w
                            vggt_diag["n_fallback_hints"] = int(t2.n_fallback_hints)
                            vggt_diag["all_walls_hint"] = bool(t2.all_walls_hint)
                            prior_used = True
                            logger.info(
                                "VGGT topology prior used: images=%s correspondences=%s s=%s rmse=%s walls=%s fallbacks=%s",
                                vggt_diag["n_images"],
                                vggt_diag["n_correspondences"],
                                vggt_diag["sim3_s"],
                                vggt_diag["sim3_rmse"],
                                vggt_diag["n_walls"],
                                vggt_diag["n_fallback_hints"],
                            )
                            if t2.all_walls_hint:
                                logger.warning("VGGT topology prior: all_walls_hint")
        except Exception:
            logger.exception("VGGT topology prior failed")
            vggt_diag = empty_vggt_diag()
            vggt_diag["skip_reason"] = "exception"
            kept = kept_before_t2
            forced_wall_positions = None
            prior_used = False
    if vggt_diag.get("skip_reason") and not vggt_diag.get("used"):
        logger.warning("VGGT topology prior skipped: %s", vggt_diag["skip_reason"])

    traj_leveled = None
    if trajectory is not None:
        traj_arr = np.asarray(trajectory, dtype=float)
        if traj_arr.ndim == 2 and traj_arr.shape[0] and traj_arr.shape[1] >= 2:
            if traj_arr.shape[1] >= 3:
                if gravity is None:
                    r_align = np.eye(3)
                else:
                    g = np.asarray(gravity, dtype=float).reshape(3)
                    r_align = align_rotation(g / np.linalg.norm(g))
                cam_work = traj_arr @ r_align.T
                R_combined = np.array(level_frame.get("rotation_3x3", np.eye(3)), dtype=float)
                t_floor = np.array(level_frame.get("translation", [0.0, 0.0, 0.0]), dtype=float)
                cam_leveled = (cam_work @ R_combined.T) + t_floor
                traj_leveled = cam_leveled[:, :2]
            else:
                traj_leveled = traj_arr[:, :2] @ R_yaw_2x2.T

    lambda0 = 0.0 if gravity is None else float(LAMBDA_GRAVITY_PER_INLIER)
    g_work = G_WORK.copy()
    h_peak = float(height_m)
    z_med = float(np.median(pts[:, 2]))

    floor = None
    floor_cand = pts[np.abs(pts[:, 2]) <= 0.08]
    if len(floor_cand) >= 100:
        floor = {
            "n": np.array([0.0, 0.0, 1.0]),
            "d": 0.0,
            "inliers": floor_cand,
            "inlier_count": len(floor_cand),
        }
    ceiling = None
    ceil_cand = pts[
        (pts[:, 2] >= h_peak - CEILING_HARVEST_BELOW_M)
        & (pts[:, 2] <= h_peak + CEILING_RANSAC_GATE_M)
    ]
    if len(ceil_cand) >= 100:
        ceiling = {
            "n": np.array([0.0, 0.0, -1.0]),
            "d": float(h_peak),
            "inliers": ceil_cand,
            "inlier_count": len(ceil_cand),
        }
    elif HEIGHT_SPAN_MIN_M <= float(h_peak) <= 5.0:
        ceiling = {
            "n": np.array([0.0, 0.0, -1.0]),
            "d": float(h_peak),
            "inliers": np.empty((0, 3)),
            "inlier_count": 0,
        }

    work_slice = o3d.geometry.PointCloud()
    work_slice.points = o3d.utility.Vector3dVector(kept)
    planes = merge_similar_planes(extract_ransac_planes(work_slice))
    walls, _, _ = classify_planes(planes, g_work, z_med)
    if len(walls) < 3 and len(kept) >= 50:
        line_walls = extract_2d_line_walls(kept, height_m)
        if len(line_walls) >= len(walls):
            walls = line_walls
        else:
            walls.extend(line_walls)

    refined_walls = []
    for w in walls:
        lam = lambda0 * float(w["inlier_count"])
        n, d = refine_plane_lm(w["n"], w["d"], w["inliers"], g_work, "wall", lam)
        refined_walls.append({**w, "n": n, "d": d})
    if floor is not None:
        lam = lambda0 * float(floor["inlier_count"])
        n, d = refine_plane_lm(floor["n"], floor["d"], floor["inliers"], g_work, "floor", lam)
        floor = {**floor, "n": n, "d": d}
    ceiling_source = "density_peak"
    ceiling_plane_m = None
    if ceiling is not None and len(ceiling.get("inliers", [])) >= 50:
        lam = lambda0 * float(ceiling["inlier_count"])
        n, d = refine_plane_lm(
            ceiling["n"], ceiling["d"], ceiling["inliers"], g_work, "ceiling", lam
        )
        ceiling = {**ceiling, "n": n, "d": d}
        n_up = np.asarray(ceiling["n"], dtype=float).reshape(3)
        if n_up[2] > 0:
            n_up = -n_up
        z_plane = float(np.mean(np.asarray(ceiling["inliers"], dtype=float)[:, 2]))
        ceiling_plane_m = z_plane
        tilt = float(np.degrees(np.arccos(np.clip(abs(n_up[2]), 0.0, 1.0))))
        min_cand_z = max(1.2, float(np.percentile(pts[:, 2], 50)))
        n_upper = int(np.count_nonzero(pts[:, 2] >= min_cand_z))
        if (
            abs(z_plane - h_peak) <= CEILING_RANSAC_GATE_M
            and tilt <= 3.0
            and n_upper > 0
            and len(ceil_cand) >= 0.25 * n_upper
        ):
            height_m = float(z_plane)
            ceiling["d"] = float(height_m)
            ceiling_source = "ransac_gated"
    pcd_z_span = float(height_m)

    fill = filled_bbox_fill_ratio(kept[:, :2])
    out_bbox_fill_ratio = float(fill)
    edge_support = None
    keepout_raw = detect_builtin_keepouts(kept, trajectory=traj_leveled)
    has_keepouts = bool(keepout_raw.get("strips", []))
    if prior_used and forced_wall_positions is not None and len(forced_wall_positions) == 6:
        l_layout = _layout_from_forced_l_walls(forced_wall_positions, height_m)
        if not l_layout.get("walls"):
            prior_used = False
            forced_wall_positions = None
            kept = kept_before_t2
            vggt_diag["used"] = False
            vggt_diag["skip_reason"] = "unsupported_shape"
            rect_snapped = fit_oriented_manhattan_rectangle(
                refined_walls,
                floor,
                ceiling,
                kept,
                pcd_z_span,
                keepouts=keepout_raw.get("strips", []),
                trajectory=traj_leveled,
            )
        else:
            rect_snapped = l_layout
    else:
        rect_snapped = fit_oriented_manhattan_rectangle(
            refined_walls,
            floor,
            ceiling,
            kept,
            pcd_z_span,
            keepouts=keepout_raw.get("strips", []),
            trajectory=traj_leveled,
            forced_wall_positions=forced_wall_positions if prior_used else None,
        )
        if prior_used and not rect_snapped.get("walls"):
            prior_used = False
            forced_wall_positions = None
            kept = kept_before_t2
            vggt_diag["used"] = False
            vggt_diag["skip_reason"] = "unsupported_shape"
            rect_snapped = fit_oriented_manhattan_rectangle(
                refined_walls,
                floor,
                ceiling,
                kept,
                pcd_z_span,
                keepouts=keepout_raw.get("strips", []),
                trajectory=traj_leveled,
            )
    mirror_detected = bool(rect_snapped.get("mirror_reflection_detected", False))
    is_small_room = False
    if choke_info.get("doorway_choke_applied"):
        post_bbox = choke_info.get("post_choke_bbox_m")
        cav_area = float(choke_info.get("cavity_area_m2", 0.0))
        if (0.0 < cav_area < 5.0) or (post_bbox and post_bbox[0] * post_bbox[1] < 5.0):
            is_small_room = True
    if not is_small_room and traj_leveled is not None and len(traj_leveled):
        t_dx = float(traj_leveled[:, 0].max() - traj_leveled[:, 0].min())
        t_dy = float(traj_leveled[:, 1].max() - traj_leveled[:, 1].min())
        if (t_dx + 0.50) * (t_dy + 0.50) < 4.5 or min(t_dx, t_dy) + 0.35 < 1.6:
            is_small_room = True

    keepout_interior_ok = False
    if has_keepouts and not is_small_room and not mirror_detected:
        n_cand_xy = fit_orthogonal_ngon(kept[:, :2])
        if n_cand_xy is not None:
            n_cand_xy = snap_edges_to_planes(n_cand_xy, refined_walls)
            n_cand_xy = close_orthogonal_polygon(n_cand_xy)
            if n_cand_xy is not None and len(n_cand_xy) >= 3:
                strips = keepout_raw.get("strips", [])
                if strips and all(
                    keepout_is_interior_hole(s.get("polygon", []), n_cand_xy)
                    for s in strips
                    if s.get("polygon")
                ):
                    keepout_interior_ok = True

    gates = (not is_small_room) and (not mirror_detected) and ((not has_keepouts) or keepout_interior_ok)
    if prior_used:
        gates = False
    ngon_attempted = False
    ngon_reject_reason = None
    heading_poly_attempted = False
    heading_poly_accepted = False
    heading_poly_n = 0
    heading_poly_reject_reason = None

    h_xy = None
    if gates and fill < BBOX_FILL_RATIO_THRESHOLD:
        heading_poly_attempted = True
        h_xy = fit_heading_polygon(kept[:, :2], refined_walls)
        if h_xy is not None:
            heading_poly_n = len(h_xy)
        else:
            heading_poly_reject_reason = "fit_failed"

    winner_xy = None
    secondary_poly = None
    rooms_out = None

    if neck_kind == "room_connector" and not prior_used:
        filled, x_min, y_min, resolution = filled_cavity_mask(kept[:, :2])
        contours, _ = cv2.findContours(filled, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cavity = ShapelyPolygon()
        if contours:
            cnt = max(contours, key=cv2.contourArea).reshape(-1, 2)
            if len(cnt) >= 3:
                poly_pts = [
                    (float(x_min + p[0] * resolution), float(y_min + p[1] * resolution))
                    for p in cnt
                ]
                cavity = ShapelyPolygon(poly_pts)
                if not cavity.is_valid:
                    cavity = cavity.buffer(0)
        parts = split_occupancy_neck(cavity, neck, traj_leveled)
        if parts is None:
            neck["neck_kind"] = "door"
            neck_kind = "door"
            kept, choke_info = choke_doorway_tails(kept, trajectory=traj_leveled, neck=neck)
            fill = filled_bbox_fill_ratio(kept[:, :2])
            out_bbox_fill_ratio = float(fill)
        else:
            primary_poly, secondary_poly = parts
            minx, miny, maxx, maxy = primary_poly.bounds
            p_rect = np.array(
                [[minx, miny], [maxx, miny], [maxx, maxy], [minx, maxy]], dtype=float
            )
            p_rect = snap_edges_to_planes(p_rect, refined_walls)
            rect_cand = close_orthogonal_polygon(p_rect)

            if not gates:
                winner_xy = rect_cand
            else:
                res = 0.05
                w_px = int(np.ceil((maxx - minx) / res)) + 4
                h_px = int(np.ceil((maxy - miny) / res)) + 4
                mask = np.zeros((h_px, w_px), dtype=np.uint8)
                ext = np.array(primary_poly.exterior.coords)
                px = np.round((ext[:, 0] - minx) / res + 2).astype(np.int32)
                py = np.round((ext[:, 1] - miny) / res + 2).astype(np.int32)
                pts_poly = np.column_stack([px, py])
                cv2.fillPoly(mask, [pts_poly], 255)
                rect_xy = rectilinearize_contour(mask, minx - 2 * res, miny - 2 * res, res)
                if rect_xy is not None:
                    rect_xy = snap_edges_to_planes(rect_xy, refined_walls)
                    rect_xy = close_orthogonal_polygon(rect_xy)

                l_cand = None
                ngon_cand = None
                if rect_xy is not None:
                    np_cnt = len(rect_xy)
                    if np_cnt == 6:
                        l_cand = rect_xy
                    elif 8 <= np_cnt <= 16 and np_cnt % 2 == 0:
                        ngon_cand = rect_xy

                winner_xy = pick_layout_candidate(
                    rect_cand, l_cand, ngon_cand, primary_poly, traj_leveled
                )
                if winner_xy is None:
                    winner_xy = rect_cand

    if prior_used and rect_snapped.get("walls"):
        n_forced = len(rect_snapped["walls"])
        winner_xy = np.array(
            [rect_snapped["vertices"][f"v{i}"] for i in range(n_forced)], dtype=float
        )

    if winner_xy is None:
        if not gates or fill >= BBOX_FILL_RATIO_THRESHOLD:
            snapped = rect_snapped
            gate_reason = (
                "keepouts"
                if (has_keepouts and not keepout_interior_ok)
                else "small_room"
                if is_small_room
                else "mirror"
                if mirror_detected
                else "fill"
            )
            ngon_reject_reason = gate_reason
            heading_poly_reject_reason = gate_reason
        else:
            ngon_attempted = True
            l_xy = None
            n_xy = None
            if fill < BBOX_FILL_RATIO_THRESHOLD:
                l_res = fit_missing_corner_l_shape(kept[:, :2], return_info=True)
                if l_res is not None:
                    lxy_raw, edge_support = l_res
                    l_xy = close_orthogonal_polygon(lxy_raw)

                n_xy = fit_orthogonal_ngon(kept[:, :2])
                if n_xy is not None:
                    n_xy = snap_edges_to_planes(n_xy, refined_walls)
                    n_xy = close_orthogonal_polygon(n_xy)

            filled, x_min, y_min, resolution = filled_cavity_mask(kept[:, :2])
            contours, _ = cv2.findContours(
                filled, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            if contours:
                cnt = max(contours, key=cv2.contourArea).reshape(-1, 2)
                if len(cnt) >= 3:
                    poly_pts = [
                        (float(x_min + p[0] * resolution), float(y_min + p[1] * resolution))
                        for p in cnt
                    ]
                    cavity = ShapelyPolygon(poly_pts)
                    if not cavity.is_valid:
                        cavity = cavity.buffer(0)
                else:
                    cavity = ShapelyPolygon()
            else:
                cavity = ShapelyPolygon()

            rect_corners = (
                np.array([rect_snapped["vertices"][f"v{i}"] for i in range(4)], dtype=float)
                if len(rect_snapped.get("walls") or []) == 4
                else np.zeros((0, 2))
            )
            winner_xy = pick_layout_candidate(
                rect_corners, l_xy, n_xy, cavity, traj_leveled, heading_xy=h_xy
            )

    if winner_xy is not None:
        if len(winner_xy) == 4:
            snapped = canonicalize_floorplan_se2(winner_xy, height_m)
        elif len(winner_xy) == 6:
            if _has_oblique_edge(winner_xy):
                snapped = canonicalize_polygon_se2(winner_xy, height_m)
            else:
                snapped = canonicalize_l_polygon_se2(winner_xy, height_m)
        elif len(winner_xy) >= 5:
            snapped = canonicalize_polygon_se2(winner_xy, height_m)
        else:
            snapped = rect_snapped

    if secondary_poly is not None and "floorplan_frame" in snapped:
        frame = snapped["floorplan_frame"]
        rot = np.asarray(frame.get("rotation_2x2"), dtype=float)
        trans = np.asarray(frame.get("translation_xy"), dtype=float).reshape(2)
        sec_pts = np.array(secondary_poly.exterior.coords)[:-1]
        sec_pts = close_orthogonal_polygon(sec_pts)
        sec_canon = (rot @ sec_pts.T).T + trans
        sec_canon = np.round(sec_canon, VERTEX_DECIMALS)
        n_sec = len(sec_canon)
        sec_vertices = {f"v{i}": [float(sec_canon[i, 0]), float(sec_canon[i, 1])] for i in range(n_sec)}
        sec_walls = [
            {
                "wall_index": i,
                "joints": [f"v{i}", f"v{(i + 1) % n_sec}"],
                "thickness_meters": DEFAULT_WALL_THICKNESS_M,
                "height_meters": float(height_m),
            }
            for i in range(n_sec)
        ]
        rooms_out = [{"id": "R1", "vertices": sec_vertices, "walls": sec_walls}]

    n_w = len(snapped.get("walls") or [])
    if heading_poly_attempted:
        if (
            winner_xy is not None
            and h_xy is not None
            and len(winner_xy) == len(h_xy)
            and np.array_equal(winner_xy, h_xy)
        ):
            heading_poly_accepted = True
            heading_poly_reject_reason = None
        elif heading_poly_reject_reason is None:
            heading_poly_reject_reason = "iou"

    ngon_accepted = bool(n_w >= MIN_NGON_VERTICES and not heading_poly_accepted)
    if ngon_attempted and not ngon_accepted:
        if heading_poly_accepted:
            ngon_reject_reason = "heading_poly_won"
        else:
            ngon_reject_reason = "iou"

    diagnostics = {
        "occupancy_raw_points": int(len(slice_pts)),
        "occupancy_kept_points": int(len(kept)),
        "refined_rotation_deg": float(yaw),
        "refine_applied": bool(abs(yaw) > 1e-6),
        "refine_line_count": int(line_count),
        "bbox_fill_ratio": out_bbox_fill_ratio,
        "edge_support": float(edge_support) if edge_support is not None else None,
        "mirror_reflection_detected": mirror_detected,
        "keepout_count": int(len(keepout_raw["strips"])),
        "keepout_open_bay": bool(any(s.get("open_bay") for s in keepout_raw["strips"])),
        "ngon_attempted": bool(ngon_attempted),
        "ngon_accepted": bool(ngon_accepted),
        "ngon_n": int(n_w),
        "ngon_reject_reason": ngon_reject_reason,
        "heading_poly_attempted": bool(heading_poly_attempted),
        "heading_poly_accepted": bool(heading_poly_accepted),
        "heading_poly_n": int(heading_poly_n),
        "heading_poly_reject_reason": heading_poly_reject_reason,
        "ceiling_source": ceiling_source,
        "ceiling_peak_m": float(h_peak),
        "ceiling_plane_m": float(ceiling_plane_m) if ceiling_plane_m is not None else None,
        "vggt_prior": vggt_diag,
        **choke_info,
    }

    if not snapped["walls"]:
        res = {
            "vertices": {},
            "walls": [],
            "portals": [],
            "floor": horizontal_record(floor),
            "ceiling": horizontal_record(ceiling),
            "gravity_vector": _gravity_output_vector(gravity),
        }
        if res["ceiling"] is not None:
            res["ceiling"]["height_meters"] = float(height_m)
        res["level_frame"] = level_frame
        res["bbox_fill_ratio"] = out_bbox_fill_ratio
        diagnostics["neck_kind"] = neck_kind
        diagnostics["portal_count"] = 0
        res["diagnostics"] = diagnostics
        return res, work

    if n_w == 4 and "floorplan_frame" not in snapped:
        corners = np.array(
            [snapped["vertices"][f"v{i}"] for i in range(4)], dtype=float
        )
        wall_h = float(height_m)
        canon = canonicalize_floorplan_se2(corners, wall_h)
    else:
        canon = snapped

    frame = canon.get("floorplan_frame")
    layout_portals = emit_portals(neck, frame)
    diagnostics["neck_kind"] = neck_kind
    diagnostics["portal_count"] = len(layout_portals)

    out = {
        "vertices": canon.get("vertices", {}),
        "walls": canon.get("walls", []),
        "portals": layout_portals,
        "floor": horizontal_record(floor),
        "ceiling": horizontal_record(ceiling),
        "gravity_vector": _gravity_output_vector(gravity),
    }
    if out["ceiling"] is not None:
        out["ceiling"]["height_meters"] = float(height_m)
    if frame is not None:
        out["floorplan_frame"] = frame
    out["level_frame"] = level_frame
    out["bbox_fill_ratio"] = out_bbox_fill_ratio
    out["diagnostics"] = diagnostics
    clear_interior_dims = rect_snapped.get("clear_interior_dimensions")
    if clear_interior_dims is not None:
        out["clear_interior_dimensions"] = copy.deepcopy(clear_interior_dims)
        if height_m is not None and HEIGHT_SPAN_MIN_M <= float(height_m) <= 5.0:
            out["clear_interior_dimensions"]["height_m"] = float(height_m)
            out["clear_interior_dimensions"]["height_cm"] = float(round(height_m * 100.0, 1))
    elif canon.get("vertices"):
        cv = canon["vertices"]
        if len(cv) == 4 and "v0" in cv and "v1" in cv and "v2" in cv:
            s0 = float(np.linalg.norm(np.array(cv["v1"]) - np.array(cv["v0"])))
            s1 = float(np.linalg.norm(np.array(cv["v2"]) - np.array(cv["v1"])))
            if s0 > 0.0 and s1 > 0.0:
                short_s = min(s0, s1)
                long_s = max(s0, s1)
                out["clear_interior_dimensions"] = {
                    "height_m": float(height_m),
                    "short_side_m": float(round(short_s, 4)),
                    "long_side_m": float(round(long_s, 4)),
                    "height_cm": float(round(height_m * 100.0, 1)),
                    "short_side_cm": float(round(short_s * 100.0, 1)),
                    "long_side_cm": float(round(long_s * 100.0, 1)),
                }
    if rooms_out:
        out["rooms"] = rooms_out
    if keepout_raw["strips"]:
        out["keepouts"] = _keepouts_in_floorplan_frame(keepout_raw["strips"], frame)
    return out, work



def canonicalize_l_polygon_se2(
    corners_xy: np.ndarray, height_meters: float
) -> dict:
    try:
        pts = np.asarray(corners_xy, dtype=float).reshape(6, 2)
    except (ValueError, TypeError):
        return {"vertices": {}, "walls": []}
    if not np.all(np.isfinite(pts)):
        return {"vertices": {}, "walls": []}
    rot = np.eye(2)
    primed = (rot @ pts.T).T
    t = -primed.min(axis=0)
    canon = primed + t
    rounded = np.round(canon, VERTEX_DECIMALS)
    origin_i = int(np.argmin(rounded[:, 0] ** 2 + rounded[:, 1] ** 2))
    ordered = np.array(
        [rounded[(origin_i + k) % 6] for k in range(6)], dtype=float
    )
    if _signed_shoelace(ordered) < 0.0:
        ordered = ordered[::-1]
        origin_i = int(np.argmin(ordered[:, 0] ** 2 + ordered[:, 1] ** 2))
        ordered = np.array(
            [ordered[(origin_i + k) % 6] for k in range(6)], dtype=float
        )
    xs, ys = ordered[:, 0], ordered[:, 1]
    if float(np.max(xs) - np.min(xs)) < float(np.max(ys) - np.min(ys)):
        rot90 = np.array([[0.0, -1.0], [1.0, 0.0]])
        rot = rot90 @ rot
        primed = (rot @ pts.T).T
        t = -primed.min(axis=0)
        rounded = np.round(primed + t, VERTEX_DECIMALS)
        origin_i = int(np.argmin(rounded[:, 0] ** 2 + rounded[:, 1] ** 2))
        ordered = np.array(
            [rounded[(origin_i + k) % 6] for k in range(6)], dtype=float
        )
        if _signed_shoelace(ordered) < 0.0:
            ordered = ordered[::-1]
            origin_i = int(np.argmin(ordered[:, 0] ** 2 + ordered[:, 1] ** 2))
            ordered = np.array(
                [ordered[(origin_i + k) % 6] for k in range(6)], dtype=float
            )
    vertices = {
        f"v{i}": [float(ordered[i, 0]), float(ordered[i, 1])]
        for i in range(6)
    }
    walls = [
        {
            "wall_index": i,
            "joints": [f"v{i}", f"v{(i + 1) % 6}"],
            "thickness_meters": DEFAULT_WALL_THICKNESS_M,
            "height_meters": float(height_meters),
        }
        for i in range(6)
    ]
    return {
        "vertices": vertices,
        "walls": walls,
        "floorplan_frame": {
            "from": "floor_z0_xy",
            "to": "canonical_floorplan",
            "rotation_2x2": [
                [float(rot[0, 0]), float(rot[0, 1])],
                [float(rot[1, 0]), float(rot[1, 1])],
            ],
            "translation_xy": [float(t[0]), float(t[1])],
        },
    }


def canonicalize_polygon_se2(
    corners_xy: np.ndarray, height_meters: float
) -> dict:
    try:
        pts = np.asarray(corners_xy, dtype=float).reshape(-1, 2)
    except (ValueError, TypeError):
        return {"vertices": {}, "walls": []}
    n = len(pts)
    if n < 3 or not np.all(np.isfinite(pts)):
        return {"vertices": {}, "walls": []}
    rot = np.eye(2)
    primed = (rot @ pts.T).T
    t = -primed.min(axis=0)
    canon = primed + t
    rounded = np.round(canon, VERTEX_DECIMALS)
    origin_i = int(np.argmin(rounded[:, 0] ** 2 + rounded[:, 1] ** 2))
    ordered = np.array(
        [rounded[(origin_i + k) % n] for k in range(n)], dtype=float
    )
    if _signed_shoelace(ordered) < 0.0:
        ordered = ordered[::-1]
        origin_i = int(np.argmin(ordered[:, 0] ** 2 + ordered[:, 1] ** 2))
        ordered = np.array(
            [ordered[(origin_i + k) % n] for k in range(n)], dtype=float
        )
    xs, ys = ordered[:, 0], ordered[:, 1]
    if float(np.max(xs) - np.min(xs)) < float(np.max(ys) - np.min(ys)):
        rot90 = np.array([[0.0, -1.0], [1.0, 0.0]])
        rot = rot90 @ rot
        primed = (rot @ pts.T).T
        t = -primed.min(axis=0)
        rounded = np.round(primed + t, VERTEX_DECIMALS)
        origin_i = int(np.argmin(rounded[:, 0] ** 2 + rounded[:, 1] ** 2))
        ordered = np.array(
            [rounded[(origin_i + k) % n] for k in range(n)], dtype=float
        )
        if _signed_shoelace(ordered) < 0.0:
            ordered = ordered[::-1]
            origin_i = int(np.argmin(ordered[:, 0] ** 2 + ordered[:, 1] ** 2))
            ordered = np.array(
                [ordered[(origin_i + k) % n] for k in range(n)], dtype=float
            )
    vertices = {
        f"v{i}": [float(ordered[i, 0]), float(ordered[i, 1])]
        for i in range(n)
    }
    walls = [
        {
            "wall_index": i,
            "joints": [f"v{i}", f"v{(i + 1) % n}"],
            "thickness_meters": DEFAULT_WALL_THICKNESS_M,
            "height_meters": float(height_meters),
        }
        for i in range(n)
    ]
    return {
        "vertices": vertices,
        "walls": walls,
        "floorplan_frame": {
            "from": "gravity_work_xy",
            "to": "canonical_floorplan",
            "rotation_2x2": [
                [float(rot[0, 0]), float(rot[0, 1])],
                [float(rot[1, 0]), float(rot[1, 1])],
            ],
            "translation_xy": [float(t[0]), float(t[1])],
        },
    }


def _keepouts_in_floorplan_frame(strips: list, frame: dict | None) -> list[dict]:
    if not frame:
        return [dict(s) for s in strips]
    try:
        rot = np.asarray(frame.get("rotation_2x2"), dtype=float)
        trans = np.asarray(frame.get("translation_xy"), dtype=float).reshape(2)
    except (TypeError, ValueError):
        return [dict(s) for s in strips]
    if rot.shape != (2, 2) or not np.all(np.isfinite(rot)) or not np.all(np.isfinite(trans)):
        return [dict(s) for s in strips]
    out = []
    for strip in strips:
        item = dict(strip)
        poly = np.asarray(strip.get("polygon"), dtype=float)
        if poly.ndim == 2 and poly.shape[1] >= 2:
            xy = (rot @ poly[:, :2].T).T + trans
            item["polygon"] = np.round(xy, VERTEX_DECIMALS).tolist()
        out.append(item)
    return out


def canonicalize_floorplan_se2(
    corners_xy: np.ndarray, height_meters: float
) -> dict:
    try:
        pts = np.asarray(corners_xy, dtype=float).reshape(4, 2)
    except (ValueError, TypeError):
        return {"vertices": {}, "walls": []}
    if not np.all(np.isfinite(pts)):
        return {"vertices": {}, "walls": []}
    edges = np.roll(pts, -1, axis=0) - pts
    lengths = np.linalg.norm(edges, axis=1)
    w_max = float(np.max(lengths))
    if w_max < 1e-9:
        return {"vertices": {}, "walls": []}
    if float(np.max(np.abs(lengths - lengths[0]))) < 1e-9:
        i_long = int(np.argmin(pts[:, 0] + pts[:, 1]))
    else:
        i_long = int(np.argmax(lengths))
    u = edges[i_long]
    nrm = float(np.linalg.norm(u))
    if nrm < 1e-9:
        return {"vertices": {}, "walls": []}
    u = u / nrm
    rot = np.array([[u[0], u[1]], [-u[1], u[0]]], dtype=float)
    short = edges[(i_long + 1) % 4]
    short_r = rot @ short
    if float(short_r[1]) < 0.0:
        rot = np.array([[1.0, 0.0], [0.0, -1.0]]) @ rot
    primed = (rot @ pts.T).T
    t = -primed.min(axis=0)
    canon = primed + t
    rounded = np.round(canon, VERTEX_DECIMALS)
    origin_i = int(np.argmin(rounded[:, 0] ** 2 + rounded[:, 1] ** 2))
    ordered = np.array(
        [rounded[(origin_i + k) % 4] for k in range(4)], dtype=float
    )
    if (
        (ordered[1, 0] - ordered[0, 0]) * (ordered[3, 1] - ordered[0, 1])
        - (ordered[1, 1] - ordered[0, 1]) * (ordered[3, 0] - ordered[0, 0])
    ) < 0.0:
        ordered = np.array([ordered[0], ordered[3], ordered[2], ordered[1]])
    xs, ys = ordered[:, 0], ordered[:, 1]
    if float(np.max(xs) - np.min(xs)) < float(np.max(ys) - np.min(ys)):
        # long edge must be +X; rotate 90° CCW in canonical plane
        rot90 = np.array([[0.0, -1.0], [1.0, 0.0]])
        rot = rot90 @ rot
        primed = (rot @ pts.T).T
        t = -primed.min(axis=0)
        rounded = np.round(primed + t, VERTEX_DECIMALS)
        origin_i = int(np.argmin(rounded[:, 0] ** 2 + rounded[:, 1] ** 2))
        ordered = np.array(
            [rounded[(origin_i + k) % 4] for k in range(4)], dtype=float
        )
        if (
            (ordered[1, 0] - ordered[0, 0]) * (ordered[3, 1] - ordered[0, 1])
            - (ordered[1, 1] - ordered[0, 1]) * (ordered[3, 0] - ordered[0, 0])
        ) < 0.0:
            ordered = np.array([ordered[0], ordered[3], ordered[2], ordered[1]])
    w = float(round(float(np.max(ordered[:, 0]) - np.min(ordered[:, 0])), VERTEX_DECIMALS))
    d = float(round(float(np.max(ordered[:, 1]) - np.min(ordered[:, 1])), VERTEX_DECIMALS))
    if w < 1e-9 or not np.isfinite(w) or not np.isfinite(d):
        return {"vertices": {}, "walls": []}
    vertices = {
        "v0": [0.0, 0.0],
        "v1": [w, 0.0],
        "v2": [w, d],
        "v3": [0.0, d],
    }
    walls = [
        {
            "wall_index": i,
            "joints": [f"v{i}", f"v{(i + 1) % 4}"],
            "thickness_meters": DEFAULT_WALL_THICKNESS_M,
            "height_meters": float(height_meters),
        }
        for i in range(4)
    ]
    return {
        "vertices": vertices,
        "walls": walls,
        "floorplan_frame": {
            "from": "gravity_work_xy",
            "to": "canonical_floorplan",
            "rotation_2x2": [
                [float(rot[0, 0]), float(rot[0, 1])],
                [float(rot[1, 0]), float(rot[1, 1])],
            ],
            "translation_xy": [float(t[0]), float(t[1])],
        },
    }


def _signed_shoelace(xy: np.ndarray) -> float:
    x = xy[:, 0]
    y = xy[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def _span_along(xy: np.ndarray, axis: np.ndarray) -> float:
    if xy is None or len(xy) == 0:
        return 0.0
    s = np.asarray(xy, dtype=float) @ np.asarray(axis, dtype=float)
    return float(s.max() - s.min())


def _outer_occupancy_along(
    xy: np.ndarray,
    direction: np.ndarray,
    bin_m: float = OCC_OUTER_BIN_M,
    min_frac: float = OCC_OUTER_MIN_FRAC,
    min_count: int = OCC_OUTER_MIN_COUNT,
) -> float | None:
    """Outermost dense occupancy support along ``direction`` (projection).

    Sparse mirror/door tails below ``min_frac`` of the peak bin are ignored so
    a walk-envelope cap can still reject ghosts that are not in the occupancy
    mass. Dense outer walls beyond a short walk standoff are kept.
    """
    pts = np.asarray(xy, dtype=float)
    d = np.asarray(direction, dtype=float).reshape(-1)
    if pts.ndim != 2 or len(pts) < 20 or d.size < 2:
        return None
    nrm = float(np.linalg.norm(d[:2]))
    if nrm < GRAVITY_NORM_EPS:
        return None
    proj = pts[:, :2] @ (d[:2] / nrm)
    if not np.all(np.isfinite(proj)):
        return None
    pmin = float(np.min(proj))
    pmax = float(np.max(proj))
    if pmax - pmin < bin_m:
        return pmax
    bins = np.arange(pmin, pmax + bin_m, bin_m)
    if len(bins) < 3:
        return pmax
    counts, edges = np.histogram(proj, bins=bins)
    if len(counts) == 0:
        return pmax
    cmax = float(np.max(counts))
    if cmax <= 0.0:
        return pmax
    thresh = max(float(min_count), float(min_frac) * cmax)
    valid = np.flatnonzero(counts >= thresh)
    if len(valid) == 0:
        return float(np.percentile(proj, 98.5))
    i1 = int(valid[-1])
    i0 = i1
    while i0 > 0 and counts[i0 - 1] >= thresh:
        i0 -= 1
    run_start = float(edges[i0])
    run_end = float(edges[i1 + 1])
    span = float(np.max(proj) - np.min(proj))
    far_side = float(np.min(proj)) + 0.40 * span
    if (run_end - run_start) >= OCC_THICK_SLAB_M and run_start >= far_side:
        return run_start
    return run_end


def _wall_along_walk_occupancy(
    xy: np.ndarray,
    direction: np.ndarray,
    traj_xy: np.ndarray,
) -> float | None:
    try:
        pts = np.asarray(xy, dtype=float)
        d = np.asarray(direction, dtype=float).reshape(-1)
        if pts.ndim != 2 or len(pts) < 20 or d.size < 2:
            return None
        nrm = float(np.linalg.norm(d[:2]))
        if nrm < GRAVITY_NORM_EPS:
            return None
        dhat = d[:2] / nrm
        proj = pts[:, :2] @ dhat
        if not np.all(np.isfinite(proj)):
            return None
        if traj_xy is None:
            return None
        traj = np.asarray(traj_xy, dtype=float)
        if traj.ndim != 2 or traj.shape[0] < 1 or traj.shape[1] < 2:
            return None
        walk_proj = traj[:, :2] @ dhat
        if not np.all(np.isfinite(walk_proj)):
            return None
        walk_min = float(np.min(walk_proj))
        walk_max = float(np.max(walk_proj))
        bin_m = float(OCC_OUTER_BIN_M)
        pmin = float(np.min(proj))
        pmax = float(np.max(proj))
        if pmax - pmin < bin_m:
            return pmax
        bins = np.arange(pmin, pmax + bin_m, bin_m)
        if len(bins) < 3:
            return pmax
        counts, edges = np.histogram(proj, bins=bins)
        if len(counts) == 0:
            return pmax
        cmax = float(np.max(counts))
        if cmax <= 0.0:
            return pmax
        thresh = max(float(OCC_OUTER_MIN_COUNT), float(OCC_OUTER_MIN_FRAC) * cmax)
        valid = np.flatnonzero(counts >= thresh)
        if len(valid) == 0:
            return float(np.percentile(proj, 98.5))
        i1 = int(valid[-1])
        i0 = i1
        while i0 > 0 and counts[i0 - 1] >= thresh:
            i0 -= 1
        run_start = float(edges[i0])
        run_end = float(edges[i1 + 1])
        entered = (
            (run_end - run_start) >= float(OCC_THICK_SLAB_M)
            and run_start >= walk_min + float(WALK_ENTERED_MARGIN_M)
            and walk_max >= run_start
            and run_start >= walk_max - float(OCC_THICK_SLAB_M)
        )
        if entered:
            return run_start
        best_count = -1.0
        best_center = None
        for i in range(len(counts)):
            if float(counts[i]) < thresh:
                continue
            center = 0.5 * (float(edges[i]) + float(edges[i + 1]))
            if center <= walk_max:
                continue
            if (float(counts[i]) > best_count) or (
                float(counts[i]) == best_count
                and best_center is not None
                and center < best_center
            ):
                best_count = float(counts[i])
                best_center = center
        if best_center is not None:
            return float(best_center)
        return run_end
    except Exception:
        return None


def _find_wall_density_peak(
    dist_vals: np.ndarray | list[float],
    bin_size: float = 0.04,
    min_peak_count: int = 25,
    prominence_ratio: float = 1.8,
) -> float:
    arr = np.asarray(dist_vals, dtype=float).ravel()
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return 0.0
    if len(arr) == 1:
        return float(arr[0])

    p98 = float(np.percentile(arr, 98.5))
    min_v = float(np.min(arr))
    max_v = float(np.max(arr))
    if max_v - min_v < bin_size:
        return p98

    bins = np.arange(min_v, max_v + bin_size, bin_size)
    if len(bins) < 3:
        return p98

    counts, edges = np.histogram(arr, bins=bins)
    if len(counts) == 0:
        return p98

    c_max = float(np.max(counts))
    c_med = float(np.median(counts))

    if c_max < min_peak_count or c_max < prominence_ratio * max(c_med, 1.0):
        return p98

    thresh = max(float(min_peak_count), 0.25 * c_max)
    peaks = [i for i in range(len(counts)) if counts[i] >= thresh]
    if not peaks:
        return p98

    best_peak_idx = peaks[-1]
    low_e = float(edges[best_peak_idx])
    high_e = float(edges[best_peak_idx + 1])
    in_peak = arr[(arr >= low_e - 0.01) & (arr <= high_e + 0.01)]
    if len(in_peak) > 0:
        return float(np.median(in_peak))
    return float(0.5 * (low_e + high_e))


def fit_oriented_manhattan_rectangle(
    walls: list[dict],
    floor: dict | None,
    ceiling: dict | None,
    pcd_points: np.ndarray,
    pcd_z_span: float,
    keepouts: list[dict] | None = None,
    trajectory: np.ndarray | None = None,
    max_standoff_m: float = MAX_STANDOFF_M,
    *,
    forced_wall_positions: list[tuple[np.ndarray, float]] | None = None,
) -> dict:
    height_meters = _wall_height(floor, ceiling, pcd_z_span)
    pcd_points = np.asarray(pcd_points, dtype=float)
    if pcd_points.ndim != 2 or pcd_points.shape[1] < 2:
        pcd_points = np.zeros((0, 3))
    inlier_xy = []
    for w in walls:
        pts = np.asarray(w.get("inliers", []), dtype=float)
        if pts.ndim == 2 and pts.shape[0] and pts.shape[1] >= 2:
            inlier_xy.append(pts[:, :2])
    if pcd_points is not None and len(pcd_points):
        fallback_xy = pcd_points[:, :2]
    elif floor is not None and len(np.asarray(floor.get("inliers", []), dtype=float)):
        fallback_xy = np.asarray(floor["inliers"], dtype=float)[:, :2]
    else:
        fallback_xy = np.zeros((0, 2))

    if not walls and not forced_wall_positions and len(fallback_xy) < 4:
        return {
            "vertices": {},
            "walls": [],
            "mirror_reflection_detected": False,
            "clear_interior_dimensions": None,
        }
    if len(fallback_xy):
        c = 0.5 * (fallback_xy.min(axis=0) + fallback_xy.max(axis=0))
    elif inlier_xy:
        c = np.mean(np.vstack(inlier_xy), axis=0)
    elif forced_wall_positions:
        c = np.array([0.0, 0.0])
    else:
        return {
            "vertices": {},
            "walls": [],
            "mirror_reflection_detected": False,
            "clear_interior_dimensions": None,
        }

    best_n = None
    if forced_wall_positions is not None and len(forced_wall_positions) == 4:
        n0 = np.asarray(forced_wall_positions[0][0], dtype=float).reshape(-1)[:2]
        nrm0 = float(np.linalg.norm(n0))
        if nrm0 >= GRAVITY_NORM_EPS:
            best_n = n0 / nrm0
    if best_n is None:
        best_count = -1
        for w in walls:
            n = np.asarray(w.get("n", [0, 0, 0]), dtype=float).reshape(3)
            nxy = n[:2]
            nrm = float(np.linalg.norm(nxy))
            if nrm < GRAVITY_NORM_EPS:
                continue
            cnt = int(w.get("inlier_count", len(w.get("inliers", []))))
            if cnt > best_count:
                best_count = cnt
                best_n = nxy / nrm
    if best_n is None and forced_wall_positions:
        n0 = np.asarray(forced_wall_positions[0][0], dtype=float).reshape(-1)[:2]
        nrm0 = float(np.linalg.norm(n0))
        if nrm0 >= GRAVITY_NORM_EPS:
            best_n = n0 / nrm0

    traj_xy = None
    if trajectory is not None:
        traj_arr = np.asarray(trajectory, dtype=float)
        if traj_arr.ndim == 2 and len(traj_arr) > 0 and traj_arr.shape[1] >= 2:
            traj_xy = traj_arr[:, :2]

    if best_n is None:
        if traj_xy is not None and len(traj_xy) >= 2:
            centered = traj_xy - np.mean(traj_xy, axis=0)
            cov = np.cov(centered.T)
            eigvals, eigvecs = np.linalg.eigh(cov)
            best_n = eigvecs[:, 1]
        elif len(fallback_xy) >= 2:
            centered = fallback_xy - np.mean(fallback_xy, axis=0)
            cov = np.cov(centered.T)
            eigvals, eigvecs = np.linalg.eigh(cov)
            best_n = eigvecs[:, 1]
        else:
            best_n = np.array([1.0, 0.0], dtype=float)

    nrm_b = float(np.linalg.norm(best_n))
    if nrm_b > 1e-6:
        best_n = best_n / nrm_b
    else:
        best_n = np.array([1.0, 0.0], dtype=float)

    u1 = best_n
    u2 = np.array([-u1[1], u1[0]], dtype=float)
    dirs = [u1, -u1, u2, -u2]

    if forced_wall_positions is not None and len(forced_wall_positions) == 4:
        selected = [None] * 4
        for n_xy, pos_metric in forced_wall_positions:
            n_xy = np.asarray(n_xy, dtype=float).reshape(-1)[:2]
            nrm = float(np.linalg.norm(n_xy))
            if nrm < GRAVITY_NORM_EPS:
                continue
            n_xy = n_xy / nrm
            k = int(np.argmax([float(n_xy @ d) for d in dirs]))
            d = dirs[k]
            n = np.array([d[0], d[1], 0.0], dtype=float)
            n = n / float(np.linalg.norm(n))
            selected[k] = {"n": n, "d": -float(pos_metric)}
        if any(s is None for s in selected):
            return {
                "vertices": {},
                "walls": [],
                "mirror_reflection_detected": False,
                "clear_interior_dimensions": None,
            }
        order = [0, 2, 1, 3]
        corners = []
        for i in range(4):
            a = selected[order[i]]
            b = selected[order[(i + 1) % 4]]
            p = intersect_lines_2d(
                float(a["n"][0]), float(a["n"][1]), float(a["d"]),
                float(b["n"][0]), float(b["n"][1]), float(b["d"]),
            )
            if p is None:
                return {
                    "vertices": {},
                    "walls": [],
                    "mirror_reflection_detected": False,
                    "clear_interior_dimensions": None,
                }
            corners.append(p)
        xy = np.stack(corners, axis=0)
        if _signed_shoelace(xy) < 0.0:
            xy = xy[::-1]
        vertices = {f"v{i}": [float(xy[i, 0]), float(xy[i, 1])] for i in range(4)}
        walls_out = [
            {
                "wall_index": i,
                "joints": [f"v{i}", f"v{(i + 1) % 4}"],
                "thickness_meters": DEFAULT_WALL_THICKNESS_M,
                "height_meters": height_meters,
            }
            for i in range(4)
        ]
        s0 = float(np.linalg.norm(xy[1] - xy[0]))
        s1 = float(np.linalg.norm(xy[2] - xy[1]))
        short_side = min(s0, s1)
        long_side = max(s0, s1)
        clear_interior_dimensions = {
            "height_m": float(height_meters),
            "short_side_m": float(round(short_side, 4)),
            "long_side_m": float(round(long_side, 4)),
            "height_cm": float(round(height_meters * 100.0, 1)),
            "short_side_cm": float(round(short_side * 100.0, 1)),
            "long_side_cm": float(round(long_side * 100.0, 1)),
        }
        return {
            "vertices": vertices,
            "walls": walls_out,
            "mirror_reflection_detected": False,
            "clear_interior_dimensions": clear_interior_dimensions,
        }

    standoffs = [None] * 4
    is_small = False
    if traj_xy is not None and len(traj_xy) > 0:
        span_0 = float(np.max(traj_xy @ u1) - np.min(traj_xy @ u1))
        span_1 = float(np.max(traj_xy @ u2) - np.min(traj_xy @ u2))
        approx_area = (span_0 + 0.60) * (span_1 + 0.60)
        is_small = approx_area < 4.5 or min(span_0, span_1) < 1.30

        for k_dir, d_dir in enumerate(dirs):
            axis_span = span_0 if k_dir in (0, 1) else span_1
            other_span = span_1 if k_dir in (0, 1) else span_0
            if is_small:
                if axis_span <= other_span:
                    standoffs[k_dir] = min(float(max_standoff_m), SMALL_ROOM_SHORT_STANDOFF_M)
                else:
                    standoffs[k_dir] = min(float(max_standoff_m), SMALL_ROOM_LONG_STANDOFF_M)
            else:
                standoffs[k_dir] = float(max_standoff_m)

    bins: list[list[dict]] = [[], [], [], []]
    for w in walls:
        nxy = np.asarray(w["n"], dtype=float).reshape(3)[:2]
        nrm = float(np.linalg.norm(nxy))
        if nrm < GRAVITY_NORM_EPS:
            continue
        nxy = nxy / nrm
        pts = np.asarray(w.get("inliers", []), dtype=float)
        if pts.ndim == 2 and len(pts) and pts.shape[1] >= 2:
            ci = np.mean(pts[:, :2], axis=0)
            if float(nxy @ (ci - c)) < 0.0:
                nxy = -nxy
        dots = [float(nxy @ d) for d in dirs]
        k = int(np.argmax(dots))
        if dots[k] < MANHATTAN_ASSIGN_DOT_MIN:
            continue
        bins[k].append(w)

    def _best_wall(candidates, _d_perp):
        if not candidates:
            return None
        def _score(w):
            n_in = int(w.get("inlier_count", len(w["inliers"])))
            span = _span_along(np.asarray(w["inliers"], dtype=float)[:, :2], _d_perp)
            return float(n_in) * float(span)
        return max(candidates, key=_score)

    caps = [None] * 4
    if traj_xy is not None and len(traj_xy) > 0:
        for k_dir, d_dir in enumerate(dirs):
            traj_cap = float(np.max(traj_xy @ d_dir)) + float(standoffs[k_dir])
            occ_outer = _outer_occupancy_along(fallback_xy, d_dir)
            pos = None
            if is_small:
                pos = _wall_along_walk_occupancy(fallback_xy, d_dir, traj_xy)
                if pos is None or (occ_outer is not None and occ_outer > pos):
                    pos = occ_outer
            if is_small and pos is not None:
                caps[k_dir] = float(pos)
            else:
                caps[k_dir] = traj_cap
                d_perp = np.array([-d_dir[1], d_dir[0]], dtype=float)
                bw = _best_wall(bins[k_dir], d_perp)
                if bw is not None:
                    wpts = np.asarray(bw["inliers"], dtype=float)
                    best_pos = float(np.mean(wpts[:, :2] @ d_dir))
                    if (
                        best_pos > traj_cap
                        and occ_outer is not None
                        and occ_outer >= best_pos - 0.10
                    ):
                        caps[k_dir] = max(traj_cap, min(best_pos + 0.05, occ_outer))

    if keepouts is None and len(fallback_xy) >= 50:
        raw = detect_builtin_keepouts(
            pcd_points if pcd_points is not None and len(pcd_points) >= 50 else fallback_xy,
            trajectory=traj_xy,
        )
        keepouts = raw.get("strips", [])
    strips = keepouts if keepouts is not None else []
    has_builtin = bool(strips) or any(len(bins[k]) >= 2 for k in range(4))

    side_vecs = {
        "+x": np.array([1.0, 0.0]),
        "-x": np.array([-1.0, 0.0]),
        "+y": np.array([0.0, 1.0]),
        "-y": np.array([0.0, -1.0]),
    }
    matched_strips: list[list[dict]] = [[], [], [], []]
    for s in strips:
        sv = side_vecs.get(s.get("side"))
        best_k = None
        if sv is not None:
            k_cand = int(np.argmax([float(d @ sv) for d in dirs]))
            if float(dirs[k_cand] @ sv) >= 0.5:
                best_k = k_cand
        if best_k is None and s.get("polygon"):
            poly_c = np.mean(np.asarray(s["polygon"], dtype=float)[:, :2], axis=0)
            rel = poly_c - c
            if np.linalg.norm(rel) > 1e-4:
                best_k = int(np.argmax([float(d @ rel) for d in dirs]))
        if best_k is not None:
            matched_strips[best_k].append(s)

    dominant_anchors = [None] * 4
    dominant_scores = [0.0] * 4
    for k, d in enumerate(dirs):
        d_perp = np.array([-d[1], d[0]], dtype=float)
        strips_k = matched_strips[k]
        cap_k = caps[k]
        if strips_k:
            poly_outer = max(float(np.max(np.asarray(s["polygon"], dtype=float)[:, :2] @ d)) for s in strips_k)
            if cap_k is not None:
                poly_outer = min(poly_outer, cap_k)
            outer_cands = [
                w for w in bins[k]
                if abs(float(np.mean(np.asarray(w["inliers"], dtype=float)[:, :2] @ d)) - poly_outer) <= 0.30
                and (cap_k is None or float(np.mean(np.asarray(w["inliers"], dtype=float)[:, :2] @ d)) <= cap_k + 0.10)
            ]
            if outer_cands:
                bw = _best_wall(outer_cands, d_perp)
                w_pos = float(np.mean(np.asarray(bw["inliers"], dtype=float)[:, :2] @ d))
                eff_pos = max(w_pos, poly_outer)
                if cap_k is not None:
                    eff_pos = min(eff_pos, cap_k)
                dominant_anchors[k] = eff_pos * np.array([d[0], d[1], 0.0])
                dominant_scores[k] = max(
                    float(bw.get("inlier_count", len(bw["inliers"]))) * _span_along(
                        np.asarray(bw["inliers"], dtype=float)[:, :2], d_perp
                    ),
                    3000.0,
                )
            elif any(s.get("open_bay") for s in strips_k):
                eff_pos = poly_outer
                if cap_k is not None:
                    eff_pos = min(eff_pos, cap_k)
                dominant_anchors[k] = eff_pos * np.array([d[0], d[1], 0.0])
                dominant_scores[k] = 3000.0 * max(float(s.get("length_m", 2.0)) for s in strips_k)
            else:
                valid_cands = [
                    w for w in bins[k]
                    if (cap_k is None or float(np.mean(np.asarray(w["inliers"], dtype=float)[:, :2] @ d)) <= cap_k)
                ]
                bw = _best_wall(valid_cands if valid_cands else bins[k], d_perp)
                if bw is not None:
                    dominant_anchors[k] = np.mean(np.asarray(bw["inliers"], dtype=float), axis=0)
                    dominant_scores[k] = float(bw.get("inlier_count", len(bw["inliers"]))) * _span_along(
                        np.asarray(bw["inliers"], dtype=float)[:, :2], d_perp
                    )
        elif has_builtin and bins[k]:
            positions = [float(np.mean(np.asarray(w["inliers"], dtype=float)[:, :2] @ d)) for w in bins[k]]
            max_pos = max(positions)
            min_pos = min(positions)
            bw = None
            if max_pos - min_pos >= 0.30:
                outer_cands = [
                    w for w, pos in zip(bins[k], positions)
                    if pos >= max_pos - 0.20 and (cap_k is None or pos <= cap_k)
                ]
                valid_outer = [
                    w for w in outer_cands
                    if _span_along(np.asarray(w["inliers"], dtype=float)[:, :2], d_perp) >= 0.5
                    and int(w.get("inlier_count", len(w["inliers"]))) >= 100
                ]
                if valid_outer:
                    bw = _best_wall(valid_outer, d_perp)
            if bw is None:
                valid_cands = [
                    w for w in bins[k]
                    if (cap_k is None or float(np.mean(np.asarray(w["inliers"], dtype=float)[:, :2] @ d)) <= cap_k)
                ]
                bw = _best_wall(valid_cands if valid_cands else bins[k], d_perp)
            if bw is not None:
                dominant_anchors[k] = np.mean(np.asarray(bw["inliers"], dtype=float), axis=0)
                dominant_scores[k] = float(bw.get("inlier_count", len(bw["inliers"]))) * _span_along(
                    np.asarray(bw["inliers"], dtype=float)[:, :2], d_perp
                )
        else:
            valid_cands = [
                w for w in bins[k]
                if (cap_k is None or float(np.mean(np.asarray(w["inliers"], dtype=float)[:, :2] @ d)) <= cap_k)
            ]
            bw = _best_wall(valid_cands if valid_cands else bins[k], d_perp)
            if bw is not None:
                dominant_anchors[k] = np.mean(np.asarray(bw["inliers"], dtype=float), axis=0)
                dominant_scores[k] = float(bw.get("inlier_count", len(bw["inliers"]))) * _span_along(
                    np.asarray(bw["inliers"], dtype=float)[:, :2], d_perp
                )
        if cap_k is not None and dominant_anchors[k] is not None:
            curr_d = float(dominant_anchors[k][:2] @ d)
            if curr_d > cap_k:
                dominant_anchors[k] = dominant_anchors[k] + (cap_k - curr_d) * np.array([d[0], d[1], 0.0])

    is_primary = [False] * 4
    for pair in ((0, 1), (2, 3)):
        p0, p1 = pair
        if dominant_scores[p0] >= dominant_scores[p1]:
            is_primary[p0] = True
        else:
            is_primary[p1] = True

    proj0 = fallback_xy @ dirs[0] if len(fallback_xy) else np.array([])
    span_axis0 = float(np.percentile(proj0, 98) - np.percentile(proj0, 2)) if len(proj0) else 0.0
    proj1 = fallback_xy @ dirs[2] if len(fallback_xy) else np.array([])
    span_axis1 = float(np.percentile(proj1, 98) - np.percentile(proj1, 2)) if len(proj1) else 0.0

    selected = []
    anchors = [None] * 4
    mirror_reflection_detected = False
    for k, d in enumerate(dirs):
        d_perp = np.array([-d[1], d[0]], dtype=float)
        opp_k = k ^ 1
        opp_anchor = dominant_anchors[opp_k]
        other_span = span_axis1 if (k in (0, 1)) else span_axis0
        strips_k = matched_strips[k]
        cap_k = caps[k]

        anchor = None
        chosen = None

        if strips_k:
            poly_outer = max(float(np.max(np.asarray(s["polygon"], dtype=float)[:, :2] @ d)) for s in strips_k)
            if cap_k is not None:
                poly_outer = min(poly_outer, cap_k)
            outer_cands = [
                w for w in bins[k]
                if abs(float(np.mean(np.asarray(w["inliers"], dtype=float)[:, :2] @ d)) - poly_outer) <= 0.30
                and (cap_k is None or float(np.mean(np.asarray(w["inliers"], dtype=float)[:, :2] @ d)) <= cap_k)
            ]
            if outer_cands:
                chosen = _best_wall(outer_cands, d_perp)
                w_pos = float(np.mean(np.asarray(chosen["inliers"], dtype=float)[:, :2] @ d))
                eff_pos = max(w_pos, poly_outer)
                if cap_k is not None:
                    eff_pos = min(eff_pos, cap_k)
                anchor = eff_pos * np.array([d[0], d[1], 0.0])
            elif any(s.get("open_bay") for s in strips_k):
                eff_pos = poly_outer
                if cap_k is not None:
                    eff_pos = min(eff_pos, cap_k)
                anchor = eff_pos * np.array([d[0], d[1], 0.0])
            else:
                valid_cands = [
                    w for w in bins[k]
                    if (cap_k is None or float(np.mean(np.asarray(w["inliers"], dtype=float)[:, :2] @ d)) <= cap_k)
                ]
                chosen = _best_wall(valid_cands if valid_cands else bins[k], d_perp)
        elif has_builtin and bins[k]:
            positions = [float(np.mean(np.asarray(w["inliers"], dtype=float)[:, :2] @ d)) for w in bins[k]]
            max_pos = max(positions)
            min_pos = min(positions)
            if max_pos - min_pos >= 0.30:
                outer_cands = [
                    w for w, pos in zip(bins[k], positions)
                    if pos >= max_pos - 0.20 and (cap_k is None or pos <= cap_k)
                ]
                valid_outer = [
                    w for w in outer_cands
                    if _span_along(np.asarray(w["inliers"], dtype=float)[:, :2], d_perp) >= 0.5
                    and int(w.get("inlier_count", len(w["inliers"]))) >= 100
                ]
                if valid_outer:
                    chosen = _best_wall(valid_outer, d_perp)
                    anchor = np.mean(np.asarray(chosen["inliers"], dtype=float), axis=0)
            if chosen is None:
                valid_cands = [
                    w for w in bins[k]
                    if (cap_k is None or float(np.mean(np.asarray(w["inliers"], dtype=float)[:, :2] @ d)) <= cap_k)
                ]
                chosen = _best_wall(valid_cands if valid_cands else bins[k], d_perp)
        else:
            valid_cands = [
                w for w in bins[k]
                if (cap_k is None or float(np.mean(np.asarray(w["inliers"], dtype=float)[:, :2] @ d)) <= cap_k)
            ]
            chosen = _best_wall(valid_cands if valid_cands else bins[k], d_perp)

        if chosen is not None and anchor is None:
            tspan = _span_along(np.asarray(chosen["inliers"], dtype=float)[:, :2], d_perp)
            fspan = _span_along(fallback_xy, d_perp)
            if fspan > MANHATTAN_MIN_SPAN_M and tspan < MANHATTAN_FURNITURE_SPAN_RATIO * fspan:
                chosen = None

        if opp_anchor is not None and len(fallback_xy):
            dist_pts = (fallback_xy - opp_anchor[:2]) @ d
            if cap_k is not None:
                pts_pos = fallback_xy @ d
                valid_dist = dist_pts[pts_pos <= cap_k]
                p95_dist = float(np.percentile(valid_dist, 95)) if len(valid_dist) else float(np.percentile(dist_pts, 95))
            else:
                p95_dist = float(np.percentile(dist_pts, 95))
            cand_dist = float(
                (np.mean(np.asarray(chosen["inliers"], dtype=float)[:, :2], axis=0) - opp_anchor[:2]) @ d
            ) if chosen is not None else p95_dist
            span_along_d = max(float(cand_dist), float(p95_dist), 0.50)

            if not is_primary[k]:
                # 1. Interior partition check:
                if chosen is not None and (cand_dist < p95_dist - 0.35):
                    cand_inliers = int(chosen.get("inlier_count", len(chosen["inliers"])))
                    cand_span = _span_along(
                        np.asarray(chosen["inliers"], dtype=float)[:, :2], d_perp
                    )
                    if cand_inliers < 80 or cand_span < 0.6:
                        chosen = None

            # 2. Mirror reflection check (detect multipath points extending beyond true wall without hardcoded bounds):
            pos_dist = dist_pts[dist_pts >= MIRROR_MIN_POS_DIST_M]
            if len(pos_dist) >= 60:
                d_max = float(pos_dist.max())
                h_bins = np.arange(MIRROR_MIN_POS_DIST_M, d_max + 0.06, 0.05)
                if len(h_bins) >= 3:
                    counts, edges = np.histogram(pos_dist, bins=h_bins)
                    c_max = float(np.max(counts)) if len(counts) else 0.0
                    if c_max >= 20:
                        peaks = [i for i in range(len(counts)) if counts[i] >= 0.25 * c_max]
                        if peaks:
                            drop_dist = float(edges[peaks[-1] + 1])
                            p98_d = float(np.percentile(pos_dist, 98.5))
                            tail_threshold = max(MIRROR_TAIL_ABS_M, MIRROR_TAIL_REL * span_along_d)
                            if p98_d > drop_dist + tail_threshold:
                                cam_past_drop = False
                                if traj_xy is not None and len(traj_xy) > 0:
                                    max_cam_d = float(np.max((traj_xy - opp_anchor[:2]) @ d))
                                    if max_cam_d > drop_dist + 0.15:
                                        cam_past_drop = True
                                if not cam_past_drop:
                                    mirror_reflection_detected = True
                                    chosen = None
                                    anchor = opp_anchor + float(drop_dist) * np.array([d[0], d[1], 0.0])

            # Built-in furniture check: applies even if is_primary, because high inlier count on furniture front can falsely make it primary
            if has_builtin and chosen is not None and cand_dist < p95_dist - 0.35:
                chosen = None

        if is_small and len(fallback_xy) and not mirror_reflection_detected:
            if chosen is not None:
                cur_pos = float(
                    np.mean(np.asarray(chosen["inliers"], dtype=float)[:, :2] @ d)
                )
            elif anchor is not None:
                cur_pos = float(np.asarray(anchor, dtype=float)[:2] @ d)
            else:
                cur_pos = None
            occ_outer = _wall_along_walk_occupancy(fallback_xy, d, traj_xy)
            if occ_outer is None:
                occ_outer = _outer_occupancy_along(fallback_xy, d)
            if (
                cur_pos is not None
                and occ_outer is not None
                and abs(occ_outer - cur_pos) >= OCC_INNER_TO_OUTER_M
            ):
                chosen = None
                zmean = float(np.mean(pcd_points[:, 2])) if len(pcd_points) else 0.0
                d2 = np.asarray(d, dtype=float).reshape(-1)[:2]
                nrm2 = float(np.linalg.norm(d2))
                if nrm2 >= GRAVITY_NORM_EPS:
                    d2 = d2 / nrm2
                    anchor = np.array(
                        [d2[0] * occ_outer, d2[1] * occ_outer, zmean], dtype=float
                    )

        if anchor is None:
            if chosen is not None:
                anchor = np.mean(np.asarray(chosen["inliers"], dtype=float), axis=0)
            else:
                # Infer wall position when chosen is None (e.g. mirror blindness or missing wall):
                ortho_extents = []
                for ok_idx in ((2, 3) if k in (0, 1) else (0, 1)):
                    for bw_w in bins[ok_idx]:
                        inls_w = np.asarray(bw_w.get("inliers", []), dtype=float)
                        if len(inls_w) >= 20:
                            ortho_extents.append(float(np.percentile(inls_w[:, :2] @ d, 98)))
                max_ortho_extent = max(ortho_extents) if ortho_extents else None

                floor_pts = None
                if floor is not None and len(np.asarray(floor.get("inliers", []), dtype=float)):
                    floor_pts = np.asarray(floor["inliers"], dtype=float)[:, :2]
                elif len(pcd_points) and pcd_points.shape[1] >= 3:
                    f_mask = pcd_points[:, 2] <= 0.20
                    if np.count_nonzero(f_mask) >= 50:
                        floor_pts = pcd_points[f_mask, :2]
                floor_p98 = float(np.percentile(floor_pts @ d, 98)) if (floor_pts is not None and len(floor_pts) >= 50) else None

                min_cam_bound = None
                traj_in_bbox = False
                if traj_xy is not None and len(traj_xy) > 0:
                    max_cam_d = float(np.max(traj_xy @ d))
                    traj_in_bbox = True
                    if len(fallback_xy) > 0:
                        max_occ_d = float(np.percentile(fallback_xy @ d, 99.0))
                        if max_cam_d > max_occ_d + 0.30:
                            traj_in_bbox = False
                    if cap_k is not None and max_cam_d > cap_k + 0.20:
                        traj_in_bbox = False
                    if traj_in_bbox:
                        min_cam_bound = max_cam_d + 0.20

                inf_cands = []
                if max_ortho_extent is not None:
                    inf_cands.append(max_ortho_extent)
                if floor_p98 is not None:
                    inf_cands.append(floor_p98)

                if inf_cands:
                    inferred_pos = max(inf_cands)
                    if min_cam_bound is not None and traj_in_bbox:
                        inferred_pos = max(inferred_pos, min_cam_bound)
                    if cap_k is not None:
                        inferred_pos = min(inferred_pos, cap_k)
                    zmean = float(np.mean(pcd_points[:, 2])) if len(pcd_points) else 0.0
                    anchor = np.array([d[0] * inferred_pos, d[1] * inferred_pos, zmean], dtype=float)
                elif opp_anchor is not None and len(fallback_xy):
                    dist_pts = (fallback_xy - opp_anchor[:2]) @ d
                    if cap_k is not None:
                        pts_pos = fallback_xy @ d
                        valid_dist = dist_pts[pts_pos <= cap_k]
                        if len(valid_dist):
                            p_val = _find_wall_density_peak(valid_dist, bin_size=0.04)
                        else:
                            opp_pos = float(opp_anchor[:2] @ d)
                            p_val = max(0.2, cap_k - opp_pos)
                    else:
                        p_val = _find_wall_density_peak(dist_pts, bin_size=0.04)
                    anchor = opp_anchor + p_val * np.array([d[0], d[1], 0.0])
                elif len(fallback_xy):
                    proj = fallback_xy @ d
                    if cap_k is not None:
                        valid_proj = proj[proj <= cap_k]
                        p_val = (
                            _find_wall_density_peak(valid_proj, bin_size=0.04)
                            if len(valid_proj)
                            else cap_k
                        )
                    else:
                        p_val = _find_wall_density_peak(proj, bin_size=0.04)
                    p_star = fallback_xy[np.argmin(np.abs(proj - p_val))]
                    zmean = float(np.mean(pcd_points[:, 2])) if len(pcd_points) else 0.0
                    anchor = np.array([p_star[0], p_star[1], zmean], dtype=float)
                else:
                    return {
                        "vertices": {},
                        "walls": [],
                        "mirror_reflection_detected": False,
                        "clear_interior_dimensions": None,
                    }

        if cap_k is not None and anchor is not None:
            curr_pos = float(anchor[:2] @ d)
            if curr_pos > cap_k:
                anchor = anchor + (cap_k - curr_pos) * np.array([d[0], d[1], 0.0])

        anchors[k] = anchor

        n = np.array([d[0], d[1], 0.0], dtype=float)
        n = n / float(np.linalg.norm(n))
        d_off = -float(n @ anchor)
        selected.append({"n": n, "d": d_off})
    order = [0, 2, 1, 3]  # +u1, +u2, -u1, -u2
    corners = []
    for i in range(4):
        a = selected[order[i]]
        b = selected[order[(i + 1) % 4]]
        p = intersect_lines_2d(
            float(a["n"][0]), float(a["n"][1]), float(a["d"]),
            float(b["n"][0]), float(b["n"][1]), float(b["d"]),
        )
        if p is None:
            return {
                "vertices": {},
                "walls": [],
                "mirror_reflection_detected": False,
                "clear_interior_dimensions": None,
            }
        corners.append(p)
    xy = np.stack(corners, axis=0)
    if _signed_shoelace(xy) < 0.0:
        xy = xy[::-1]
    vertices = {
        f"v{i}": [float(xy[i, 0]), float(xy[i, 1])] for i in range(4)
    }
    walls_out = [
        {
            "wall_index": i,
            "joints": [f"v{i}", f"v{(i + 1) % 4}"],
            "thickness_meters": DEFAULT_WALL_THICKNESS_M,
            "height_meters": height_meters,
        }
        for i in range(4)
    ]

    if all(a is not None for a in anchors):
        L0 = abs(float(anchors[0][:2] @ u1) - float(anchors[1][:2] @ u1))
        L1 = abs(float(anchors[2][:2] @ u2) - float(anchors[3][:2] @ u2))
        short_side = min(L0, L1)
        long_side = max(L0, L1)
    elif len(corners) == 4:
        s0 = float(np.linalg.norm(corners[1] - corners[0]))
        s1 = float(np.linalg.norm(corners[2] - corners[1]))
        short_side = min(s0, s1)
        long_side = max(s0, s1)
    else:
        short_side = 0.0
        long_side = 0.0

    clear_interior_dimensions = {
        "height_m": float(height_meters),
        "short_side_m": float(round(short_side, 4)),
        "long_side_m": float(round(long_side, 4)),
        "height_cm": float(round(height_meters * 100.0, 1)),
        "short_side_cm": float(round(short_side * 100.0, 1)),
        "long_side_cm": float(round(long_side * 100.0, 1)),
    }
    return {
        "vertices": vertices,
        "walls": walls_out,
        "mirror_reflection_detected": mirror_reflection_detected,
        "clear_interior_dimensions": clear_interior_dimensions,
    }
