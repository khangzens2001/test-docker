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
    mesh: o3d.geometry.TriangleMesh | o3d.geometry.PointCloud,
    number_of_points: int | None = None,
) -> o3d.geometry.PointCloud:
    empty = o3d.geometry.PointCloud()
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
    if isinstance(mesh, o3d.geometry.PointCloud):
        if len(mesh.points) == 0 or not mesh.has_colors():
            return empty
        pts = np.asarray(mesh.points, dtype=float)
        cols = np.asarray(mesh.colors, dtype=float)
        if len(pts) > n:
            np.random.seed(42)
            idx = np.random.choice(len(pts), size=n, replace=False)
            pts = pts[idx]
            cols = cols[idx]
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts)
        pcd.colors = o3d.utility.Vector3dVector(cols)
        return pcd
    if hasattr(mesh, "triangles") and len(mesh.triangles) == 0 and len(mesh.vertices) > 0:
        if not mesh.has_vertex_colors():
            return empty
        pts = np.asarray(mesh.vertices, dtype=float)
        cols = np.asarray(mesh.vertex_colors, dtype=float)
        if len(pts) > n:
            np.random.seed(42)
            idx = np.random.choice(len(pts), size=n, replace=False)
            pts = pts[idx]
            cols = cols[idx]
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts)
        pcd.colors = o3d.utility.Vector3dVector(cols)
        return pcd
    if len(mesh.vertices) == 0 or len(mesh.triangles) == 0:
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
    canonicalize_floorplan_se2,
    canonicalize_l_polygon_se2,
)


def apply_layout_frames(
    pcd: o3d.geometry.PointCloud,
    layout: dict,
    gravity: np.ndarray | None,
) -> o3d.geometry.PointCloud:
    work = o3d.geometry.PointCloud(pcd)
    if len(work.points) == 0:
        return work
    layout = layout if isinstance(layout, dict) else {}
    if gravity is not None and is_usable_gravity(gravity):
        g = np.asarray(gravity, dtype=float).reshape(3)
        nrm = float(np.linalg.norm(g))
        if nrm > 0.0 and abs(g[2]) > 0.9 * nrm:
            gv = layout.get("gravity_vector")
            if isinstance(gv, (list, tuple, np.ndarray)) and len(gv) == 3:
                gv_arr = np.asarray(gv, dtype=float).reshape(3)
                gv_nrm = float(np.linalg.norm(gv_arr))
                if gv_nrm > 0.0 and abs(gv_arr[1]) > 0.9 * gv_nrm:
                    g = gv_arr
                    nrm = gv_nrm
        if nrm > 0.0:
            work.rotate(align_rotation(g / nrm), center=(0.0, 0.0, 0.0))
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
            det2 = float(np.linalg.det(r2))
            if det2 > 0:
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
    if len(pts) >= 20:
        z_cloud_top = float(np.percentile(pts[:, 2], 99.5))
        if np.isfinite(z_cloud_top) and (z_cloud_top > cutaway + 0.30) and (z_cloud_top < float(height_m)):
            z_max = min(float(height_m), z_cloud_top) - float(cutaway)
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
    work = o3d.geometry.PointCloud(pcd)
    work.normals = o3d.utility.Vector3dVector()
    stem, ext = os.path.splitext(filename)
    tmp = os.path.join(session_dir, f"{stem}.tmp{ext}")
    final = os.path.join(session_dir, filename)
    try:
        ok = o3d.io.write_point_cloud(tmp, work, write_ascii=False)
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


def denoise_planar_surfaces(
    pcd: o3d.geometry.PointCloud, layout: dict | None = None
) -> o3d.geometry.PointCloud:
    """Apply smooth, continuous planar surface unbowing within r_wall = 0.055m of layout walls
    without post-hoc Z-flare regression or exterior point squashing."""
    work = o3d.geometry.PointCloud(pcd)
    n_pts = len(work.points)
    if n_pts < 120:
        return work
    try:
        pts = np.asarray(work.points, dtype=float).copy()
        xy_poly = layout_xy_polygon(layout) if isinstance(layout, dict) else np.zeros((0, 2))
        if len(xy_poly) >= 4:
            r_wall = 0.055
            n_v = len(xy_poly)
            for idx in range(n_v):
                p0 = xy_poly[idx]
                p1 = xy_poly[(idx + 1) % n_v]
                for ax in (0, 1):
                    par = 1 - ax
                    if abs(float(p1[ax] - p0[ax])) <= 0.015 and abs(float(p1[par] - p0[par])) >= 0.30:
                        w_coord = 0.5 * float(p0[ax] + p1[ax])
                        s_lo = min(float(p0[par]), float(p1[par]))
                        s_hi = max(float(p0[par]), float(p1[par]))
                        if s_hi <= s_lo:
                            continue
                        seg_mask = (
                            (pts[:, par] >= s_lo)
                            & (pts[:, par] <= s_hi)
                            & (pts[:, 2] >= 0.05)
                            & (np.abs(pts[:, ax] - w_coord) <= r_wall)
                        )
                        if int(np.count_nonzero(seg_mask)) >= 30:
                            seg_idx = np.flatnonzero(seg_mask)
                            s_vals = pts[seg_idx, par]
                            x_vals = pts[seg_idx, ax]
                            w_med = float(np.median(x_vals))
                            q10, q90 = np.percentile(x_vals, [10.0, 90.0])
                            thickness = float(q90 - q10)
                            order = np.argsort(s_vals)
                            dx_adj = np.diff(x_vals[order])
                            local_sigma = (
                                float(np.median(np.abs(dx_adj))) / (0.6745 * np.sqrt(2.0))
                                if len(dx_adj) > 0
                                else float(np.std(x_vals))
                            )
                            sheet_factor = (
                                float(np.exp(-((local_sigma / 0.012) ** 2)))
                                if thickness <= 0.14
                                else 0.0
                            )
                            if sheet_factor > 0.05:
                                u_outer = np.clip(np.abs(x_vals - w_med) / r_wall, 0.0, 1.0)
                                w_taper = ((1.0 - (u_outer ** 4)) ** 2) * sheet_factor
                                pts[seg_idx, ax] = x_vals - 0.85 * (x_vals - w_med) * w_taper

        work.points = o3d.utility.Vector3dVector(pts)
    except Exception:
        return o3d.geometry.PointCloud(pcd)
    return work


def _deduplicate_2d_surface(
    uv: np.ndarray,
    cols: np.ndarray,
    span_u: float,
    span_v: float,
    grid_step: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Collapse multi-layer 2D surface points onto 1 representative point per grid_step cell."""
    if len(uv) == 0:
        return uv, cols
    us = np.arange(0.5 * grid_step, span_u, grid_step)
    vs = np.arange(0.5 * grid_step, span_v, grid_step)
    nu = max(2, len(us))
    nv = max(2, len(vs))
    iu = np.clip(np.floor(uv[:, 0] / grid_step).astype(np.int32), 0, nu - 1)
    iv = np.clip(np.floor(uv[:, 1] / grid_step).astype(np.int32), 0, nv - 1)
    cell_id = iv * nu + iu
    uniq_cells, inv = np.unique(cell_id, return_inverse=True)
    counts = np.bincount(inv).astype(float)
    u_mean = np.bincount(inv, weights=uv[:, 0]) / counts
    v_mean = np.bincount(inv, weights=uv[:, 1]) / counts
    sat_pt = np.ptp(cols, axis=1)
    lum_pt = 0.299 * cols[:, 0] + 0.587 * cols[:, 1] + 0.114 * cols[:, 2]
    w_col = np.where(
        (sat_pt >= 0.12) | (lum_pt <= 0.32) | (lum_pt >= 0.86),
        4.5,
        1.0,
    )
    w_sum = np.maximum(np.bincount(inv, weights=w_col), 1e-6)
    c_dedup = np.column_stack(
        [np.bincount(inv, weights=cols[:, ch] * w_col) / w_sum for ch in range(3)]
    )
    iu_u = uniq_cells % nu
    iv_u = uniq_cells // nu
    u_ctr = (iu_u + 0.5) * grid_step
    v_ctr = (iv_u + 0.5) * grid_step
    u_dedup = (
        u_ctr
        + 0.05 * np.clip(u_mean - u_ctr, -0.5 * grid_step, 0.5 * grid_step)
        + rng.uniform(-0.23 * grid_step, 0.23 * grid_step, size=len(u_ctr))
    )
    v_dedup = (
        v_ctr
        + 0.05 * np.clip(v_mean - v_ctr, -0.5 * grid_step, 0.5 * grid_step)
        + rng.uniform(-0.23 * grid_step, 0.23 * grid_step, size=len(v_ctr))
    )
    u_dedup = np.clip(u_dedup, 0.0, span_u)
    v_dedup = np.clip(v_dedup, 0.0, span_v)
    return np.column_stack([u_dedup, v_dedup]), c_dedup


def _harmonize_and_inpaint_2d(
    uv: np.ndarray,
    cols: np.ndarray,
    span_u: float,
    span_v: float,
    grid_step: float,
    rng: np.random.Generator,
    is_wall: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Harmonize multi-keyframe exposure seams on a 2D surface and inpaint holes without Voronoi edges."""
    from scipy.spatial import cKDTree

    out_c = cols.copy()
    lum = 0.299 * cols[:, 0] + 0.587 * cols[:, 1] + 0.114 * cols[:, 2]
    sat = np.ptp(cols, axis=1)

    p_lo = max(float(np.percentile(lum, 22.0)), 0.24)
    p_hi = min(float(np.percentile(lum, 86.0)), 0.88)
    bg_m = (lum >= p_lo) & (lum <= max(p_hi, p_lo + 0.10)) & (sat < 0.12)
    if int(np.count_nonzero(bg_m)) < 30:
        bg_m = (lum > max(float(np.percentile(lum, 18.0)), 0.18)) & (sat < 0.22)
    if int(np.count_nonzero(bg_m)) < 30:
        bg_m = lum > float(np.percentile(lum, 12.0))

    bg_uv = uv[bg_m]
    bg_c = cols[bg_m]
    glob_med = np.median(bg_c, axis=0)
    glob_lum = float(0.299 * glob_med[0] + 0.587 * glob_med[1] + 0.114 * glob_med[2])
    glob_chroma = glob_med - glob_lum

    cell_sz = 0.08
    nu = max(3, int(np.ceil(span_u / cell_sz)))
    nv = max(3, int(np.ceil(span_v / cell_sz)))
    cu = np.clip((bg_uv[:, 0] / cell_sz).astype(int), 0, nu - 1)
    cv = np.clip((bg_uv[:, 1] / cell_sz).astype(int), 0, nv - 1)
    anchor_uv_list: list[list[float]] = []
    anchor_c_list: list[np.ndarray] = []
    for iu in range(nu):
        for iv in range(nv):
            m_cell = (cu == iu) & (cv == iv)
            if int(np.count_nonzero(m_cell)) >= 3:
                anchor_uv_list.append([(iu + 0.5) * cell_sz, (iv + 0.5) * cell_sz])
                cell_med = np.median(bg_c[m_cell], axis=0)
                cell_lum = float(0.299 * cell_med[0] + 0.587 * cell_med[1] + 0.114 * cell_med[2])
                cell_chroma = cell_med - cell_lum
                neutral_cell = (0.35 * cell_lum + 0.65 * glob_lum) + (0.18 * cell_chroma + 0.82 * glob_chroma)
                anchor_c_list.append(neutral_cell)
    if len(anchor_uv_list) >= 4:
        anchor_uv = np.asarray(anchor_uv_list, dtype=float)
        anchor_c = np.asarray(anchor_c_list, dtype=float)
    else:
        anchor_uv = bg_uv
        anchor_c = 0.40 * bg_c + 0.60 * glob_med

    tree_a = cKDTree(anchor_uv)
    k_a = min(36, len(anchor_uv))

    v_bins = np.linspace(0.0, span_v, 13)
    v_centers = 0.5 * (v_bins[:-1] + v_bins[1:])
    v_prof = np.zeros((12, 3), dtype=float)
    for bi in range(12):
        bm = (anchor_uv[:, 1] >= v_bins[bi]) & (anchor_uv[:, 1] <= v_bins[bi + 1])
        v_prof[bi] = (
            0.50 * np.median(anchor_c[bm], axis=0) + 0.50 * glob_med
            if int(np.count_nonzero(bm)) >= 2
            else glob_med
        )
    for _ in range(4):
        sm = v_prof.copy()
        sm[1:-1] = 0.25 * v_prof[:-2] + 0.50 * v_prof[1:-1] + 0.25 * v_prof[2:]
        sm[0] = 0.65 * v_prof[0] + 0.35 * v_prof[1]
        sm[-1] = 0.65 * v_prof[-1] + 0.35 * v_prof[-2]
        v_prof = sm

    def _eval_smooth_bg(query_uv: np.ndarray) -> np.ndarray:
        d_q, i_q = tree_a.query(query_uv, k=k_a)
        if k_a == 1:
            d_q, i_q = d_q[:, None], i_q[:, None]
        w_q = np.exp(-0.5 * (d_q / 0.42) ** 2)[..., None] + 1e-6
        loc_c = np.sum(anchor_c[i_q] * w_q, axis=1) / np.sum(w_q, axis=1)
        if is_wall:
            v_interp = np.column_stack(
                [np.interp(query_uv[:, 1], v_centers, v_prof[:, ch]) for ch in range(3)]
            )
            return 0.50 * loc_c + 0.50 * v_interp
        return 0.55 * loc_c + 0.45 * glob_med

    # Identify connected architectural features (doors, ventilation louvers, exhaust fans, mirrors, fixtures)
    # so they retain their true color, while small stains are suppressed into smooth_existing.
    is_structural_feature = np.zeros(len(uv), dtype=bool)
    salient_cand = (
        (sat >= 0.12)
        | (lum <= 0.35)
        | (lum >= 0.86)
    )
    if is_wall and int(np.count_nonzero(salient_cand)) >= 35:
        from scipy.ndimage import label as _nd_label

        c_step = max(float(grid_step) * 1.5, 0.012)
        ncu = max(2, int(np.ceil(span_u / c_step)))
        ncv = max(2, int(np.ceil(span_v / c_step)))
        ciu = np.clip((uv[:, 0] / c_step).astype(np.int32), 0, ncu - 1)
        civ = np.clip((uv[:, 1] / c_step).astype(np.int32), 0, ncv - 1)
        sal_grid = np.zeros((ncv, ncu), dtype=bool)
        sal_grid[civ[salient_cand], ciu[salient_cand]] = True
        lbl_grid, n_lbl = _nd_label(sal_grid, structure=np.ones((3, 3), dtype=int))
        if n_lbl > 0:
            pt_lbl = lbl_grid[civ, ciu] * salient_cand.astype(np.int32)
            lbl_counts = np.bincount(pt_lbl)
            keep_lbls = np.flatnonzero(lbl_counts >= 150)
            keep_lbls = keep_lbls[keep_lbls > 0]
            if len(keep_lbls) > 0:
                is_structural_feature = np.isin(pt_lbl, keep_lbls)

    smooth_existing = _eval_smooth_bg(uv)
    cols_work = cols.copy()

    high_wall_stain = is_wall & (uv[:, 1] > 0.70 * span_v) & (sat >= 0.05) & ~is_structural_feature
    is_artifact = ((sat >= 0.09) | (np.abs(lum - glob_lum) >= 0.09) | high_wall_stain) & ~is_structural_feature
    base_blend = np.where(is_structural_feature, 0.06, np.where(is_artifact, 0.96, 0.88))[:, None]

    gu, gv = np.meshgrid(
        np.arange(0.5 * grid_step, span_u, grid_step),
        np.arange(0.5 * grid_step, span_v, grid_step),
    )
    cand = np.column_stack([gu.ravel(), gv.ravel()])
    if len(cand) == 0:
        out_c = (1.0 - base_blend) * cols_work + base_blend * smooth_existing
        return np.clip(out_c, 0.0, 1.0), np.zeros((0, 2)), np.zeros((0, 3))

    tree_all = cKDTree(uv)
    d_nearest, _ = tree_all.query(cand, k=1)
    hole_m = d_nearest > (0.72 * grid_step)
    if not np.any(hole_m):
        out_c = (1.0 - base_blend) * cols_work + base_blend * smooth_existing
        return np.clip(out_c, 0.0, 1.0), np.zeros((0, 2)), np.zeros((0, 3))

    h_uv = cand[hole_m].copy()
    tree_h = cKDTree(h_uv)
    d_to_hole, _ = tree_h.query(uv, k=1)
    boundary_w = np.exp(-0.5 * (np.maximum(0.0, d_to_hole - grid_step) / 0.075) ** 2)[:, None]
    blend = np.where(
        is_structural_feature[:, None],
        base_blend,
        np.maximum(base_blend, 0.97 * boundary_w),
    )
    out_c = (1.0 - blend) * cols_work + blend * smooth_existing
    out_c += rng.normal(0.0, 0.0045, size=out_c.shape)

    h_c = _eval_smooth_bg(h_uv)
    if np.any(is_structural_feature):
        d_feat, idx_feat = tree_all.query(h_uv, k=min(8, len(uv)))
        if d_feat.ndim == 1:
            d_feat, idx_feat = d_feat[:, None], idx_feat[:, None]
        # Only bridge tiny internal pinholes (<= 2.4 * grid_step) surrounded by a structural feature;
        # never extrapolate feature colors radially into large open wall holes.
        near_feat = (d_feat <= max(2.4 * grid_step, 0.018)) & is_structural_feature[idx_feat]
        has_feat_nb = np.sum(near_feat, axis=1) >= 5
        if np.any(has_feat_nb):
            w_f = np.where(near_feat[has_feat_nb], 1.0 / np.maximum(d_feat[has_feat_nb], 1e-4), 0.0)[..., None]
            c_f = np.sum(cols_work[idx_feat[has_feat_nb]] * w_f, axis=1) / np.maximum(np.sum(w_f, axis=1), 1e-6)
            h_c[has_feat_nb] = 0.88 * c_f + 0.12 * h_c[has_feat_nb]
    h_c += rng.normal(0.0, 0.0045, size=h_c.shape)
    h_uv += rng.uniform(-0.24 * grid_step, 0.24 * grid_step, size=h_uv.shape)
    h_uv[:, 0] = np.clip(h_uv[:, 0], 0.0, span_u)
    h_uv[:, 1] = np.clip(h_uv[:, 1], 0.0, span_v)
    return np.clip(out_c, 0.0, 1.0), h_uv, np.clip(h_c, 0.0, 1.0)


def _detect_and_circularize_basins(
    pts: np.ndarray,
    cols: np.ndarray,
    w_r: float,
    d_r: float,
    rng: np.random.Generator | None = None,
    synthesize_bottom: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Deprecated: procedural basin circularization removed per [S5-1].
    Preserves genuine sensor points unchanged."""
    return pts, cols, np.zeros(len(pts), dtype=bool)


def _proper_manhattan_rotations() -> list[np.ndarray]:
    """Return all 24 proper rotations in SO(3) formed by signed permutations of axes (det = +1)."""
    import itertools

    rotations = []
    for p in itertools.permutations(range(3)):
        for signs in itertools.product([-1.0, 1.0], repeat=3):
            R = np.zeros((3, 3), dtype=float)
            for i in range(3):
                R[i, p[i]] = signs[i]
            if np.isclose(np.linalg.det(R), 1.0):
                rotations.append(R)
    return rotations


def _proper_yaw_rotations() -> list[np.ndarray]:
    """Return 4 proper 90-degree yaw rotations in SO(2) preserving +Z."""
    return [
        np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
        np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]),
        np.array([[-1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0]]),
        np.array([[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]),
    ]


def consolidate_inter_keyframe_surfels(
    pts: np.ndarray,
    cols: np.ndarray,
    frame_indices: np.ndarray | None = None,
    conf: np.ndarray | None = None,
    intra_voxel_m: float = 0.006,
    surfel_radius_m: float = 0.020,
    sigma_s_m: float = 0.010,
) -> tuple[np.ndarray, np.ndarray]:
    """Consolidate multi-keyframe surfels via per-keyframe 6mm voxel downsampling and
    cross-keyframe normal-projected weighted consensus within a 20mm radius (Module 3.2).
    """
    pts_arr = np.asarray(pts, dtype=np.float64)
    cols_arr = np.asarray(cols, dtype=np.float64)
    if len(pts_arr) < 100 or frame_indices is None or len(frame_indices) != len(pts_arr):
        return pts_arr, cols_arr

    fidx_arr = np.asarray(frame_indices, dtype=np.int32)
    if len(np.unique(fidx_arr)) < 2:
        return pts_arr, cols_arr

    conf_arr = (
        np.asarray(conf, dtype=np.float64)
        if (conf is not None and len(conf) == len(pts_arr))
        else np.full(len(pts_arr), 1.35, dtype=np.float64)
    )

    # 1. Per-keyframe 6mm voxel downsampling
    p_min = np.min(pts_arr, axis=0)
    q_ijk = np.floor((pts_arr - p_min) / max(float(intra_voxel_m), 1e-3)).astype(np.int64)
    v_hash = (
        q_ijk[:, 0] * 73856093
        ^ q_ijk[:, 1] * 19349663
        ^ q_ijk[:, 2] * 83492791
    )
    order = np.lexsort((fidx_arr, v_hash))
    vh_s = v_hash[order]
    fv_s = fidx_arr[order]
    pt_s = pts_arr[order]
    cl_s = cols_arr[order]
    cf_s = conf_arr[order]

    change = np.empty(len(vh_s), dtype=bool)
    change[0] = True
    change[1:] = (vh_s[1:] != vh_s[:-1]) | (fv_s[1:] != fv_s[:-1])
    starts = np.flatnonzero(change)
    counts = np.diff(np.append(starts, len(vh_s))).astype(np.float64)

    pts_ds = np.add.reduceat(pt_s, starts, axis=0) / counts[:, None]
    cols_ds = np.add.reduceat(cl_s, starts, axis=0) / counts[:, None]
    conf_ds = np.add.reduceat(cf_s, starts) / counts
    fidx_ds = fv_s[starts]

    if len(pts_ds) < 100:
        return pts_ds, cols_ds

    # 2. Estimate surface normals via Open3D hybrid KDTree search
    pcd_tmp = o3d.geometry.PointCloud()
    pcd_tmp.points = o3d.utility.Vector3dVector(pts_ds)
    pcd_tmp.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.14, max_nn=24)
    )
    nrms_ds = np.asarray(pcd_tmp.normals, dtype=np.float64)
    if len(nrms_ds) != len(pts_ds):
        return pts_ds, cols_ds

    # 3. Cross-keyframe normal-projected consensus within surfel_radius_m (20mm)
    from scipy.spatial import cKDTree

    tree = cKDTree(pts_ds)
    k_nn = 8
    dists, idxs = tree.query(pts_ds, k=k_nn, distance_upper_bound=float(surfel_radius_m))
    w_self = np.square(np.maximum(conf_ds - 1.0, 0.05))
    w_sum = w_self.copy()
    dn_sum = np.zeros(len(pts_ds), dtype=np.float64)
    col_sum = cols_ds * w_self[:, None]

    inv_2sig2 = 1.0 / (2.0 * (max(float(sigma_s_m), 1e-3) ** 2))
    cos_25 = float(np.cos(np.radians(25.0)))

    for k_i in range(1, k_nn):
        valid = np.isfinite(dists[:, k_i]) & (idxs[:, k_i] < len(pts_ds))
        if not np.any(valid):
            continue
        src = np.flatnonzero(valid)
        dst = idxs[src, k_i]
        # Cross-keyframe neighbors whose surface normals agree within 25 deg
        cross_kf = fidx_ds[src] != fidx_ds[dst]
        src = src[cross_kf]
        dst = dst[cross_kf]
        if len(src) == 0:
            continue
        ndot = np.abs(np.sum(nrms_ds[src] * nrms_ds[dst], axis=1))
        ok_n = ndot >= cos_25
        src = src[ok_n]
        dst = dst[ok_n]
        if len(src) == 0:
            continue
        diff_vec = pts_ds[dst] - pts_ds[src]
        d2 = np.sum(diff_vec * diff_vec, axis=1)
        w_ij = np.square(np.maximum(conf_ds[dst] - 1.0, 0.05)) * np.exp(-d2 * inv_2sig2)
        proj_n = np.sum(nrms_ds[src] * diff_vec, axis=1)
        w_sum[src] += w_ij
        dn_sum[src] += w_ij * proj_n
        col_sum[src] += w_ij[:, None] * cols_ds[dst]

    delta_d = np.clip(dn_sum / np.maximum(w_sum, 1e-9), -0.015, 0.015)
    pts_out = pts_ds + delta_d[:, None] * nrms_ds
    cols_out = np.clip(col_sum / np.maximum(w_sum[:, None], 1e-9), 0.0, 1.0)
    return pts_out, cols_out


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

    # Synchronize frame_indices and confidence when filtering unconfident VGGT collapse points (Module 4.1)
    fidx_raw = getattr(vggt_prior, "frame_indices", None)
    fidx_arr = (
        np.asarray(fidx_raw, dtype=np.int32)
        if (fidx_raw is not None and len(fidx_raw) == len(pts_vggt))
        else None
    )
    conf = getattr(vggt_prior, "confidence", None)
    conf_arr = None
    vggt_conf_low = False
    if conf is not None and len(conf) == len(pts_vggt) and len(conf) > 0:
        conf_arr = np.asarray(conf, dtype=float)
        conf_mask = np.isfinite(conf_arr) & (conf_arr > 1.05)
        conf_ratio = float(np.mean(conf_mask))
        if conf_ratio < 0.50:
            vggt_conf_low = True
        if int(np.count_nonzero(conf_mask)) >= 50:
            pts_vggt = pts_vggt[conf_mask]
            cols_arr = cols_arr[conf_mask]
            conf_arr = conf_arr[conf_mask]
            if fidx_arr is not None:
                fidx_arr = fidx_arr[conf_mask]

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
                c_vggt_chk = getattr(vggt_prior, "C_vggt", None)
                t_vio_chk = getattr(vggt_prior, "t_vio", None)
                if c_vggt_chk is not None and t_vio_chk is not None and len(c_vggt_chk) >= 3 and len(t_vio_chk) >= 3:
                    from app.services.vggt_prior import estimate_sim3_camera_centers

                    sim3_chk = estimate_sim3_camera_centers(c_vggt_chk, t_vio_chk)
                    diag_rmse = float(diag_vggt.get("sim3_rmse") or 1.0)
                    if (
                        sim3_chk.ok
                        and sim3_chk.rmse is not None
                        and sim3_chk.rmse <= 0.020
                        and sim3_chk.s is not None
                        and (
                            abs(float(sim3_chk.s) - s_use) > 0.06
                            or float(np.linalg.norm(np.asarray(sim3_chk.t, dtype=float).reshape(3) - t_use)) > 0.02
                            or diag_rmse > float(sim3_chk.rmse) + 0.015
                        )
                    ):
                        s_use = float(sim3_chk.s)
                        R_use = np.asarray(sim3_chk.R, dtype=float).reshape(3, 3)
                        t_use = np.asarray(sim3_chk.t, dtype=float).reshape(3)
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
            vggt_camera_centers=c_vggt,
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

    # 2. Produce pcd_room in canonical coordinates using clean rigid transformations
    from app.services.manhattan_alignment import (
        align_point_cloud_to_manhattan,
        clean_and_align_point_cloud,
        estimate_vggt_gravity_up,
        yaw_rotation_matrix,
    )
    from app.services.occupancy_layout import estimate_manhattan_angle_bbox_sharpness
    from shapely.vectorized import contains as contains_xy

    clean_ply = getattr(vggt_prior, "clean_ply_path", None)
    sess_dir = (
        os.path.dirname(os.path.dirname(clean_ply))
        if (clean_ply and os.path.basename(os.path.dirname(clean_ply)) == "vggt")
        else None
    )

    # When VGGT confidence collapsed across most frames on a 360-degree room scan (< 50% confident points),
    # do not use the collapsed VGGT cloud (or stale project_point_cloud_clean.ply) to overwrite the
    # full-room canonical visual cloud if a metric TSDF point cloud is available.
    if vggt_conf_low and isinstance(layout, dict) and ("level_frame" in layout or "floorplan_frame" in layout):
        tsdf_pcd = None
        if sess_dir:
            tsdf_ply = os.path.join(sess_dir, "reconstructed.ply")
            if os.path.isfile(tsdf_ply) and os.path.getsize(tsdf_ply) > 0:
                try:
                    cand_tsdf = o3d.io.read_point_cloud(tsdf_ply)
                    if len(cand_tsdf.points) >= 500:
                        tsdf_pcd = cand_tsdf
                except Exception:
                    tsdf_pcd = None
        if tsdf_pcd is None and reference_points is not None and len(reference_points) >= 500:
            tsdf_pcd = o3d.geometry.PointCloud()
            tsdf_pcd.points = o3d.utility.Vector3dVector(np.asarray(reference_points, dtype=float))
            tsdf_pcd.colors = o3d.utility.Vector3dVector(np.full((len(reference_points), 3), 0.80, dtype=float))
        if tsdf_pcd is not None:
            framed_tsdf = apply_layout_frames(tsdf_pcd, layout, gravity)
            cleaned_tsdf = remove_visual_outliers(framed_tsdf)
            rect_tsdf = _rectify_non_vggt_framed_cloud(cleaned_tsdf, layout)
            crop_tsdf = crop_to_layout_polygon(rect_tsdf, layout)
            pcd_room = denoise_planar_surfaces(crop_tsdf, layout=layout)
            if len(pcd_room.points) >= 500:
                return pcd_room, pcd_vio

    pcd_canon_vio = None
    if not vggt_conf_low and isinstance(layout, dict) and "level_frame" in layout and "floorplan_frame" in layout:
        try:
            pcd_canon_vio = apply_layout_frames(pcd_vio, layout, gravity)
            c_vggt_cv = getattr(vggt_prior, "C_vggt", None)
            t_vio_cv = getattr(vggt_prior, "t_vio", None)
            if (
                c_vggt_cv is not None
                and t_vio_cv is not None
                and len(c_vggt_cv) >= 3
                and len(t_vio_cv) >= 3
                and len(pcd_canon_vio.points) > 0
            ):
                from app.services.vggt_prior import estimate_sim3_camera_centers

                sim3_cv = estimate_sim3_camera_centers(c_vggt_cv, t_vio_cv)
                if sim3_cv.ok:
                    pts_cv_use = np.asarray(pcd_canon_vio.points, dtype=float).copy()
                    cols_cv_use = (
                        np.asarray(pcd_canon_vio.colors, dtype=float).copy()
                        if pcd_canon_vio.has_colors()
                        else cols_arr.copy()
                    )
                    if fidx_arr is not None and len(fidx_arr) == len(pts_cv_use):
                        pts_cv_use, cols_cv_use = consolidate_inter_keyframe_surfels(
                            pts_cv_use, cols_cv_use, fidx_arr, conf_arr
                        )
                    pcd_room_cv = o3d.geometry.PointCloud()
                    pcd_room_cv.points = o3d.utility.Vector3dVector(pts_cv_use)
                    pcd_room_cv.colors = o3d.utility.Vector3dVector(cols_cv_use)
                    if len(pcd_room_cv.points) > 650_000:
                        pcd_room_cv = pcd_room_cv.voxel_down_sample(voxel_size=0.002)
                    pcd_room_cv = remove_visual_outliers(pcd_room_cv)
                    pcd_room_cv = denoise_planar_surfaces(pcd_room_cv, layout=layout)
                    if len(pcd_room_cv.points) > 0:
                        return pcd_room_cv, pcd_vio
        except Exception:
            pcd_canon_vio = None

    imu_up = estimate_vggt_gravity_up(sess_dir) if sess_dir else None
    manhattan_pcd = None
    if not vggt_conf_low and clean_ply and os.path.isfile(clean_ply) and os.path.getsize(clean_ply) > 0:
        cand_pcd = o3d.io.read_point_cloud(clean_ply)
        if len(cand_pcd.points) >= 50:
            if imu_up is not None and len(cand_pcd.points) >= 200:
                pts_cand = np.asarray(cand_pcd.points, dtype=float)
                stale_tilt = False
                for axes_chk in ((0, 2), (1, 2)):
                    info_chk = estimate_manhattan_angle_bbox_sharpness(pts_cand[:, axes_chk])
                    if (
                        abs(float(info_chk.get("angle_deg", 0.0))) >= 2.0
                        and float(info_chk.get("bbox_area_improvement", 0.0)) >= 0.015
                    ):
                        stale_tilt = True
                        break
                if not stale_tilt:
                    manhattan_pcd = cand_pcd
            else:
                manhattan_pcd = cand_pcd

    if manhattan_pcd is None:
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
        cols_m = (
            np.asarray(manhattan_pcd.colors, dtype=float).copy()
            if manhattan_pcd.has_colors() and len(manhattan_pcd.colors) == len(pts_m)
            else None
        )

        if imu_up is None and len(pts_m) >= 200:
            has_3d_tilt = False
            for axes_chk in ((0, 1), (0, 2), (1, 2)):
                info_chk = estimate_manhattan_angle_bbox_sharpness(pts_m[:, axes_chk])
                if (
                    abs(float(info_chk.get("angle_deg", 0.0))) >= 0.5
                    and float(info_chk.get("bbox_area_improvement", 0.0)) >= 0.01
                ):
                    has_3d_tilt = True
                    break
            if has_3d_tilt:
                c_3d = np.median(pts_m, axis=0)
                plane_axes = {"xy": (0, 1), "xz": (0, 2), "yz": (1, 2)}

                def _level_box_order(order: tuple[str, str, str]) -> tuple[np.ndarray, float]:
                    p_cand = pts_m.copy()
                    for _ in range(3):
                        for pl in order:
                            a0, a1 = plane_axes[pl]
                            info_pl = estimate_manhattan_angle_bbox_sharpness(p_cand[:, (a0, a1)])
                            ang_pl = float(info_pl.get("angle_deg", 0.0))
                            impr_pl = float(info_pl.get("bbox_area_improvement", 0.0))
                            if abs(ang_pl) >= 0.2 and impr_pl >= 0.01:
                                rad_pl = np.radians(ang_pl)
                                co, si = float(np.cos(rad_pl)), float(np.sin(rad_pl))
                                R_pl = np.eye(3, dtype=float)
                                R_pl[a0, a0], R_pl[a0, a1] = co, -si
                                R_pl[a1, a0], R_pl[a1, a1] = si, co
                                p_cand = (p_cand - c_3d) @ R_pl.T + c_3d
                    sp_cand = np.percentile(p_cand, 98, axis=0) - np.percentile(p_cand, 2, axis=0)
                    vol_cand = float(np.prod(np.maximum(sp_cand, 1e-3)))
                    return p_cand, vol_cand

                p_o1, v_o1 = _level_box_order(("xz", "yz", "xy"))
                p_o2, v_o2 = _level_box_order(("xy", "xz", "yz"))
                pts_m = p_o1 if v_o1 <= v_o2 else p_o2

        metric_scale = float(s_use) if (s_use is not None and np.isfinite(s_use) and s_use > 0) else 1.0
        pts_m = pts_m * metric_scale
        xy_layout = layout_xy_polygon(layout) if isinstance(layout, dict) else np.zeros((0, 2))

        if len(xy_layout) >= 4:
            from shapely.geometry import box as shapely_box

            w_r = float(np.ptp(xy_layout[:, 0]))
            d_r = float(np.ptp(xy_layout[:, 1]))
            h_r = float(layout_height_m(layout))

            candidates = _proper_yaw_rotations() if imu_up is not None else _proper_manhattan_rotations()
            poly_raw = Polygon(xy_layout)
            poly_buffered = poly_raw.buffer(0.05)
            notch_poly = shapely_box(
                float(np.min(xy_layout[:, 0])),
                float(np.min(xy_layout[:, 1])),
                float(np.max(xy_layout[:, 0])),
                float(np.max(xy_layout[:, 1])),
            ).difference(poly_raw.buffer(0.025))
            has_notch = (not notch_poly.is_empty) and (float(notch_poly.area) > 0.02)

            sub_idx = np.linspace(0, len(pts_m) - 1, min(2000, len(pts_m)), dtype=int)
            pts_sub = pts_m[sub_idx]

            best_score = -1e9
            best_R = np.eye(3)

            for R_cand in candidates:
                P_cand = pts_sub @ R_cand.T
                p_min = np.percentile(P_cand, 2, axis=0)
                P_shifted = P_cand - p_min
                spans = np.percentile(P_shifted, 98, axis=0) - np.percentile(P_shifted, 2, axis=0)

                h_err = abs(spans[2] - h_r) / max(h_r, 1.0)
                w_err = abs(spans[0] - w_r) / max(w_r, 1.0)
                d_err = abs(spans[1] - d_r) / max(d_r, 1.0)

                try:
                    inside = contains_xy(poly_buffered, P_shifted[:, 0], P_shifted[:, 1])
                    containment = float(np.mean(inside))
                    notch_frac = (
                        float(np.mean(contains_xy(notch_poly, P_shifted[:, 0], P_shifted[:, 1])))
                        if has_notch
                        else 0.0
                    )
                except Exception:
                    containment = 0.5
                    notch_frac = 0.0

                score = (
                    containment * 2.0
                    - notch_frac * 8.0
                    - h_err
                    - 0.5 * (w_err + d_err)
                    - 0.001 * float(np.linalg.norm(R_cand - np.eye(3)))
                )
                if score > best_score:
                    best_score = score
                    best_R = R_cand

            pts_m = pts_m @ best_R.T

        # Shift to align origin: floor at z=0, min x, min y
        x_p02 = float(np.percentile(pts_m[:, 0], 2))
        y_p02 = float(np.percentile(pts_m[:, 1], 2))
        z_floor = float(np.percentile(pts_m[:, 2], 1.5))
        pts_m[:, 0] -= x_p02
        pts_m[:, 1] -= y_p02
        pts_m[:, 2] -= z_floor

        # Fine-tune residual yaw if walls slightly off-axis
        z_lo = float(np.percentile(pts_m[:, 2], 2))
        z_hi = float(np.percentile(pts_m[:, 2], 98))
        dz = z_hi - z_lo
        mid_mask = (pts_m[:, 2] >= z_lo + 0.15 * dz) & (pts_m[:, 2] <= z_lo + 0.85 * dz)
        slice_xy = pts_m[mid_mask, :2] if int(np.count_nonzero(mid_mask)) >= 200 else pts_m[:, :2]
        if len(slice_xy) >= 100:
            sharp_info = estimate_manhattan_angle_bbox_sharpness(slice_xy)
            residual_yaw = float(sharp_info.get("angle_deg", 0.0))
            impr = float(sharp_info.get("bbox_area_improvement", 0.0))
            if abs(residual_yaw) > 0.2 and impr >= 0.02:
                r_yaw = yaw_rotation_matrix(2, residual_yaw)
                c_xy = pts_m.mean(axis=0)
                pts_m = (pts_m - c_xy) @ r_yaw.T + c_xy
                pts_m[:, 0] -= float(np.percentile(pts_m[:, 0], 2))
                pts_m[:, 1] -= float(np.percentile(pts_m[:, 1], 2))
                pts_m[:, 2] -= float(np.percentile(pts_m[:, 2], 1.5))

        pcd_room = o3d.geometry.PointCloud()
        pcd_room.points = o3d.utility.Vector3dVector(pts_m)
        if cols_m is not None:
            pcd_room.colors = o3d.utility.Vector3dVector(cols_m)

        if len(pcd_room.points) > 50:
            pcd_room = remove_visual_outliers(pcd_room)
        pcd_room = denoise_planar_surfaces(pcd_room, layout=layout)
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

    # Module 5.3: 3*sigma (0.03m) polygon clipping to remove Poisson interpolation bubbles in non-convex L-notches
    if len(xy) >= 4 and len(mesh.vertices) > 0:
        try:
            from shapely.vectorized import contains as _contains_mesh_xy

            poly_clip = Polygon(xy)
            if not poly_clip.is_valid:
                poly_clip = poly_clip.buffer(0)
            if not poly_clip.is_empty:
                v_arr = np.asarray(mesh.vertices)
                inside_poly = _contains_mesh_xy(poly_clip.buffer(0.03), v_arr[:, 0], v_arr[:, 1])
                if int(np.count_nonzero(inside_poly)) >= 50:
                    mesh.remove_vertices_by_mask(~inside_poly)
        except Exception:
            pass

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


def _rectify_non_vggt_framed_cloud(
    pcd: o3d.geometry.PointCloud,
    layout: dict,
) -> o3d.geometry.PointCloud:
    """Clean framed visual point cloud using genuine geometric filtering and statistical outlier removal.
    Does NOT hallucinate synthetic fixtures or apply anisotropic coordinate warping."""
    work = o3d.geometry.PointCloud(pcd)
    if len(work.points) < 50 or not isinstance(layout, dict):
        return work
    cleaned = remove_visual_outliers(work)
    cropped = crop_to_layout_polygon(cleaned, layout)
    return cropped


def fuse_complementary_vggt_and_tsdf_clouds(
    pcd_vggt: o3d.geometry.PointCloud,
    pcd_tsdf: o3d.geometry.PointCloud,
    layout: dict | None = None,
    d_min: float = 0.015,
    d_max: float = 0.035,
) -> o3d.geometry.PointCloud:
    """Complementary fusion of primary VGGT point cloud and dToF-anchored TSDF point cloud (Module 5.2).

    Retains 100% of VGGT points and supplements genuine sensor-backed TSDF points only in
    under-sampled gaps (d_NN >= d_min) with smooth C0 boundary blending in [d_min, d_max).
    """
    if pcd_vggt is None or len(pcd_vggt.points) < 100:
        return pcd_tsdf if pcd_tsdf is not None else o3d.geometry.PointCloud()
    if pcd_tsdf is None or len(pcd_tsdf.points) < 100:
        return pcd_vggt

    pts_v = np.asarray(pcd_vggt.points, dtype=np.float64)
    cols_v = (
        np.asarray(pcd_vggt.colors, dtype=np.float64)
        if pcd_vggt.has_colors() and len(pcd_vggt.colors) == len(pts_v)
        else np.full((len(pts_v), 3), 0.82, dtype=np.float64)
    )
    pts_t = np.asarray(pcd_tsdf.points, dtype=np.float64)
    cols_t = (
        np.asarray(pcd_tsdf.colors, dtype=np.float64)
        if pcd_tsdf.has_colors() and len(pcd_tsdf.colors) == len(pts_t)
        else np.full((len(pts_t), 3), 0.80, dtype=np.float64)
    )

    from scipy.spatial import cKDTree

    tree_v = cKDTree(pts_v)
    d_nn, idx_nn = tree_v.query(pts_t, k=1)

    # Harmonize TSDF RGB gain at the boundary band [0.008m, 0.035m]
    bnd_mask = (d_nn >= 0.008) & (d_nn <= float(d_max))
    if int(np.count_nonzero(bnd_mask)) >= 50:
        c_v_bnd = np.clip(cols_v[idx_nn[bnd_mask]], 0.05, 0.95)
        c_t_bnd = np.clip(cols_t[bnd_mask], 0.05, 0.95)
        gain_t = np.clip(np.median(c_v_bnd / c_t_bnd, axis=0), 0.82, 1.22)
        cols_t = np.clip(cols_t * gain_t[None, :], 0.0, 1.0)

    # Gate TSDF gap candidates to genuine structural wall/floor bands or supported regions
    xy = layout_xy_polygon(layout) if isinstance(layout, dict) else np.zeros((0, 2))
    h_m = float(layout_height_m(layout)) if isinstance(layout, dict) else float(DEFAULT_HEIGHT_M)
    struct_ok = (pts_t[:, 2] <= 0.04) | (pts_t[:, 2] >= h_m - 0.04) | (d_nn <= 0.14)
    if len(xy) >= 4:
        for i_w in range(len(xy)):
            a_w = xy[i_w]
            b_w = xy[(i_w + 1) % len(xy)]
            dw = b_w - a_w
            lw = float(np.hypot(dw[0], dw[1]))
            if lw <= 0.05:
                continue
            tw = dw / lw
            nw = np.array([-tw[1], tw[0]], dtype=np.float64)
            rel = pts_t[:, :2] - a_w
            s_proj = rel @ tw
            d_perp = np.abs(rel @ nw)
            struct_ok |= (s_proj >= -0.02) & (s_proj <= lw + 0.02) & (d_perp <= 0.065)

    keep_gap = (d_nn >= float(d_min)) & struct_ok
    if not np.any(keep_gap):
        return pcd_vggt

    gap_idx = np.flatnonzero(keep_gap)
    d_gap = d_nn[gap_idx]
    nn_v = pts_v[idx_nn[gap_idx]]
    q_gap = pts_t[gap_idx].copy()
    c_gap = cols_t[gap_idx].copy()

    # Smooth C0 transition in [d_min, d_max)
    in_trans = d_gap < float(d_max)
    if np.any(in_trans):
        alpha = np.clip((d_gap[in_trans] - float(d_min)) / max(float(d_max - d_min), 1e-4), 0.0, 1.0)
        q_gap[in_trans] = (1.0 - alpha[:, None]) * nn_v[in_trans] + alpha[:, None] * q_gap[in_trans]
        c_gap[in_trans] = (1.0 - alpha[:, None]) * cols_v[idx_nn[gap_idx[in_trans]]] + alpha[:, None] * c_gap[in_trans]

    # Downsample TSDF gap supplement to 5mm grid so density matches VGGT without bloating point count
    p_ds = o3d.geometry.PointCloud()
    p_ds.points = o3d.utility.Vector3dVector(q_gap)
    p_ds.colors = o3d.utility.Vector3dVector(c_gap)
    p_ds = p_ds.voxel_down_sample(0.005)
    if len(p_ds.points) == 0:
        return pcd_vggt

    fused_pts = np.vstack([pts_v, np.asarray(p_ds.points, dtype=np.float64)])
    fused_cols = np.vstack([cols_v, np.asarray(p_ds.colors, dtype=np.float64)])
    if len(fused_pts) > VISUAL_POINT_CLOUD_MAX:
        fused_pts = fused_pts[:VISUAL_POINT_CLOUD_MAX]
        fused_cols = fused_cols[:VISUAL_POINT_CLOUD_MAX]
    out = o3d.geometry.PointCloud()
    out.points = o3d.utility.Vector3dVector(fused_pts)
    out.colors = o3d.utility.Vector3dVector(np.clip(fused_cols, 0.0, 1.0))
    return out


def build_visual_clouds(
    mesh: o3d.geometry.TriangleMesh,
    layout: dict,
    gravity: np.ndarray | None,
    number_of_points: int | None = None,
    vggt_prior=None,
) -> tuple[o3d.geometry.PointCloud, o3d.geometry.PointCloud]:
    empty = o3d.geometry.PointCloud()
    if vggt_prior is not None and hasattr(vggt_prior, "prior"):
        vggt_prior = vggt_prior.prior
    has_tsdf_mesh = False
    if mesh is not None:
        if isinstance(mesh, o3d.geometry.PointCloud):
            has_tsdf_mesh = len(mesh.points) >= 500
        elif hasattr(mesh, "vertices"):
            has_tsdf_mesh = len(mesh.vertices) >= 500
    vggt_conf_ok = True
    if vggt_prior is not None:
        conf_v = getattr(vggt_prior, "confidence", None)
        if conf_v is not None and len(conf_v) > 0:
            vggt_conf_ok = float(np.mean(np.asarray(conf_v, dtype=float) > 1.05)) >= 0.50
    has_canon_frames = isinstance(layout, dict) and ("level_frame" in layout and "floorplan_frame" in layout)
    canon_crop_margin = 0.04 if has_canon_frames else float(ROOM_CROP_MARGIN_M)
    if (
        vggt_prior is not None
        and hasattr(vggt_prior, "points")
        and len(vggt_prior.points) >= 50
        and (vggt_conf_ok or not has_tsdf_mesh)
    ):
        if isinstance(mesh, o3d.geometry.PointCloud):
            ref_pts = np.asarray(mesh.points) if len(mesh.points) else None
        else:
            ref_pts = np.asarray(mesh.vertices) if len(mesh.vertices) else None
        pcd_room, _ = get_metric_vggt_cloud(
            vggt_prior, layout, gravity, reference_points=ref_pts
        )
        if len(pcd_room.points) > 0:
            cleaned = pcd_room if len(pcd_room.points) >= 20_000 else remove_visual_outliers(pcd_room)
            # Module 5.2: Complementary fusion with dToF-anchored TSDF cloud when canonical frames exist
            if has_tsdf_mesh and has_canon_frames:
                try:
                    sampled_t = sample_visual_cloud(mesh, number_of_points=220_000)
                    if len(sampled_t.points) >= 500:
                        framed_t = apply_layout_frames(sampled_t, layout, gravity)
                        cleaned_t = remove_visual_outliers(framed_t)
                        cropped_t = crop_to_layout_polygon(cleaned_t, layout, margin_m=canon_crop_margin)
                        denoised_t = denoise_planar_surfaces(cropped_t, layout=layout)
                        cleaned = fuse_complementary_vggt_and_tsdf_clouds(
                            cleaned, denoised_t, layout=layout, d_min=0.015, d_max=0.035
                        )
                except Exception:
                    pass
            cropped = crop_to_layout_polygon(cleaned, layout, margin_m=canon_crop_margin)
            cropped.normals = o3d.utility.Vector3dVector()
            doll = cut_ceiling(cropped, layout_height_m(layout))
            doll.normals = o3d.utility.Vector3dVector()
            return cropped, doll

    sample_n = number_of_points
    if sample_n is not None:
        try:
            if int(sample_n) >= 50_000:
                sample_n = min(VISUAL_POINT_CLOUD_MAX, int(sample_n) * 2)
        except (TypeError, ValueError):
            pass
    sampled = sample_visual_cloud(mesh, number_of_points=sample_n)
    if len(sampled.points) == 0:
        return empty, empty
    framed = apply_layout_frames(sampled, layout, gravity)
    cleaned = remove_visual_outliers(framed)
    rectified = _rectify_non_vggt_framed_cloud(cleaned, layout)
    cropped = crop_to_layout_polygon(rectified, layout, margin_m=canon_crop_margin)
    cropped = denoise_planar_surfaces(cropped, layout=layout)
    if number_of_points is not None:
        try:
            raw_target = int(number_of_points)
            target_n = min(raw_target, VISUAL_POINT_CLOUD_MAX) if raw_target < 100_000 else VISUAL_POINT_CLOUD_MAX
            if len(cropped.points) > target_n >= 1:
                np.random.seed(42)
                keep_idx = np.random.choice(len(cropped.points), size=target_n, replace=False)
                pts_c = np.asarray(cropped.points)[keep_idx]
                cols_c = np.asarray(cropped.colors)[keep_idx] if cropped.has_colors() else None
                cropped = o3d.geometry.PointCloud()
                cropped.points = o3d.utility.Vector3dVector(pts_c)
                if cols_c is not None:
                    cropped.colors = o3d.utility.Vector3dVector(cols_c)
        except (TypeError, ValueError):
            pass
    cropped.normals = o3d.utility.Vector3dVector()
    doll = cut_ceiling(cropped, layout_height_m(layout))
    doll.normals = o3d.utility.Vector3dVector()
    return cropped, doll





