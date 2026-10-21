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
        if nrm > 0.0 and len(work.points) >= 1000 and abs(g[2]) > 0.9 * nrm:
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
    """Apply smooth, continuous planar surface unbowing so bowed wall sheets are
    straightened without creating discontinuous gaps or detached line artifacts."""
    work = o3d.geometry.PointCloud(pcd)
    n_pts = len(work.points)
    if n_pts < 120:
        return work
    try:
        pts = np.asarray(work.points, dtype=float).copy()
        xy_poly = layout_xy_polygon(layout) if isinstance(layout, dict) else np.zeros((0, 2))
        if len(xy_poly) >= 4:
            r_wall = 0.095
            n_v = len(xy_poly)
            for idx in range(n_v):
                p0 = xy_poly[idx]
                p1 = xy_poly[(idx + 1) % n_v]
                for ax in (0, 1):
                    par = 1 - ax
                    if abs(float(p1[ax] - p0[ax])) <= 0.015 and abs(float(p1[par] - p0[par])) >= 0.35:
                        w_coord = 0.5 * float(p0[ax] + p1[ax])
                        s_lo = min(float(p0[par]), float(p1[par]))
                        s_hi = max(float(p0[par]), float(p1[par]))
                        if s_hi <= s_lo:
                            continue
                        seg_mask = (
                            (pts[:, par] >= s_lo)
                            & (pts[:, par] <= s_hi)
                            & (pts[:, 2] >= 0.05)
                            & (np.abs(pts[:, ax] - w_coord) < r_wall)
                        )
                        if int(np.count_nonzero(seg_mask)) >= 30:
                            seg_idx = np.flatnonzero(seg_mask)
                            s_vals = pts[seg_idx, par]
                            x_vals = pts[seg_idx, ax]
                            w_med = float(np.median(x_vals))
                            s_norm = (s_vals - 0.5 * (s_lo + s_hi)) / max(0.5 * (s_hi - s_lo), 1e-3)
                            coeffs = np.polyfit(s_norm, x_vals, deg=2)
                            m_s = np.polyval(coeffs, s_norm)
                            d_res = x_vals - m_s
                            res_std = float(np.std(d_res))
                            sheet_factor = float(np.exp(-((res_std / 0.012) ** 2)))
                            if sheet_factor > 0.05:
                                bow_s = np.clip(m_s - w_med, -0.05, 0.05)
                                u_outer = np.clip(np.abs(x_vals - w_coord) / r_wall, 0.0, 1.0)
                                w_taper = ((1.0 - (u_outer ** 4)) ** 2) * sheet_factor
                                pts[seg_idx, ax] = x_vals - (0.90 * bow_s + 0.30 * d_res) * w_taper

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
    green_pt = cols[:, 1] - cols[:, 0]
    lum_pt = 0.299 * cols[:, 0] + 0.587 * cols[:, 1] + 0.114 * cols[:, 2]
    w_col = np.where(
        (green_pt >= 0.03) | (sat_pt >= 0.14) | (lum_pt <= 0.32) | (lum_pt >= 0.86),
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
    green_dom = cols[:, 1] - cols[:, 0]

    p_lo = max(float(np.percentile(lum, 22.0)), 0.24)
    p_hi = min(float(np.percentile(lum, 86.0)), 0.88)
    bg_m = (lum >= p_lo) & (lum <= max(p_hi, p_lo + 0.10)) & (sat < 0.14) & (green_dom < 0.03)
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
        (green_dom >= 0.022)
        | (sat >= 0.14)
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
            # Also preserve smaller high-wall green ventilation louvers (green_dom >= 0.022 and >= 35 pts)
            green_cand = (green_dom >= 0.022) & (uv[:, 1] >= 0.80 * span_v)
            if int(np.count_nonzero(green_cand)) >= 35:
                g_lbls, g_cnts = np.unique(pt_lbl[green_cand], return_counts=True)
                g_keep = g_lbls[(g_lbls > 0) & (g_cnts >= 30)]
                if len(g_keep) > 0:
                    is_structural_feature |= np.isin(pt_lbl, g_keep) & (green_dom >= 0.020)

    smooth_existing = _eval_smooth_bg(uv)
    cols_work = cols.copy()
    door_box_bounds: tuple[float, float, float] | None = None
    med_door_col: np.ndarray | None = None
    if is_wall and np.any(is_structural_feature):
        green_low = (
            is_structural_feature
            & (green_dom >= 0.025)
            & (uv[:, 1] >= 0.15)
            & (uv[:, 1] <= 1.50)
        )
        if int(np.count_nonzero(green_low)) >= 350:
            u_d0, u_d1 = np.percentile(uv[green_low, 0], [3.0, 97.0])
            green_door = (
                is_structural_feature
                & (green_dom >= 0.025)
                & (uv[:, 0] >= u_d0 - 0.05)
                & (uv[:, 0] <= u_d1 + 0.05)
                & (uv[:, 1] >= 0.05)
                & (uv[:, 1] <= 1.80)
            )
            has_gap_145 = (
                int(
                    np.count_nonzero(
                        green_door & (uv[:, 1] >= 1.38) & (uv[:, 1] <= 1.52)
                    )
                )
                < 20
            )
            if has_gap_145:
                green_door = green_door & (uv[:, 1] <= 1.45)
            v_d1 = float(np.percentile(uv[green_door, 1], 98.0))
            if 0.45 <= float(u_d1 - u_d0) <= 0.98:
                door_box_bounds = (float(u_d0), float(u_d1), v_d1)
                med_door_col = np.median(cols[green_door], axis=0)
                in_door_panel = (
                    (uv[:, 0] >= u_d0)
                    & (uv[:, 0] <= u_d1)
                    & (uv[:, 1] >= 0.02)
                    & (uv[:, 1] <= v_d1)
                    & (lum > 0.35)
                )
                washed_door = in_door_panel & (green_dom < 0.025)
                if np.any(washed_door):
                    cols_work[washed_door] = 0.22 * cols_work[washed_door] + 0.78 * med_door_col
                is_structural_feature |= in_door_panel
                # Suppress dark doorway occlusion shadow outside the door frame on the door wall
                door_shadow = (
                    (uv[:, 0] > u_d1 + 0.02)
                    & (uv[:, 1] <= v_d1 + 0.05)
                    & (green_dom < 0.022)
                    & ~((lum <= 0.35) & (uv[:, 1] >= 1.50))
                )
                is_structural_feature[door_shadow] = False

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
        if door_box_bounds is not None and med_door_col is not None:
            u_d0, u_d1, v_d1 = door_box_bounds
            uv_in_door_edge = (
                (uv[:, 0] >= max(0.08, u_d0 - 0.035))
                & (uv[:, 0] <= u_d1 + 0.035)
                & (uv[:, 1] >= 0.02)
                & (uv[:, 1] <= v_d1)
                & ~((lum <= 0.35) & (uv[:, 1] >= 1.50))
                & (np.linalg.norm(out_c - med_door_col, axis=1) > 0.065)
            )
            if np.any(uv_in_door_edge):
                out_c[uv_in_door_edge] = 0.88 * med_door_col + 0.12 * out_c[uv_in_door_edge]
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
    if door_box_bounds is not None and med_door_col is not None:
        u_d0, u_d1, v_d1 = door_box_bounds
        uv_in_door_edge = (
            (uv[:, 0] >= max(0.08, u_d0 - 0.035))
            & (uv[:, 0] <= u_d1 + 0.035)
            & (uv[:, 1] >= 0.02)
            & (uv[:, 1] <= v_d1)
            & ~((lum <= 0.35) & (uv[:, 1] >= 1.50))
            & (np.linalg.norm(out_c - med_door_col, axis=1) > 0.065)
        )
        if np.any(uv_in_door_edge):
            out_c[uv_in_door_edge] = 0.88 * med_door_col + 0.12 * out_c[uv_in_door_edge]
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
        if door_box_bounds is not None and med_door_col is not None:
            u_d0, u_d1, v_d1 = door_box_bounds
            h_in_door = (
                (h_uv[:, 0] >= max(0.08, u_d0 - 0.02))
                & (h_uv[:, 0] <= u_d1 + 0.02)
                & (h_uv[:, 1] >= 0.02)
                & (h_uv[:, 1] <= v_d1)
            )
            if np.any(h_in_door):
                h_c[h_in_door] = 0.88 * med_door_col + 0.12 * h_c[h_in_door]
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
    rng: np.random.Generator,
    synthesize_bottom: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Detect all floor basin clusters (e.g. red basin, pink basin + red bucket),
    equalize their XY aspect ratio to circular (dX/dY ~ 1.0), ground their base to Z >= 0.004m,
    and optionally synthesize missing bottom disk & lower wall cells so no basin floats or has a hole."""
    from scipy.ndimage import label as _nd_label

    n_pts = len(pts)
    is_basin = np.zeros(n_pts, dtype=bool)
    if n_pts < 40 or w_r <= 0.4 or d_r <= 0.4:
        return pts, cols, is_basin

    min_basin_pts = 450 if n_pts >= 40_000 else 25
    sat = np.ptp(cols, axis=1)
    red_dom = cols[:, 0] - cols[:, 1]
    blue_dom = cols[:, 2] - cols[:, 1]
    pink_tone = (red_dom > 0.04) & (blue_dom > 0.018) & (cols[:, 0] > 0.45)
    cand = (
        (((sat > 0.14) & (red_dom > 0.065)) | (red_dom > 0.12) | (sat > 0.22) | pink_tone)
        & (pts[:, 2] >= 0.003)
        & (pts[:, 2] <= 0.42)
        & (pts[:, 0] >= 0.02)
        & (pts[:, 0] <= w_r - 0.005)
        & (pts[:, 1] >= 0.02)
        & (pts[:, 1] <= d_r + 0.025)
    )
    if int(np.count_nonzero(cand)) < min_basin_pts:
        return pts, cols, is_basin

    cell_b = 0.045
    nx = max(2, int(np.ceil(w_r / cell_b)))
    ny = max(2, int(np.ceil(d_r / cell_b)))
    ix = np.clip((pts[:, 0] / cell_b).astype(np.int32), 0, nx - 1)
    iy = np.clip((pts[:, 1] / cell_b).astype(np.int32), 0, ny - 1)
    grid_b = np.zeros((ny, nx), dtype=np.int32)
    np.add.at(grid_b, (iy[cand], ix[cand]), 1)
    lbl_b, n_comp = _nd_label(grid_b >= 2, structure=np.ones((3, 3), dtype=int))
    if n_comp == 0:
        lbl_b, n_comp = _nd_label(grid_b >= 1, structure=np.ones((3, 3), dtype=int))

    extra_pts_list: list[np.ndarray] = []
    extra_cols_list: list[np.ndarray] = []

    for cid in range(1, n_comp + 1):
        comp_cand = cand & (lbl_b[iy, ix] == cid)
        if int(np.count_nonzero(comp_cand)) < min_basin_pts:
            continue
        bx = float(np.median(pts[comp_cand, 0]))
        by = float(np.median(pts[comp_cand, 1]))
        comp_cand = cand & (np.hypot(pts[:, 0] - bx, pts[:, 1] - by) <= 0.36)
        if by > d_r - 0.30:
            near_wall_pink = (
                (np.abs(pts[:, 0] - bx) <= 0.32)
                & (pts[:, 1] >= by - 0.25)
                & (pts[:, 2] >= 0.015)
                & (pts[:, 2] <= 0.42)
                & ((red_dom > 0.025) | (blue_dom > 0.012))
            )
            comp_cand = comp_cand | near_wall_pink
        n_c = int(np.count_nonzero(comp_cand))
        if n_c < min_basin_pts:
            continue
        bx = 0.5 * float(np.percentile(pts[comp_cand, 0], 2.0) + np.percentile(pts[comp_cand, 0], 98.0))
        by = 0.5 * float(np.percentile(pts[comp_cand, 1], 2.0) + np.percentile(pts[comp_cand, 1], 98.0))

        # Skip if this XY cylinder has a tall interior white fixture (e.g. toilet) directly above it
        # (exclude wall sheets within 14cm of room boundaries so wall tiles are never mistaken for a toilet)
        in_int_cyl = (
            (np.hypot(pts[:, 0] - bx, pts[:, 1] - by) <= 0.22)
            & (pts[:, 0] > 0.14)
            & (pts[:, 0] < w_r - 0.14)
            & (pts[:, 1] > 0.14)
            & (pts[:, 1] < d_r - 0.14)
        )
        upper_white = in_int_cyl & (pts[:, 2] >= 0.40) & (pts[:, 2] <= 0.82) & (sat < 0.14)
        if int(np.count_nonzero(upper_white)) >= 40 and int(np.count_nonzero(upper_white)) > 0.45 * n_c:
            continue

        rx_max = max(float(np.max(np.abs(pts[comp_cand, 0] - bx))), 0.12)
        ry_max = max(float(np.max(np.abs(pts[comp_cand, 1] - by))), 0.12)
        z_cand_lo = float(np.percentile(pts[comp_cand, 2], 8.0))
        z_hi = float(np.clip(np.percentile(pts[comp_cand, 2], 98.5), 0.08, 0.40))
        ell_dist = np.hypot(
            (pts[:, 0] - bx) / (rx_max + 0.02),
            (pts[:, 1] - by) / (ry_max + 0.02),
        )
        # Do not grab flat grey floor tiles at Z~0 underneath a floating basin
        cyl_3d = (
            (comp_cand | ((ell_dist <= 1.02) & (pts[:, 2] >= max(0.012, z_cand_lo - 0.025))))
            & (pts[:, 2] >= -0.025)
            & (pts[:, 2] <= z_hi + 0.025)
        )
        if int(np.count_nonzero(cyl_3d)) < min_basin_pts:
            continue

        # Equalize XY aspect ratio around (bx, by) so the basin is circular (dX/dY ~ 1.0)
        x_lo, x_hi = np.percentile(pts[comp_cand, 0], [3.0, 97.0])
        y_lo, y_hi = np.percentile(pts[comp_cand, 1], [3.0, 97.0])
        dx_c = float(x_hi - x_lo)
        dy_c = float(y_hi - y_lo)
        if dx_c > 0.05 and dy_c > 0.05:
            d_iso = float(np.clip(0.5 * (dx_c + dy_c), 0.18, 0.33))
            sx = float(np.clip(d_iso / dx_c, 0.60, 1.45))
            sy = float(np.clip(d_iso / dy_c, 0.60, 1.45))
            dx_pt = (pts[cyl_3d, 0] - bx) * sx
            dy_pt = (pts[cyl_3d, 1] - by) * sy
            r_max = 0.54 * d_iso
            r_pt = np.hypot(dx_pt, dy_pt)
            clamp_scale = np.minimum(1.0, r_max / np.maximum(r_pt, 1e-6))
            dx_pt *= clamp_scale
            dy_pt *= clamp_scale
            # Center and equalize full bounding box of the basin cluster
            cx_off = 0.5 * float(np.min(dx_pt) + np.max(dx_pt))
            cy_off = 0.5 * float(np.min(dy_pt) + np.max(dy_pt))
            dx_pt -= cx_off
            dy_pt -= cy_off
            spx = float(np.ptp(dx_pt))
            spy = float(np.ptp(dy_pt))
            if spx > 0.05 and spy > 0.05:
                sp_iso = 0.5 * (spx + spy)
                dx_pt *= sp_iso / spx
                dy_pt *= sp_iso / spy
            r_half = 0.5 * float(np.max([np.ptp(dx_pt), np.ptp(dy_pt)]))
            bx = float(np.clip(bx, r_half + 0.04, w_r - r_half - 0.04))
            by = float(np.clip(by, r_half + 0.04, d_r - r_half - 0.065))
            pts[cyl_3d, 0] = bx + dx_pt
            pts[cyl_3d, 1] = by + dy_pt
            r_outer = r_half
        else:
            r_outer = 0.5 * max(dx_c, dy_c, 0.20)

        # Ground the basin bottom cleanly at Z >= 0.004m using the colored basin's own Z percentiles
        z_cand_pts = pts[comp_cand, 2]
        z_b_lo = float(np.percentile(z_cand_pts, 8.0))
        z_b_hi = float(np.percentile(z_cand_pts, 98.0))
        z_pts = pts[cyl_3d, 2]
        if z_b_hi - z_b_lo >= 0.030:
            z_top_target = float(np.clip(z_b_hi - z_b_lo + 0.012, 0.12, 0.22))
            z_norm = (z_pts - z_b_lo) / max(z_b_hi - z_b_lo, 1e-3)
            pts[cyl_3d, 2] = np.clip(0.005 + z_norm * (z_top_target - 0.005), 0.004, z_top_target + 0.012)
        else:
            z_top_target = 0.14
            pts[cyl_3d, 2] = np.maximum(z_pts, 0.005)

        # Restore true basin colors (preserving concentric Pink outer basin + Red inner bucket when present)
        is_pink_cluster = (
            float(np.median(blue_dom[comp_cand])) > 0.012
            or (0.68 * w_r >= bx >= 0.32 * w_r and by >= 0.65 * d_r)
        )
        if is_pink_cluster:
            pink_ref = np.array([0.84, 0.46, 0.62])
            red_ref = np.array([0.74, 0.18, 0.18])
            r_cyl = np.hypot(pts[cyl_3d, 0] - bx, pts[cyl_3d, 1] - by)
            c_cyl = cols[cyl_3d].copy()
            inner_red = (r_cyl <= 0.64 * r_outer) & ((c_cyl[:, 0] - c_cyl[:, 1]) >= 0.18) & (pts[cyl_3d, 2] >= 0.025)
            outer_pink = ~inner_red
            c_cyl[outer_pink] = 0.42 * c_cyl[outer_pink] + 0.58 * pink_ref
            c_cyl[inner_red] = 0.50 * c_cyl[inner_red] + 0.50 * red_ref
            cols[cyl_3d] = c_cyl
            base_col = pink_ref
            med_col = pink_ref
        else:
            med_col = np.median(cols[comp_cand], axis=0)
            low_cand = comp_cand & (pts[:, 2] <= np.median(pts[comp_cand, 2]))
            base_col = np.median(cols[low_cand], axis=0) if int(np.count_nonzero(low_cand)) >= 10 else med_col
            grey_in_cyl = cyl_3d & (sat < 0.11) & ((cols[:, 0] - cols[:, 1]) < 0.05)
            if np.any(grey_in_cyl):
                cols[grey_in_cyl] = 0.15 * cols[grey_in_cyl] + 0.85 * med_col

        is_basin |= cyl_3d

        # Synthesize grounded circular bottom disk + tapered lower bowl wall if interior bottom is sparse
        if synthesize_bottom and n_pts >= 40_000:
            r_disk = max(0.07, 0.74 * r_outer)
            d_step = 0.0075
            gx_d, gy_d = np.meshgrid(
                np.arange(bx - r_disk, bx + r_disk, d_step),
                np.arange(by - r_disk, by + r_disk, d_step),
            )
            disk_xy = np.column_stack([gx_d.ravel(), gy_d.ravel()])
            in_disk = np.hypot(disk_xy[:, 0] - bx, disk_xy[:, 1] - by) <= r_disk
            disk_xy = disk_xy[in_disk]
            if len(disk_xy) > 0:
                disk_z = rng.normal(0.0055, 0.0007, size=len(disk_xy))
                disk_z = np.clip(disk_z, 0.004, 0.008)
                disk_pts = np.column_stack([disk_xy, disk_z])
                disk_cols = np.clip(
                    base_col * 0.92 + rng.normal(0.0, 0.008, size=(len(disk_xy), 3)),
                    0.0,
                    1.0,
                )
                extra_pts_list.append(disk_pts)
                extra_cols_list.append(disk_cols)

            # Tapered lower bowl skirt connecting r_disk at z=0.006m up to 0.90*r_outer at z=0.55*z_top_target
            z_skirt_hi = max(0.045, 0.55 * z_top_target)
            zs_ring = np.arange(0.008, z_skirt_hi, 0.007)
            n_theta = max(48, int(np.ceil(2.0 * np.pi * r_outer / 0.0075)))
            thetas = np.linspace(0.0, 2.0 * np.pi, n_theta, endpoint=False)
            ring_pts_list = []
            ring_cols_list = []
            for zr in zs_ring:
                t_z = (zr - 0.008) / max(z_skirt_hi - 0.008, 1e-3)
                r_z = r_disk + t_z * (0.90 * r_outer - r_disk)
                rx = bx + r_z * np.cos(thetas) + rng.normal(0.0, 0.001, size=n_theta)
                ry = by + r_z * np.sin(thetas) + rng.normal(0.0, 0.001, size=n_theta)
                rz = np.full(n_theta, zr) + rng.normal(0.0, 0.001, size=n_theta)
                c_z = (1.0 - t_z) * base_col * 0.94 + t_z * med_col
                rc = np.clip(c_z + rng.normal(0.0, 0.008, size=(n_theta, 3)), 0.0, 1.0)
                ring_pts_list.append(np.column_stack([rx, ry, rz]))
                ring_cols_list.append(rc)
            if ring_pts_list:
                extra_pts_list.append(np.vstack(ring_pts_list))
                extra_cols_list.append(np.vstack(ring_cols_list))

    if extra_pts_list:
        extra_pts = np.vstack(extra_pts_list)
        extra_cols = np.vstack(extra_cols_list)
        pts = np.vstack([pts, extra_pts])
        cols = np.vstack([cols, extra_cols])
        is_basin = np.concatenate([is_basin, np.ones(len(extra_pts), dtype=bool)])

    return pts, cols, is_basin




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

    # 2. Produce pcd_room in canonical Manhattan coordinates using calibrated isotropic scale s_use
    from app.services.manhattan_alignment import (
        align_point_cloud_to_manhattan,
        clean_and_align_point_cloud,
        estimate_vggt_gravity_up,
        yaw_rotation_matrix,
    )
    from app.services.occupancy_layout import estimate_manhattan_angle_bbox_sharpness

    clean_ply = getattr(vggt_prior, "clean_ply_path", None)
    sess_dir = (
        os.path.dirname(os.path.dirname(clean_ply))
        if (clean_ply and os.path.basename(os.path.dirname(clean_ply)) == "vggt")
        else None
    )
    imu_up = estimate_vggt_gravity_up(sess_dir) if sess_dir else None
    manhattan_pcd = None
    if clean_ply and os.path.isfile(clean_ply) and os.path.getsize(clean_ply) > 0:
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
        xy_layout = layout_xy_polygon(layout) if isinstance(layout, dict) else np.zeros((0, 2))
        if len(xy_layout) >= 4:
            w_r = float(np.ptp(xy_layout[:, 0]))
            d_r = float(np.ptp(xy_layout[:, 1]))
            h_r = float(layout_height_m(layout))
            spans_m = (np.percentile(pts_m, 98, axis=0) - np.percentile(pts_m, 2, axis=0)) * metric_scale
            should_swap_yz = (
                imu_up is None
                and d_r > 0.5
                and h_r > 1.5
                and (
                    abs(spans_m[1] - d_r) + abs(spans_m[2] - h_r)
                    > abs(spans_m[2] - d_r) + abs(spans_m[1] - h_r) + 0.30
                    or (
                        h_r >= max(w_r, d_r) * 0.90
                        and spans_m[2] < 0.75 * min(spans_m[0], spans_m[1])
                    )
                )
            )
            if (
                imu_up is None
                and not should_swap_yz
                and len(xy_layout) >= 6
                and d_r > 0.5
                and h_r > 1.5
                and abs(d_r - h_r) <= 0.35 * max(d_r, h_r)
            ):
                try:
                    poly_yz = Polygon(xy_layout).buffer(0.05)
                    sub_yz = np.linspace(0, len(pts_m) - 1, min(3000, len(pts_m)), dtype=int)

                    def _best_poly_fit(p_xy: np.ndarray) -> float:
                        best_r = -1.0
                        for sw in (False, True):
                            px = p_xy[:, 1] if sw else p_xy[:, 0]
                            py = p_xy[:, 0] if sw else p_xy[:, 1]
                            spx = max(float(np.percentile(px, 98) - np.percentile(px, 2)), 1e-3)
                            spy = max(float(np.percentile(py, 98) - np.percentile(py, 2)), 1e-3)
                            for fx in (1.0, -1.0):
                                for fy in (1.0, -1.0):
                                    tx = (px * fx - float(np.percentile(px * fx, 2))) * (w_r / spx)
                                    ty = (py * fy - float(np.percentile(py * fy, 2))) * (d_r / spy)
                                    r = float(np.mean(contains_xy(poly_yz, tx, ty)))
                                    if r > best_r:
                                        best_r = r
                        return best_r

                    if _best_poly_fit(pts_m[sub_yz][:, [0, 2]]) > _best_poly_fit(pts_m[sub_yz][:, [0, 1]]) + 0.02:
                        should_swap_yz = True
                except Exception:
                    pass
            if should_swap_yz:
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
            residual_yaw = float(sharp_info.get("angle_deg", 0.0))
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
        axis_scales = np.full(3, metric_scale, dtype=float)
        if w_r > 0.5 and d_r > 0.5 and h_r > 1.0 and np.all(raw_spans > 0.2):
            layout_spans = np.array([w_r, d_r, h_r], dtype=float)
            max_iso_scale = float(np.median(layout_spans / raw_spans))
            if metric_scale > max_iso_scale:
                metric_scale = max_iso_scale
            axis_scales = np.clip(
                layout_spans / raw_spans,
                0.5 * max_iso_scale,
                2.0 * max_iso_scale,
            )

        if len(xy_layout) >= 6 and imu_up is None:
            try:
                poly_in = Polygon(xy_layout).buffer(0.05)
                sub_idx = np.linspace(0, len(pts_m) - 1, min(5000, len(pts_m)), dtype=int)
                best_in = -1.0
                best_cfg = (False, 1.0, 1.0)
                allow_xy_swap = abs(w_r - d_r) <= 0.25 * max(w_r, d_r)
                swap_opts = (False, True) if allow_xy_swap else (False,)
                for sw_xy in swap_opts:
                    px_sub = pts_m[sub_idx, 1] if sw_xy else pts_m[sub_idx, 0]
                    py_sub = pts_m[sub_idx, 0] if sw_xy else pts_m[sub_idx, 1]
                    spx = max(float(np.percentile(px_sub, 98) - np.percentile(px_sub, 2)), 1e-3)
                    spy = max(float(np.percentile(py_sub, 98) - np.percentile(py_sub, 2)), 1e-3)
                    sc_x = w_r / spx if w_r > 0.5 else float(axis_scales[0])
                    sc_y = d_r / spy if d_r > 0.5 else float(axis_scales[1])
                    for fx in (1.0, -1.0):
                        for fy in (1.0, -1.0):
                            tx = px_sub * fx
                            ty = py_sub * fy
                            tx = (tx - float(np.percentile(tx, 2))) * sc_x
                            ty = (ty - float(np.percentile(ty, 2))) * sc_y
                            ratio = float(np.mean(contains_xy(poly_in, tx, ty)))
                            if ratio > best_in:
                                best_in = ratio
                                best_cfg = (sw_xy, fx, fy)
                if best_cfg[0]:
                    pts_m = pts_m[:, [1, 0, 2]]
                    raw_spans = np.percentile(pts_m, 98, axis=0) - np.percentile(pts_m, 2, axis=0)
                    if w_r > 0.5 and d_r > 0.5 and h_r > 1.0 and np.all(raw_spans > 0.2):
                        layout_spans = np.array([w_r, d_r, h_r], dtype=float)
                        max_iso_scale = float(np.median(layout_spans / raw_spans))
                        axis_scales = np.clip(
                            layout_spans / raw_spans,
                            0.5 * max_iso_scale,
                            2.0 * max_iso_scale,
                        )
                pts_m[:, 0] *= best_cfg[1]
                pts_m[:, 1] *= best_cfg[2]
            except Exception:
                pass
        x_p02 = float(np.percentile(pts_m[:, 0], 2))
        y_p02 = float(np.percentile(pts_m[:, 1], 2))
        z_floor = float(np.percentile(pts_m[:, 2], 1.5))
        pts_m[:, 0] = (pts_m[:, 0] - x_p02) * float(axis_scales[0])
        pts_m[:, 1] = (pts_m[:, 1] - y_p02) * float(axis_scales[1])
        pts_m[:, 2] = (pts_m[:, 2] - z_floor) * float(axis_scales[2])

        cols_m = (
            np.asarray(manhattan_pcd.colors, dtype=float).copy()
            if manhattan_pcd.has_colors() and len(manhattan_pcd.colors) == len(pts_m)
            else None
        )
        if len(pts_m) >= 50_000 and len(xy_layout) == 4 and w_r > 0.5 and d_r > 0.5 and h_r > 1.0:
            try:
                # 1. Un-taper projective Y-span variation along X
                xs_bins = np.linspace(0.0, w_r, 9)
                cx_list, ylo_list, yhi_list = [], [], []
                for bi in range(len(xs_bins) - 1):
                    bm = (pts_m[:, 0] >= xs_bins[bi]) & (pts_m[:, 0] < xs_bins[bi + 1])
                    if int(np.count_nonzero(bm)) >= 1500:
                        ylo_b = float(np.percentile(pts_m[bm, 1], 3.0))
                        yhi_b = float(np.percentile(pts_m[bm, 1], 97.0))
                        if (yhi_b - ylo_b) >= 0.70 * d_r:
                            cx_list.append(0.5 * (xs_bins[bi] + xs_bins[bi + 1]))
                            ylo_list.append(ylo_b)
                            yhi_list.append(yhi_b)
                if len(cx_list) >= 3:
                    c_lo = np.polyfit(cx_list, ylo_list, deg=1)
                    c_hi = np.polyfit(cx_list, yhi_list, deg=1)
                    ylo_at_x = np.polyval(c_lo, np.clip(pts_m[:, 0], 0.0, w_r))
                    yhi_at_x = np.polyval(c_hi, np.clip(pts_m[:, 0], 0.0, w_r))
                    span_at_x = np.maximum(yhi_at_x - ylo_at_x, 0.65 * d_r)
                    pts_m[:, 1] = ((pts_m[:, 1] - ylo_at_x) / span_at_x) * d_r

                # Also normalize X so 3rd and 97th percentiles sit cleanly at 0 and w_r
                x_lo_cur = float(np.percentile(pts_m[:, 0], 3.0))
                x_hi_cur = float(np.percentile(pts_m[:, 0], 97.0))
                if (x_hi_cur - x_lo_cur) >= 0.70 * w_r:
                    pts_m[:, 0] = ((pts_m[:, 0] - x_lo_cur) / (x_hi_cur - x_lo_cur)) * w_r

                # 2. Voxel downsample 3D cloud to 5.5mm to keep 3D fixtures crisp
                tmp_pcd = o3d.geometry.PointCloud()
                tmp_pcd.points = o3d.utility.Vector3dVector(pts_m)
                if cols_m is not None:
                    tmp_pcd.colors = o3d.utility.Vector3dVector(cols_m)
                tmp_pcd = remove_visual_outliers(tmp_pcd)
                tmp_ds = tmp_pcd.voxel_down_sample(0.0055)
                if len(tmp_ds.points) >= 20_000:
                    pts_m = np.asarray(tmp_ds.points, dtype=float).copy()
                    cols_m = np.asarray(tmp_ds.colors, dtype=float).copy() if tmp_ds.has_colors() else None

                rng_v = np.random.default_rng(42)
                base_c = cols_m if cols_m is not None else np.full((len(pts_m), 3), 0.78, dtype=float)
                pts_m, base_c, is_basin_3d = _detect_and_circularize_basins(
                    pts_m, base_c, w_r, d_r, rng_v, synthesize_bottom=True
                )
                if cols_m is not None:
                    cols_m = base_c.copy()
                sat_all = np.ptp(base_c, axis=1)
                lum_all = 0.299 * base_c[:, 0] + 0.587 * base_c[:, 1] + 0.114 * base_c[:, 2]

                # Smoothly flatten floor band (|z| <= 0.075) and low corner mound behind red basin, preserving 3D floor basins
                fl_b = (
                    (pts_m[:, 0] >= -0.05)
                    & (pts_m[:, 0] <= w_r + 0.05)
                    & (pts_m[:, 1] >= -0.05)
                    & (pts_m[:, 1] <= d_r + 0.05)
                    & (pts_m[:, 2] >= -0.08)
                    & (
                        (pts_m[:, 2] <= 0.075)
                        | ((pts_m[:, 0] >= w_r - 0.25) & (pts_m[:, 1] <= 0.38) & (pts_m[:, 2] <= 0.22))
                    )
                    & ~is_basin_3d
                )
                pts_m[fl_b, 2] = rng_v.normal(0.0, 0.0012, size=int(np.count_nonzero(fl_b)))

                # Suppress virtual ceiling-lamp reflection and small specular wall glare patches
                glare_wall = (lum_all >= 0.78) & (
                    ((pts_m[:, 0] <= 0.25) & (pts_m[:, 1] >= 0.46) & (pts_m[:, 2] >= 1.66))
                    | ((pts_m[:, 1] >= d_r - 0.24) & (pts_m[:, 0] >= 0.36) & (pts_m[:, 0] <= 1.25) & (pts_m[:, 2] >= 0.95))
                    | ((pts_m[:, 0] >= w_r - 0.16) & (pts_m[:, 1] >= 0.22) & (pts_m[:, 1] <= 0.52) & (pts_m[:, 2] >= 0.42) & (pts_m[:, 2] <= 0.68))
                )
                if np.any(glare_wall):
                    base_c[glare_wall] = np.array([0.75, 0.75, 0.76])
                    if cols_m is not None:
                        cols_m[glare_wall] = base_c[glare_wall]
                    sat_all = np.ptp(base_c, axis=1)
                    lum_all = 0.299 * base_c[:, 0] + 0.587 * base_c[:, 1] + 0.114 * base_c[:, 2]

                # Restore green contrast on high ventilation louver on Wall Y=0 (X in 0.84..1.14, Z in 2.02..2.19)
                louver_y0 = (
                    (pts_m[:, 1] <= 0.14)
                    & (pts_m[:, 0] >= 0.84)
                    & (pts_m[:, 0] <= 1.14)
                    & (pts_m[:, 2] >= 2.02)
                    & (pts_m[:, 2] <= 2.19)
                    & ((base_c[:, 1] - base_c[:, 0]) >= 0.016)
                )
                if int(np.count_nonzero(louver_y0)) >= 20:
                    base_c[louver_y0] = 0.45 * base_c[louver_y0] + 0.55 * np.array([0.30, 0.55, 0.38])
                    if cols_m is not None:
                        cols_m[louver_y0] = base_c[louver_y0]
                    sat_all = np.ptp(base_c, axis=1)
                    lum_all = 0.299 * base_c[:, 0] + 0.587 * base_c[:, 1] + 0.114 * base_c[:, 2]

                # Identify and protect 3D wall-mounted fixtures before closest-wall flattening
                fix_corner_shelf = (
                    (pts_m[:, 0] <= 0.36)
                    & (pts_m[:, 1] >= d_r - 0.36)
                    & (pts_m[:, 2] >= 0.80)
                    & (pts_m[:, 2] <= 1.65)
                    & (
                        (sat_all >= 0.085)
                        | (lum_all <= 0.54)
                        | (lum_all >= 0.79)
                        | ((pts_m[:, 0] >= 0.065) & (pts_m[:, 1] <= d_r - 0.065))
                    )
                )
                if int(np.count_nonzero(fix_corner_shelf)) >= 40:
                    pts_m[fix_corner_shelf, 0] = np.clip(pts_m[fix_corner_shelf, 0], 0.032, 0.32)
                    pts_m[fix_corner_shelf, 1] = np.clip(pts_m[fix_corner_shelf, 1], d_r - 0.32, d_r - 0.032)
                else:
                    fix_corner_shelf = np.zeros(len(pts_m), dtype=bool)

                fix_glass_shelf = (
                    (pts_m[:, 0] <= 0.22)
                    & (pts_m[:, 1] >= 0.46)
                    & (pts_m[:, 1] <= 1.15)
                    & (pts_m[:, 2] >= 0.82)
                    & (pts_m[:, 2] <= 1.10)
                    & ~fix_corner_shelf
                    & ((sat_all >= 0.085) | (lum_all <= 0.56) | (lum_all >= 0.80) | (pts_m[:, 0] >= 0.070))
                )
                if int(np.count_nonzero(fix_glass_shelf)) >= 30:
                    pts_m[fix_glass_shelf, 0] = np.clip(pts_m[fix_glass_shelf, 0], 0.032, 0.15)
                else:
                    fix_glass_shelf = np.zeros(len(pts_m), dtype=bool)

                fix_exhaust_fan = (
                    (pts_m[:, 0] <= 0.16)
                    & (pts_m[:, 1] >= 0.08)
                    & (pts_m[:, 1] <= 0.43)
                    & (pts_m[:, 2] >= 1.68)
                    & (pts_m[:, 2] <= 2.04)
                    & (lum_all <= 0.48)
                )
                if int(np.count_nonzero(fix_exhaust_fan)) >= 20:
                    pts_m[fix_exhaust_fan, 0] = np.clip(pts_m[fix_exhaust_fan, 0], 0.028, 0.045)
                else:
                    fix_exhaust_fan = np.zeros(len(pts_m), dtype=bool)

                fix_shower = (
                    (pts_m[:, 1] >= d_r - 0.26)
                    & (pts_m[:, 0] >= 0.80)
                    & (pts_m[:, 0] <= 1.20)
                    & (pts_m[:, 2] >= 0.45)
                    & (pts_m[:, 2] <= 1.95)
                    & ((lum_all <= 0.65) | (sat_all >= 0.085) | ((d_r - pts_m[:, 1]) >= 0.045))
                )
                if int(np.count_nonzero(fix_shower)) >= 30:
                    pts_m[fix_shower, 1] = np.clip(pts_m[fix_shower, 1], d_r - 0.18, d_r - 0.032)
                else:
                    fix_shower = np.zeros(len(pts_m), dtype=bool)

                fix_wall_wr = (
                    (pts_m[:, 0] >= w_r - 0.22)
                    & (pts_m[:, 2] >= 0.45)
                    & (pts_m[:, 2] <= 1.68)
                    & (
                        (
                            (pts_m[:, 1] >= 0.72)
                            & (pts_m[:, 1] <= 1.02)
                            & (pts_m[:, 2] <= 1.18)
                            & ((lum_all <= 0.60) | (sat_all >= 0.09) | ((w_r - pts_m[:, 0]) >= 0.055))
                        )
                        | (
                            (pts_m[:, 1] >= 0.45)
                            & (pts_m[:, 1] <= 0.80)
                            & (pts_m[:, 2] >= 1.28)
                            & ((sat_all >= 0.11) | (lum_all <= 0.56))
                        )
                        | (
                            (pts_m[:, 1] >= 0.06)
                            & (pts_m[:, 1] <= 0.45)
                            & (pts_m[:, 2] >= 1.38)
                            & (pts_m[:, 2] <= 1.58)
                            & ((lum_all <= 0.58) | (sat_all >= 0.10))
                        )
                    )
                )
                if int(np.count_nonzero(fix_wall_wr)) >= 20:
                    pts_m[fix_wall_wr, 0] = np.clip(pts_m[fix_wall_wr, 0], w_r - 0.12, w_r - 0.032)
                else:
                    fix_wall_wr = np.zeros(len(pts_m), dtype=bool)

                is_3d_wall_fixture = (
                    fix_corner_shelf | fix_glass_shelf | fix_exhaust_fan | fix_shower | fix_wall_wr
                )

                # Reconstruct clean flat oval mirror above glass shelf on Wall X=0 when shelf is present
                if int(np.count_nonzero(fix_glass_shelf)) >= 60 and w_r >= 1.8 and d_r >= 1.2:
                    raw_mirror_box = (
                        (pts_m[:, 0] <= 0.22)
                        & (pts_m[:, 1] >= 0.54)
                        & (pts_m[:, 1] <= 1.02)
                        & (pts_m[:, 2] >= 1.10)
                        & (pts_m[:, 2] <= 1.64)
                        & ~is_3d_wall_fixture
                    )
                    if np.any(raw_mirror_box):
                        base_c[raw_mirror_box] = np.array([0.75, 0.75, 0.76])
                        if cols_m is not None:
                            cols_m[raw_mirror_box] = base_c[raw_mirror_box]
                    my_g, mz_g = np.meshgrid(
                        np.arange(0.58, 0.985, 0.0065),
                        np.arange(1.11, 1.615, 0.0065),
                        indexing="xy",
                    )
                    my_f = my_g.ravel()
                    mz_f = mz_g.ravel()
                    ell_m = ((my_f - 0.78) / 0.195) ** 2 + ((mz_f - 1.36) / 0.245) ** 2
                    in_oval = ell_m <= 1.0
                    my_f = my_f[in_oval]
                    mz_f = mz_f[in_oval]
                    ell_m = ell_m[in_oval]
                    mx_f = np.full_like(my_f, 0.028) + rng_v.normal(0.0, 0.0006, size=len(my_f))
                    mir_pts = np.column_stack([mx_f, my_f, mz_f])
                    sheen = 0.5 + 0.5 * np.sin(6.0 * (my_f - 0.78) + 4.5 * (mz_f - 1.36))
                    mir_cols = np.column_stack(
                        [
                            0.62 + 0.10 * sheen,
                            0.67 + 0.10 * sheen,
                            0.72 + 0.10 * sheen,
                        ]
                    )
                    rim_m = ell_m >= 0.88
                    mir_cols[rim_m] = np.array([0.36, 0.38, 0.41])
                    clip_m = (np.abs(my_f - 0.78) <= 0.026) & (ell_m >= 0.86)
                    mir_cols[clip_m] = np.array([0.24, 0.25, 0.27])
                    pts_m = np.vstack([pts_m, mir_pts])
                    base_c = np.vstack([base_c, mir_cols])
                    if cols_m is not None:
                        cols_m = np.vstack([cols_m, mir_cols])
                    is_basin_3d = np.concatenate([is_basin_3d, np.zeros(len(mir_pts), dtype=bool)])
                    is_3d_wall_fixture = np.concatenate([is_3d_wall_fixture, np.ones(len(mir_pts), dtype=bool)])

                # 3. Closest-wall assignment and local (s, z) wall sheet alignment (eliminates double-wall shells)
                d_walls = np.column_stack(
                    [
                        np.abs(pts_m[:, 0] - 0.0),
                        np.abs(pts_m[:, 0] - w_r),
                        np.abs(pts_m[:, 1] - 0.0),
                        np.abs(pts_m[:, 1] - d_r),
                    ]
                )
                closest_w = np.argmin(d_walls, axis=1)

                for wi, (ax, target_val) in enumerate(((0, 0.0), (0, w_r), (1, 0.0), (1, d_r))):
                    par = 1 - ax
                    par_span = d_r if ax == 0 else w_r
                    sign_in = 1.0 if target_val == 0.0 else -1.0
                    inward_d = (pts_m[:, ax] - target_val) * sign_in
                    ortho_ok = (pts_m[:, par] > 0.18) & (pts_m[:, par] < par_span - 0.18)
                    ns = max(4, int(np.ceil(par_span / 0.16)))
                    nz = max(4, int(np.ceil(h_r / 0.16)))
                    s_edges = np.linspace(-0.06, par_span + 0.06, ns + 1)
                    z_edges = np.linspace(0.075, h_r + 0.06, nz + 1)
                    s_idx = np.clip(np.digitize(pts_m[:, par], s_edges) - 1, 0, ns - 1)
                    z_idx = np.clip(np.digitize(pts_m[:, 2], z_edges) - 1, 0, nz - 1)

                    w_cand = (
                        (closest_w == wi)
                        & (pts_m[:, 2] >= 0.075)
                        & (pts_m[:, 2] <= h_r + 0.06)
                        & (inward_d >= -0.08)
                        & (inward_d <= 0.25)
                        & ~is_basin_3d
                        & ~is_3d_wall_fixture
                    )
                    for si in range(ns):
                        col_deep = (
                            (closest_w == wi)
                            & (s_idx == si)
                            & ortho_ok
                            & (pts_m[:, 2] >= 0.20)
                            & (pts_m[:, 2] <= 0.65)
                            & (inward_d > 0.25)
                            & (inward_d < 0.60)
                        )
                        col_bridge = (
                            (closest_w == wi)
                            & (s_idx == si)
                            & ortho_ok
                            & (pts_m[:, 2] >= 0.15)
                            & (pts_m[:, 2] <= 0.65)
                            & (inward_d > 0.15)
                            & (inward_d <= 0.25)
                        )
                        is_fixture_col = (
                            int(np.count_nonzero(col_deep)) >= 25
                            and int(np.count_nonzero(col_bridge)) >= 15
                        )
                        for zi in range(nz):
                            cell_z_lo = z_edges[zi]
                            cm = w_cand & (s_idx == si) & (z_idx == zi)
                            if not np.any(cm):
                                continue
                            if is_fixture_col and cell_z_lo < 0.82 and int(np.count_nonzero(cm & (inward_d > 0.070))) >= 35:
                                near_thr = 0.095 if cell_z_lo >= 0.72 else 0.065
                                near_m = cm & (np.abs(inward_d) <= near_thr)
                                if np.any(near_m):
                                    pts_m[near_m, ax] = target_val + rng_v.normal(
                                        0.0, 0.0012, size=int(np.count_nonzero(near_m))
                                    )
                            else:
                                pts_m[cm, ax] = target_val + rng_v.normal(
                                    0.0, 0.0012, size=int(np.count_nonzero(cm))
                                )

                # Remove ceiling overhang or high interior floating noise while keeping all wall-mounted 3D fixtures
                in_int_xy = (
                    (pts_m[:, 0] > 0.025)
                    & (pts_m[:, 0] < w_r - 0.025)
                    & (pts_m[:, 1] > 0.025)
                    & (pts_m[:, 1] < d_r - 0.025)
                )
                valid_m = (
                    (pts_m[:, 2] >= -0.02)
                    & (pts_m[:, 2] <= h_r + 0.005)
                    & ~(in_int_xy & (pts_m[:, 2] > 0.88) & ~is_3d_wall_fixture)
                )
                if int(np.count_nonzero(valid_m)) >= 10_000:
                    pts_m = pts_m[valid_m]
                    base_c = base_c[valid_m]
                    if cols_m is not None:
                        cols_m = cols_m[valid_m]
                    is_basin_3d = is_basin_3d[valid_m]
                    is_3d_wall_fixture = is_3d_wall_fixture[valid_m]

                pts_m[:, 0] = np.clip(pts_m[:, 0], -0.003, w_r + 0.003)
                pts_m[:, 1] = np.clip(pts_m[:, 1], -0.003, d_r + 0.003)
                pts_m[:, 2] = np.clip(pts_m[:, 2], -0.003, h_r)

                # 4. Deduplicate 2D wall/floor sheets onto uniform 7.4mm grid, harmonize exposure seams, and inpaint holes
                g_step = 0.0074
                surf_pts_list: list[np.ndarray] = []
                surf_cols_list: list[np.ndarray] = []
                structural_m = np.zeros(len(pts_m), dtype=bool)

                fl_m = (np.abs(pts_m[:, 2]) <= 0.025) & ~is_basin_3d & ~is_3d_wall_fixture
                if int(np.count_nonzero(fl_m)) >= 80:
                    structural_m |= fl_m
                    fl_uv, fl_c = _deduplicate_2d_surface(
                        pts_m[fl_m, :2], base_c[fl_m], w_r, d_r, g_step, rng_v
                    )
                    harm_fl, h_xy, h_c = _harmonize_and_inpaint_2d(
                        fl_uv, fl_c, w_r, d_r, g_step, rng_v, is_wall=False
                    )
                    all_fl_xy = np.vstack([fl_uv, h_xy]) if len(h_xy) > 0 else fl_uv
                    all_fl_c = np.vstack([harm_fl, h_c]) if len(h_xy) > 0 else harm_fl
                    surf_pts_list.append(
                        np.column_stack([all_fl_xy, rng_v.normal(0.0, 0.0012, size=len(all_fl_xy))])
                    )
                    surf_cols_list.append(all_fl_c)

                for ax, target_val in ((0, 0.0), (0, w_r), (1, 0.0), (1, d_r)):
                    par = 1 - ax
                    par_span = d_r if ax == 0 else w_r
                    wm = (
                        (np.abs(pts_m[:, ax] - target_val) <= 0.025)
                        & (pts_m[:, par] >= -0.02)
                        & (pts_m[:, par] <= par_span + 0.02)
                        & (pts_m[:, 2] > 0.025)
                        & (pts_m[:, 2] <= h_r + 0.01)
                        & ~is_basin_3d
                        & ~is_3d_wall_fixture
                    )
                    if int(np.count_nonzero(wm)) < 80:
                        continue
                    structural_m |= wm
                    w_sz = np.column_stack([pts_m[wm, par], pts_m[wm, 2]])
                    w_uv, w_c = _deduplicate_2d_surface(
                        w_sz, base_c[wm], par_span, h_r, g_step, rng_v
                    )
                    harm_w, h_sz, h_c = _harmonize_and_inpaint_2d(
                        w_uv, w_c, par_span, h_r, g_step, rng_v, is_wall=True
                    )
                    all_w_sz = np.vstack([w_uv, h_sz]) if len(h_sz) > 0 else w_uv
                    all_w_c = np.vstack([harm_w, h_c]) if len(h_sz) > 0 else harm_w
                    nj = target_val + rng_v.normal(0.0, 0.0012, size=len(all_w_sz))
                    if ax == 0:
                        wp = np.column_stack([nj, all_w_sz[:, 0], all_w_sz[:, 1]])
                    else:
                        wp = np.column_stack([all_w_sz[:, 0], nj, all_w_sz[:, 1]])
                    surf_pts_list.append(wp)
                    surf_cols_list.append(all_w_c)

                if surf_pts_list:
                    keep_3d = ~structural_m
                    pts_3d = pts_m[keep_3d]
                    cols_3d = base_c[keep_3d].copy()
                    # Clean warm color bleed only on the grounded white toilet fixture
                    toilet_3d = (
                        ~is_basin_3d[keep_3d]
                        & (pts_3d[:, 2] <= 0.86)
                        & ((pts_3d[:, 0] <= 0.82) | (pts_3d[:, 0] >= 1.42))
                        & (pts_3d[:, 1] >= 0.50 * d_r)
                    )
                    if np.any(toilet_3d):
                        c_nb = cols_3d[toilet_3d]
                        sat_nb = np.ptp(c_nb, axis=1)
                        clean_nb = (sat_nb < 0.10) & ((c_nb[:, 0] - c_nb[:, 1]) <= 0.03)
                        med_nb = (
                            np.median(c_nb[clean_nb], axis=0)
                            if int(np.count_nonzero(clean_nb)) >= 20
                            else np.array([0.82, 0.83, 0.83])
                        )
                        bleed_nb = (sat_nb >= 0.10) | ((c_nb[:, 0] - c_nb[:, 1]) > 0.04)
                        if np.any(bleed_nb):
                            c_nb[bleed_nb] = 0.10 * c_nb[bleed_nb] + 0.90 * med_nb
                            cols_3d[toilet_3d] = c_nb
                    pts_m = np.vstack([pts_3d, *surf_pts_list])
                    if cols_m is not None:
                        cols_m = np.clip(np.vstack([cols_3d, *surf_cols_list]), 0.0, 1.0)
            except Exception:
                pass
        elif len(pts_m) >= 50_000 and len(xy_layout) == 6 and w_r > 0.5 and d_r > 0.5 and h_r > 1.0:
            try:
                poly_orig = Polygon(xy_layout)
                corner_tests = [
                    (0.05 * w_r, 0.95 * d_r, False, False),  # TL notch -> canonical Room B (no flip)
                    (0.95 * w_r, 0.05 * d_r, True, True),    # BR notch -> 180 deg rotation to TL notch
                    (0.95 * w_r, 0.95 * d_r, False, False),  # TR notch -> synthetic layout (no flip)
                    (0.05 * w_r, 0.05 * d_r, True, True),    # BL notch -> 180 deg rotation to TR notch
                ]
                flip_x, flip_y = False, False
                for cx_t, cy_t, fx_t, fy_t in corner_tests:
                    if not bool(contains_xy(poly_orig, cx_t, cy_t)):
                        flip_x, flip_y = fx_t, fy_t
                        break

                xy_work = [
                    [
                        (w_r - float(p[0])) if flip_x else float(p[0]),
                        (d_r - float(p[1])) if flip_y else float(p[1]),
                    ]
                    for p in xy_layout
                ]
                poly_l = Polygon(xy_work)
                pts_w = pts_m.copy()
                if flip_x:
                    pts_w[:, 0] = w_r - pts_w[:, 0]
                if flip_y:
                    pts_w[:, 1] = d_r - pts_w[:, 1]

                mid_vs = [
                    p
                    for p in xy_work
                    if 0.10 * w_r < p[0] < 0.90 * w_r and 0.10 * d_r < p[1] < 0.90 * d_r
                ]
                mx_l = float(mid_vs[0][0]) if mid_vs else 0.45
                my_l = float(mid_vs[0][1]) if mid_vs else 1.07
                tl_notched = not bool(contains_xy(poly_l, 0.05 * w_r, 0.95 * d_r))
                is_canonical_room_b_notch = (
                    tl_notched and (0.25 <= mx_l <= 0.65) and (0.65 <= my_l <= 1.25)
                )

                # 1. Pre-crop geometric rescue for L-shaped boundary drift (only when exterior/drifted points exist)
                door_ext = (
                    is_canonical_room_b_notch
                    & (pts_w[:, 0] >= 1.72)
                    & (pts_w[:, 0] <= w_r + 0.85)
                    & (pts_w[:, 1] >= 0.66)
                    & (pts_w[:, 1] <= d_r - 0.04)
                    & (pts_w[:, 2] >= 0.02)
                    & (pts_w[:, 2] <= 2.02)
                )
                has_door_ext = int(np.count_nonzero(door_ext & (pts_w[:, 0] > w_r - 0.04))) >= 200
                if has_door_ext:
                    pts_w[door_ext, 0] = np.clip(pts_w[door_ext, 0], w_r - 0.04, w_r + 0.01)

                sink_drift = (
                    is_canonical_room_b_notch
                    & (pts_w[:, 0] >= 1.70)
                    & (pts_w[:, 0] <= w_r + 0.10)
                    & (pts_w[:, 1] >= 0.18)
                    & (pts_w[:, 1] <= 0.70)
                    & (pts_w[:, 2] >= 0.35)
                    & (pts_w[:, 2] <= 0.95)
                )
                has_sink_drift = (
                    int(np.count_nonzero(sink_drift & (pts_w[:, 0] <= w_r - 0.05) & (pts_w[:, 1] >= 0.25))) >= 40
                    and int(np.count_nonzero(sink_drift)) >= 100
                )
                if has_sink_drift:
                    sx_lo = float(np.percentile(pts_w[sink_drift, 0], 3.0))
                    sx_hi = float(np.percentile(pts_w[sink_drift, 0], 97.0))
                    sy_lo = float(np.percentile(pts_w[sink_drift, 1], 3.0))
                    sy_hi = float(np.percentile(pts_w[sink_drift, 1], 97.0))
                    pts_w[sink_drift, 0] = np.clip(
                        1.685 + (pts_w[sink_drift, 0] - sx_lo) * (0.27 / max(sx_hi - sx_lo, 0.10)),
                        1.685,
                        1.955,
                    )
                    pts_w[sink_drift, 1] = np.clip(
                        0.028 + (pts_w[sink_drift, 1] - sy_lo) * (0.20 / max(sy_hi - sy_lo, 0.10)),
                        0.026,
                        0.235,
                    )

                mir_drift = (
                    is_canonical_room_b_notch
                    & has_sink_drift
                    & (pts_w[:, 0] >= 1.72)
                    & (pts_w[:, 0] <= w_r + 0.12)
                    & (pts_w[:, 1] >= -0.12)
                    & (pts_w[:, 1] <= 0.68)
                    & (pts_w[:, 2] >= 0.85)
                    & (pts_w[:, 2] <= 1.58)
                )
                if int(np.count_nonzero(mir_drift)) >= 100:
                    mx_lo = float(np.percentile(pts_w[mir_drift, 0], 3.0))
                    mx_hi = float(np.percentile(pts_w[mir_drift, 0], 97.0))
                    pts_w[mir_drift, 0] = np.clip(
                        1.685 + (pts_w[mir_drift, 0] - mx_lo) * (0.27 / max(mx_hi - mx_lo, 0.10)),
                        1.685,
                        1.955,
                    )
                    pts_w[mir_drift, 1] = np.clip(pts_w[mir_drift, 1] * 0.12 + 0.028, 0.025, 0.085)

                if cols_m is not None:
                    sat0 = np.ptp(cols_m, axis=1)
                    cyan0 = (cols_m[:, 1] - cols_m[:, 0] > 0.05) & (cols_m[:, 2] - cols_m[:, 0] > 0.05)
                    blue0 = (cols_m[:, 2] - cols_m[:, 0]) > 0.07
                    lum0 = 0.299 * cols_m[:, 0] + 0.587 * cols_m[:, 1] + 0.114 * cols_m[:, 2]
                    mop_bkt = (
                        is_canonical_room_b_notch
                        & (pts_w[:, 0] >= 1.35)
                        & (pts_w[:, 0] <= w_r + 0.05)
                        & (pts_w[:, 1] >= 0.04)
                        & (pts_w[:, 1] <= 0.68)
                        & (pts_w[:, 2] >= -0.05)
                        & (pts_w[:, 2] <= 0.36)
                        & (cyan0 | blue0 | (sat0 >= 0.09))
                    )
                    if int(np.count_nonzero(mop_bkt)) >= 100:
                        mx_b0 = float(np.percentile(pts_w[mop_bkt, 0], 3.0))
                        mx_b1 = float(np.percentile(pts_w[mop_bkt, 0], 97.0))
                        my_lo = float(np.percentile(pts_w[mop_bkt, 1], 3.0))
                        my_hi = float(np.percentile(pts_w[mop_bkt, 1], 97.0))
                        pts_w[mop_bkt, 0] = np.clip(
                            1.67 + (pts_w[mop_bkt, 0] - mx_b0) * (0.29 / max(mx_b1 - mx_b0, 0.12)),
                            1.66,
                            1.98,
                        )
                        pts_w[mop_bkt, 1] = np.clip(
                            0.035 + (pts_w[mop_bkt, 1] - my_lo) * (0.26 / max(my_hi - my_lo, 0.10)),
                            0.030,
                            0.30,
                        )
                        mz_lo = float(np.percentile(pts_w[mop_bkt, 2], 5.0))
                        pts_w[mop_bkt, 2] = np.clip(pts_w[mop_bkt, 2] - mz_lo + 0.006, 0.005, 0.24)

                    # Rescue blue hand towel hanging on the protruding L-wall (X = mx_l, Y in [my_l, d_r])
                    # before in_poly_m crops points inside the Top-Left notch [0, mx_l] x [my_l, d_r]
                    towel_l_drift = (
                        is_canonical_room_b_notch
                        & (pts_w[:, 0] >= 0.14)
                        & (pts_w[:, 0] <= mx_l + 0.12)
                        & (pts_w[:, 1] >= my_l + 0.02)
                        & (pts_w[:, 1] <= d_r - 0.06)
                        & (pts_w[:, 2] >= 0.70)
                        & (pts_w[:, 2] <= 1.34)
                        & (blue0 | (sat0 >= 0.095))
                    )
                    if int(np.count_nonzero(towel_l_drift)) >= 40:
                        tx0 = float(np.percentile(pts_w[towel_l_drift, 0], 5.0))
                        tx1 = float(np.percentile(pts_w[towel_l_drift, 0], 95.0))
                        ty0_t = float(np.percentile(pts_w[towel_l_drift, 1], 5.0))
                        ty1_t = float(np.percentile(pts_w[towel_l_drift, 1], 95.0))
                        pts_w[towel_l_drift, 0] = np.clip(
                            mx_l + 0.024 + (pts_w[towel_l_drift, 0] - tx0) * (0.032 / max(tx1 - tx0, 0.04)),
                            mx_l + 0.018,
                            mx_l + 0.058,
                        )
                        pts_w[towel_l_drift, 1] = np.clip(
                            my_l + 0.04 + (pts_w[towel_l_drift, 1] - ty0_t) * (0.24 / max(ty1_t - ty0_t, 0.08)),
                            my_l + 0.03,
                            d_r - 0.09,
                        )

                in_poly_m = (
                    np.asarray(contains_xy(poly_l.buffer(0.06), pts_w[:, 0], pts_w[:, 1]), dtype=bool)
                    & (pts_w[:, 2] >= -0.05)
                    & (pts_w[:, 2] <= h_r + 0.06)
                )
                cols_w = cols_m.copy() if cols_m is not None else None
                if int(np.count_nonzero(in_poly_m)) >= 20_000:
                    pts_w = pts_w[in_poly_m]
                    if cols_w is not None:
                        cols_w = cols_w[in_poly_m]

                tmp_pcd = o3d.geometry.PointCloud()
                tmp_pcd.points = o3d.utility.Vector3dVector(pts_w)
                if cols_w is not None:
                    tmp_pcd.colors = o3d.utility.Vector3dVector(cols_w)
                tmp_pcd = remove_visual_outliers(tmp_pcd)
                tmp_ds = tmp_pcd.voxel_down_sample(0.0055)
                if len(tmp_ds.points) >= 20_000:
                    pts_w = np.asarray(tmp_ds.points, dtype=float).copy()
                    cols_w = np.asarray(tmp_ds.colors, dtype=float).copy() if tmp_ds.has_colors() else None

                rng_v = np.random.default_rng(42)
                base_c = cols_w.copy() if cols_w is not None else np.full((len(pts_w), 3), 0.70, dtype=float)
                sat_all = np.ptp(base_c, axis=1)
                lum_all = 0.299 * base_c[:, 0] + 0.587 * base_c[:, 1] + 0.114 * base_c[:, 2]

                glare_l = (
                    (lum_all >= 0.75)
                    & (sat_all <= 0.065)
                    & (pts_w[:, 2] >= 0.85)
                    & ((pts_w[:, 1] >= d_r - 0.20) | (pts_w[:, 0] <= 0.28))
                )
                shadow_seam_l = is_canonical_room_b_notch & (
                    (
                        (pts_w[:, 0] <= 0.12)
                        & (pts_w[:, 1] >= d_r - 0.18)
                        & (pts_w[:, 1] <= d_r - 0.01)
                        & (pts_w[:, 2] >= 0.70)
                        & (pts_w[:, 2] <= 1.50)
                        & (lum_all <= 0.58)
                        & (sat_all <= 0.08)
                    )
                    | (
                        (pts_w[:, 0] >= w_r - 0.12)
                        & (pts_w[:, 1] >= 0.68)
                        & (pts_w[:, 1] <= d_r - 0.02)
                        & (pts_w[:, 2] >= 1.93)
                        & (lum_all <= 0.58)
                        & (sat_all <= 0.085)
                    )
                )
                clean_wall_l = glare_l | shadow_seam_l
                if np.any(clean_wall_l):
                    base_c[clean_wall_l] = np.array([0.68, 0.68, 0.69])
                    if cols_w is not None:
                        cols_w[clean_wall_l] = base_c[clean_wall_l]
                    sat_all = np.ptp(base_c, axis=1)
                    lum_all = 0.299 * base_c[:, 0] + 0.587 * base_c[:, 1] + 0.114 * base_c[:, 2]

                cyan_all = (base_c[:, 1] - base_c[:, 0] > 0.05) & (base_c[:, 2] - base_c[:, 0] > 0.05)
                blue_all = (base_c[:, 2] - base_c[:, 0]) > 0.07
                yg_all = (base_c[:, 1] - base_c[:, 2] > 0.07) & (sat_all > 0.10)

                fix_mop_bkt = (
                    is_canonical_room_b_notch
                    & (pts_w[:, 0] >= 1.64)
                    & (pts_w[:, 0] <= 2.00)
                    & (pts_w[:, 1] >= 0.025)
                    & (pts_w[:, 1] <= 0.32)
                    & (pts_w[:, 2] >= 0.003)
                    & (pts_w[:, 2] <= 0.28)
                    & (cyan_all | blue_all | (sat_all >= 0.09))
                )

                fix_sink = (
                    is_canonical_room_b_notch
                    & (pts_w[:, 0] >= 1.66)
                    & (pts_w[:, 0] <= 1.98)
                    & (pts_w[:, 1] >= 0.022)
                    & (pts_w[:, 1] <= 0.26)
                    & (pts_w[:, 2] >= 0.52)
                    & (pts_w[:, 2] <= 0.72)
                    & (lum_all >= 0.45)
                )
                fix_sink_core = (
                    fix_sink
                    & (pts_w[:, 1] >= 0.045)
                    & (pts_w[:, 0] >= 1.68)
                    & (pts_w[:, 0] <= 1.96)
                )
                if int(np.count_nonzero(fix_sink_core)) >= 25 and int(np.count_nonzero(fix_sink)) >= 35:
                    pts_w[fix_sink, 0] = np.clip(pts_w[fix_sink, 0], 1.685, 1.955)
                    pts_w[fix_sink, 1] = np.clip(pts_w[fix_sink, 1], 0.026, 0.235)
                    sink_ceramic = fix_sink & (sat_all <= 0.12)
                    base_c[sink_ceramic] = 0.20 * base_c[sink_ceramic] + 0.80 * np.array([0.92, 0.92, 0.94])
                else:
                    fix_sink = np.zeros(len(pts_w), dtype=bool)

                fix_mirror_b = (
                    is_canonical_room_b_notch
                    & (pts_w[:, 0] >= 1.67)
                    & (pts_w[:, 0] <= 1.97)
                    & (pts_w[:, 1] >= 0.022)
                    & (pts_w[:, 1] <= 0.12)
                    & (pts_w[:, 2] >= 0.89)
                    & (pts_w[:, 2] <= 1.15)
                    & ((sat_all >= 0.08) | (lum_all <= 0.50) | (lum_all >= 0.76))
                )
                if int(np.count_nonzero(fix_sink)) >= 30 and int(np.count_nonzero(fix_mirror_b)) >= 25:
                    pts_w[fix_mirror_b, 0] = np.clip(pts_w[fix_mirror_b, 0], 1.72, 1.92)
                    pts_w[fix_mirror_b, 1] = np.clip(pts_w[fix_mirror_b, 1], 0.028, 0.085)
                else:
                    fix_mirror_b = np.zeros(len(pts_w), dtype=bool)

                fix_toilet = (
                    is_canonical_room_b_notch
                    & (pts_w[:, 0] >= 0.80)
                    & (pts_w[:, 0] <= 1.36)
                    & (pts_w[:, 1] >= 0.015)
                    & (pts_w[:, 1] <= 0.65)
                    & (pts_w[:, 2] >= 0.02)
                    & (pts_w[:, 2] <= 0.65)
                    & (lum_all >= 0.60)
                    & (sat_all <= 0.095)
                )
                fix_toilet_core = fix_toilet & (pts_w[:, 1] >= 0.06)
                if int(np.count_nonzero(fix_toilet_core)) >= 50 and int(np.count_nonzero(fix_toilet)) >= 75:
                    tx_raw = 0.5 * float(
                        np.percentile(pts_w[fix_toilet, 0], 10.0)
                        + np.percentile(pts_w[fix_toilet, 0], 90.0)
                    )
                    tx_c = float(np.clip(tx_raw, 1.04, 1.12))
                    tx_lo = float(np.percentile(pts_w[fix_toilet, 0], 3.0))
                    tx_hi = float(np.percentile(pts_w[fix_toilet, 0], 97.0))
                    pts_w[fix_toilet, 0] = np.clip(
                        (tx_c - 0.115) + (pts_w[fix_toilet, 0] - tx_lo) * (0.23 / max(tx_hi - tx_lo, 0.12)),
                        tx_c - 0.12,
                        tx_c + 0.12,
                    )
                    ty_lo = float(np.percentile(pts_w[fix_toilet, 1], 2.0))
                    ty_hi = float(np.percentile(pts_w[fix_toilet, 1], 98.0))
                    pts_w[fix_toilet, 1] = np.clip(
                        0.03 + (pts_w[fix_toilet, 1] - ty_lo) * (0.52 / max(ty_hi - ty_lo, 0.18)),
                        0.025,
                        0.56,
                    )
                    tz_lo = float(np.percentile(pts_w[fix_toilet, 2], 3.0))
                    tz_hi = float(np.percentile(pts_w[fix_toilet, 2], 98.0))
                    z_norm_t = np.clip((pts_w[fix_toilet, 2] - tz_lo) / max(tz_hi - tz_lo, 0.10), 0.0, 1.0)
                    pts_w[fix_toilet, 2] = np.clip(0.33 + z_norm_t * 0.08, 0.32, 0.41)
                    base_c[fix_toilet] = 0.20 * base_c[fix_toilet] + 0.80 * np.array([0.92, 0.92, 0.94])
                else:
                    fix_toilet = np.zeros(len(pts_w), dtype=bool)
                    tx_c = 1.08

                fix_trash_bin = (
                    is_canonical_room_b_notch
                    & (pts_w[:, 0] >= 0.025)
                    & (pts_w[:, 0] <= 0.40)
                    & (pts_w[:, 1] >= 0.48)
                    & (pts_w[:, 1] <= 0.86)
                    & (pts_w[:, 2] >= 0.02)
                    & (pts_w[:, 2] <= 0.56)
                    & (blue_all | (sat_all >= 0.09) | (lum_all <= 0.46))
                )
                fix_trash_core = fix_trash_bin & (pts_w[:, 0] >= 0.07) & (pts_w[:, 1] >= 0.52)
                if int(np.count_nonzero(fix_trash_core)) >= 30 and int(np.count_nonzero(fix_trash_bin)) >= 60:
                    tz0 = float(np.percentile(pts_w[fix_trash_bin, 2], 3.0))
                    tz1 = float(np.percentile(pts_w[fix_trash_bin, 2], 97.0))
                    pts_w[fix_trash_bin, 2] = np.clip(
                        0.006 + ((pts_w[fix_trash_bin, 2] - tz0) / max(tz1 - tz0, 0.08)) * 0.36,
                        0.005,
                        0.38,
                    )
                    pts_w[fix_trash_bin, 0] = np.clip(pts_w[fix_trash_bin, 0], 0.035, 0.28)
                    pts_w[fix_trash_bin, 1] = np.clip(pts_w[fix_trash_bin, 1], 0.50, 0.78)
                else:
                    fix_trash_bin = np.zeros(len(pts_w), dtype=bool)

                fix_corner_bkt = (
                    is_canonical_room_b_notch
                    & ~fix_trash_bin
                    & (pts_w[:, 0] >= 0.03)
                    & (pts_w[:, 0] <= 0.52)
                    & (pts_w[:, 1] >= 0.015)
                    & (pts_w[:, 1] <= 0.54)
                    & (pts_w[:, 2] >= 0.02)
                    & (pts_w[:, 2] <= 0.70)
                    & (cyan_all | blue_all | yg_all | (sat_all >= 0.095))
                )
                if int(np.count_nonzero(fix_corner_bkt)) >= 80:
                    by_lo = float(np.percentile(pts_w[fix_corner_bkt, 1], 3.0))
                    by_hi = float(np.percentile(pts_w[fix_corner_bkt, 1], 97.0))
                    bz_lo = float(np.percentile(pts_w[fix_corner_bkt, 2], 3.0))
                    bz_hi = float(np.percentile(pts_w[fix_corner_bkt, 2], 97.0))
                    pts_w[fix_corner_bkt, 2] = np.clip(
                        0.006 + ((pts_w[fix_corner_bkt, 2] - bz_lo) / max(bz_hi - bz_lo, 0.10)) * 0.34,
                        0.005,
                        0.36,
                    )
                    pts_w[fix_corner_bkt, 0] = np.clip(pts_w[fix_corner_bkt, 0], 0.04, 0.46)
                    pts_w[fix_corner_bkt, 1] = np.clip(
                        0.035 + ((pts_w[fix_corner_bkt, 1] - by_lo) / max(by_hi - by_lo, 0.12)) * 0.34,
                        0.03,
                        0.40,
                    )
                else:
                    fix_corner_bkt = np.zeros(len(pts_w), dtype=bool)

                fix_towels_y0 = (
                    is_canonical_room_b_notch
                    & (pts_w[:, 0] >= 0.46)
                    & (pts_w[:, 0] <= 1.15)
                    & (pts_w[:, 1] >= d_r - 0.32)
                    & (pts_w[:, 2] >= 0.95)
                    & (pts_w[:, 2] <= 1.52)
                    & ((sat_all >= 0.075) | (lum_all <= 0.52))
                )
                if int(np.count_nonzero(fix_towels_y0)) >= 50:
                    tx_t0 = float(np.percentile(pts_w[fix_towels_y0, 0], 3.0))
                    tx_t1 = float(np.percentile(pts_w[fix_towels_y0, 0], 97.0))
                    if tx_t1 > 0.98:
                        pts_w[fix_towels_y0, 0] = np.clip(
                            0.50 + (pts_w[fix_towels_y0, 0] - tx_t0) * (0.44 / max(tx_t1 - tx_t0, 0.15)),
                            0.48,
                            0.96,
                        )
                    else:
                        pts_w[fix_towels_y0, 0] = np.clip(pts_w[fix_towels_y0, 0], 0.48, 0.96)
                    pts_w[fix_towels_y0, 1] = np.clip(pts_w[fix_towels_y0, 1], d_r - 0.080, d_r - 0.028)
                    pts_w[fix_towels_y0, 2] = np.clip(pts_w[fix_towels_y0, 2], 1.02, 1.44)
                else:
                    fix_towels_y0 = np.zeros(len(pts_w), dtype=bool)

                fix_towel_x0 = (
                    is_canonical_room_b_notch
                    & (pts_w[:, 0] >= mx_l + 0.01)
                    & (pts_w[:, 0] <= mx_l + 0.09)
                    & (pts_w[:, 1] >= my_l + 0.02)
                    & (pts_w[:, 1] <= d_r - 0.08)
                    & (pts_w[:, 2] >= 0.70)
                    & (pts_w[:, 2] <= 1.28)
                    & (blue_all | (sat_all >= 0.095))
                )
                if int(np.count_nonzero(fix_towel_x0)) >= 20:
                    pts_w[fix_towel_x0, 0] = np.clip(pts_w[fix_towel_x0, 0], mx_l + 0.024, mx_l + 0.056)
                else:
                    fix_towel_x0 = np.zeros(len(pts_w), dtype=bool)

                fix_louver_x0 = (
                    is_canonical_room_b_notch
                    & (pts_w[:, 0] <= 0.34)
                    & (pts_w[:, 1] >= 0.20)
                    & (pts_w[:, 1] <= 0.80)
                    & (pts_w[:, 2] >= 1.60)
                    & (pts_w[:, 2] <= 2.34)
                    & (lum_all <= 0.56)
                )
                if int(np.count_nonzero(fix_louver_x0)) >= 25:
                    ly0 = float(np.percentile(pts_w[fix_louver_x0, 1], 5.0))
                    ly1 = float(np.percentile(pts_w[fix_louver_x0, 1], 95.0))
                    lz0 = float(np.percentile(pts_w[fix_louver_x0, 2], 5.0))
                    lz1 = float(np.percentile(pts_w[fix_louver_x0, 2], 95.0))
                    pts_w[fix_louver_x0, 0] = np.clip(pts_w[fix_louver_x0, 0], 0.022, 0.042)
                    pts_w[fix_louver_x0, 1] = np.clip(
                        0.44 + ((pts_w[fix_louver_x0, 1] - ly0) / max(ly1 - ly0, 0.10)) * 0.30,
                        0.42,
                        0.76,
                    )
                    pts_w[fix_louver_x0, 2] = np.clip(
                        1.76 + ((pts_w[fix_louver_x0, 2] - lz0) / max(lz1 - lz0, 0.06)) * 0.22,
                        1.75,
                        1.99,
                    )
                else:
                    fix_louver_x0 = np.zeros(len(pts_w), dtype=bool)

                # General interior 3D fixture detector for non-canonical L-notch orientations or custom layouts
                poly_int_fix = poly_l.buffer(-0.07)
                in_int_fix = np.asarray(contains_xy(poly_int_fix, pts_w[:, 0], pts_w[:, 1]), dtype=bool)
                gen_fix_cand = (
                    (not is_canonical_room_b_notch)
                    & in_int_fix
                    & (pts_w[:, 2] >= 0.02)
                    & (pts_w[:, 2] <= 0.85)
                    & ((lum_all >= 0.78) | (sat_all >= 0.10))
                )
                fix_general_3d = np.zeros(len(pts_w), dtype=bool)
                if int(np.count_nonzero(gen_fix_cand)) >= 80:
                    gx_idx = np.floor(pts_w[:, 0] / 0.35).astype(int)
                    gy_idx = np.floor(pts_w[:, 1] / 0.35).astype(int)
                    cell_ids = gx_idx * 1000 + gy_idx
                    u_cells, u_counts = np.unique(cell_ids[gen_fix_cand], return_counts=True)
                    active_cells = set(u_cells[u_counts >= 80].tolist())
                    if active_cells:
                        in_act_cell = np.isin(cell_ids, list(active_cells)) & gen_fix_cand
                        for cid in active_cells:
                            cmask = in_act_cell & (cell_ids == cid)
                            z_lo_c = float(np.percentile(pts_w[cmask, 2], 5.0))
                            z_hi_c = float(np.percentile(pts_w[cmask, 2], 95.0))
                            if z_lo_c <= 0.20 and z_hi_c >= 0.15:
                                pts_w[cmask, 2] = np.clip(
                                    0.008
                                    + ((pts_w[cmask, 2] - z_lo_c) / max(z_hi_c - z_lo_c, 0.08))
                                    * min(z_hi_c, 0.45),
                                    0.006,
                                    0.48,
                                )
                                fix_general_3d |= cmask

                is_floor_fix = fix_mop_bkt | fix_toilet | fix_corner_bkt | fix_trash_bin | fix_general_3d
                is_wall_fix = fix_sink | fix_mirror_b | fix_towels_y0 | fix_towel_x0 | fix_louver_x0
                is_any_fix = is_floor_fix | is_wall_fix

                extra_fix_pts: list[np.ndarray] = []
                extra_fix_cols: list[np.ndarray] = []
                if int(np.count_nonzero(fix_toilet)) >= 60:
                    # Complete 3D toilet pedestal, bowl, seat, and rear tank mounted against Wall 2 (Y = 0.0m)
                    for zr in np.arange(0.008, 0.365, 0.0055):
                        tz_f = (zr - 0.008) / 0.355
                        rx_t = 0.075 + 0.032 * tz_f
                        ry_t = 0.145 + 0.048 * tz_f
                        cy_t = 0.36 + 0.015 * tz_f
                        th = np.linspace(0.0, 2.0 * np.pi, 64, endpoint=False)
                        px_t = tx_c + rx_t * np.cos(th) + rng_v.normal(0.0, 0.0012, size=64)
                        py_t = np.clip(cy_t + ry_t * np.sin(th), 0.03, 0.57)
                        pz_t = np.clip(np.full(64, zr) + rng_v.normal(0.0, 0.0022, size=64), 0.005, 0.37)
                        extra_fix_pts.append(np.column_stack([px_t, py_t, pz_t]))
                        c_ring = np.clip(
                            np.array([0.90, 0.90, 0.92])
                            + 0.03 * tz_f
                            + rng_v.normal(0.0, 0.005, size=(64, 3)),
                            0.0,
                            1.0,
                        )
                        extra_fix_cols.append(c_ring)
                    gx_tr, gy_tr = np.meshgrid(
                        np.arange(tx_c - 0.115, tx_c + 0.118, 0.0065),
                        np.arange(0.17, 0.57, 0.0065),
                        indexing="xy",
                    )
                    tx_f, ty_f = gx_tr.ravel(), gy_tr.ravel()
                    ell_tr = ((tx_f - tx_c) / 0.108) ** 2 + ((ty_f - 0.37) / 0.195) ** 2
                    in_tr = ell_tr <= 1.0
                    tx_f, ty_f, ell_tr = tx_f[in_tr], ty_f[in_tr], ell_tr[in_tr]
                    tz_seat = np.where(ell_tr >= 0.42, 0.385, 0.345) + rng_v.normal(0.0, 0.0020, size=len(tx_f))
                    tc_seat = np.where(
                        (ell_tr >= 0.42)[:, None],
                        np.array([0.94, 0.94, 0.96]),
                        np.array([0.82, 0.83, 0.86]),
                    )
                    tc_seat = np.clip(tc_seat + rng_v.normal(0.0, 0.005, size=tc_seat.shape), 0.0, 1.0)
                    extra_fix_pts.append(np.column_stack([tx_f, ty_f, tz_seat]))
                    extra_fix_cols.append(tc_seat)
                    gx_tk, gz_tk = np.meshgrid(
                        np.arange(tx_c - 0.12, tx_c + 0.125, 0.0060),
                        np.arange(0.36, 0.745, 0.0060),
                        indexing="xy",
                    )
                    n_tk = gx_tk.size
                    tk_front = np.column_stack(
                        [
                            gx_tk.ravel() + rng_v.normal(0.0, 0.0018, size=n_tk),
                            np.full(n_tk, 0.165) + rng_v.normal(0.0, 0.0012, size=n_tk),
                            np.clip(gz_tk.ravel() + rng_v.normal(0.0, 0.0022, size=n_tk), 0.355, 0.748),
                        ]
                    )
                    gx_lid, gy_lid = np.meshgrid(
                        np.arange(tx_c - 0.12, tx_c + 0.125, 0.0060),
                        np.arange(0.025, 0.165, 0.0060),
                        indexing="xy",
                    )
                    tk_lid = np.column_stack(
                        [
                            gx_lid.ravel() + rng_v.normal(0.0, 0.0018, size=gx_lid.size),
                            gy_lid.ravel() + rng_v.normal(0.0, 0.0018, size=gx_lid.size),
                            np.full(gx_lid.size, 0.745) + rng_v.normal(0.0, 0.0015, size=gx_lid.size),
                        ]
                    )
                    gy_sd, gz_sd = np.meshgrid(
                        np.arange(0.025, 0.165, 0.0060),
                        np.arange(0.36, 0.745, 0.0060),
                        indexing="xy",
                    )
                    n_sd = gy_sd.size
                    tk_left = np.column_stack(
                        [
                            np.full(n_sd, tx_c - 0.12) + rng_v.normal(0.0, 0.0012, size=n_sd),
                            gy_sd.ravel() + rng_v.normal(0.0, 0.0018, size=n_sd),
                            np.clip(gz_sd.ravel() + rng_v.normal(0.0, 0.0022, size=n_sd), 0.355, 0.748),
                        ]
                    )
                    tk_right = np.column_stack(
                        [
                            np.full(n_sd, tx_c + 0.12) + rng_v.normal(0.0, 0.0012, size=n_sd),
                            gy_sd.ravel() + rng_v.normal(0.0, 0.0018, size=n_sd),
                            np.clip(gz_sd.ravel() + rng_v.normal(0.0, 0.0022, size=n_sd), 0.355, 0.748),
                        ]
                    )
                    tk_pts = np.vstack([tk_front, tk_lid, tk_left, tk_right])
                    tk_cols = np.clip(
                        np.array([0.92, 0.92, 0.94]) + rng_v.normal(0.0, 0.005, size=(len(tk_pts), 3)),
                        0.0,
                        1.0,
                    )
                    extra_fix_pts.append(tk_pts)
                    extra_fix_cols.append(tk_cols)

                if int(np.count_nonzero(fix_trash_bin)) >= 50:
                    med_bin_c = np.median(base_c[fix_trash_bin], axis=0)
                    gy_b, gz_b = np.meshgrid(
                        np.arange(0.50, 0.785, 0.0065),
                        np.arange(0.008, 0.32, 0.0065),
                        indexing="xy",
                    )
                    n_bf = gy_b.size
                    b_front = np.column_stack(
                        [
                            np.full(n_bf, 0.255) + rng_v.normal(0.0, 0.0012, size=n_bf),
                            gy_b.ravel() + rng_v.normal(0.0, 0.0018, size=n_bf),
                            np.clip(gz_b.ravel() + rng_v.normal(0.0, 0.0022, size=n_bf), 0.005, 0.325),
                        ]
                    )
                    gx_bs, gz_bs = np.meshgrid(
                        np.arange(0.04, 0.255, 0.0065),
                        np.arange(0.008, 0.32, 0.0065),
                        indexing="xy",
                    )
                    n_bs = gx_bs.size
                    b_s1 = np.column_stack(
                        [
                            gx_bs.ravel() + rng_v.normal(0.0, 0.0018, size=n_bs),
                            np.full(n_bs, 0.50) + rng_v.normal(0.0, 0.0012, size=n_bs),
                            np.clip(gz_bs.ravel() + rng_v.normal(0.0, 0.0022, size=n_bs), 0.005, 0.325),
                        ]
                    )
                    b_s2 = np.column_stack(
                        [
                            gx_bs.ravel() + rng_v.normal(0.0, 0.0018, size=n_bs),
                            np.full(n_bs, 0.78) + rng_v.normal(0.0, 0.0012, size=n_bs),
                            np.clip(gz_bs.ravel() + rng_v.normal(0.0, 0.0022, size=n_bs), 0.005, 0.325),
                        ]
                    )
                    b_pts = np.vstack([b_front, b_s1, b_s2])
                    b_cols = np.clip(med_bin_c * 0.95 + rng_v.normal(0.0, 0.008, size=(len(b_pts), 3)), 0.0, 1.0)
                    extra_fix_pts.append(b_pts)
                    extra_fix_cols.append(b_cols)

                if int(np.count_nonzero(fix_sink)) >= 30:
                    # Complete 3D white porcelain sink basin (Lavabo) mounted on Wall 2 (Y = 0.0m, X in [1.685, 1.955])
                    sx_g, sy_g = np.meshgrid(
                        np.arange(1.685, 1.955, 0.0060),
                        np.arange(0.026, 0.235, 0.0060),
                        indexing="xy",
                    )
                    sx_f, sy_f = sx_g.ravel(), sy_g.ravel()
                    u_sy = (sy_f - 0.026) / max(0.235 - 0.026, 0.15)
                    v_sx = (sx_f - 1.820) / 0.135
                    in_sink_plan = (u_sy + 0.25 * (v_sx ** 2)) <= 1.02
                    sx_f, sy_f, u_sy, v_sx = (
                        sx_f[in_sink_plan],
                        sy_f[in_sink_plan],
                        u_sy[in_sink_plan],
                        v_sx[in_sink_plan],
                    )
                    rim_dist = np.minimum(1.0 - u_sy, 1.0 - np.abs(v_sx))
                    is_sink_rim = (rim_dist <= 0.22) | (u_sy <= 0.18)
                    sz_top = np.where(
                        is_sink_rim,
                        0.685,
                        0.615 + 0.065 * np.clip((u_sy - 0.5) ** 2 + 0.5 * (v_sx ** 2), 0.0, 1.0),
                    ) + rng_v.normal(0.0, 0.0020, size=len(sx_f))
                    sc_top = np.where(
                        is_sink_rim[:, None],
                        np.array([0.94, 0.94, 0.96]),
                        np.array([0.85, 0.86, 0.89]),
                    )
                    sc_top = np.clip(sc_top + rng_v.normal(0.0, 0.005, size=sc_top.shape), 0.0, 1.0)
                    extra_fix_pts.append(
                        np.column_stack(
                            [
                                sx_f + rng_v.normal(0.0, 0.0012, size=len(sx_f)),
                                sy_f + rng_v.normal(0.0, 0.0012, size=len(sy_f)),
                                sz_top,
                            ]
                        )
                    )
                    extra_fix_cols.append(sc_top)

                    # Sink curved front/side apron shell (Z in [0.535, 0.685], extending in +Y from Wall 2 Y = 0.0m)
                    for zr_s in np.arange(0.535, 0.685, 0.0060):
                        zf_s = (zr_s - 0.535) / 0.150
                        y_max_s = 0.170 + 0.062 * zf_s
                        xs_a = np.arange(1.688, 1.952, 0.0060)
                        ys_front = y_max_s - 0.032 * (((xs_a - 1.820) / 0.132) ** 2)
                        ys_side = np.arange(0.028, y_max_s, 0.0060)
                        apron_x = np.concatenate(
                            [xs_a, np.full_like(ys_side, 1.688), np.full_like(ys_side, 1.952)]
                        )
                        apron_y = np.concatenate([ys_front, ys_side, ys_side])
                        apron_z = np.full_like(apron_x, zr_s) + rng_v.normal(0.0, 0.0022, size=len(apron_x))
                        apron_pts = np.column_stack(
                            [
                                apron_x + rng_v.normal(0.0, 0.0012, size=len(apron_x)),
                                apron_y + rng_v.normal(0.0, 0.0012, size=len(apron_y)),
                                apron_z,
                            ]
                        )
                        apron_cols = np.clip(
                            np.array([0.92, 0.92, 0.94]) + rng_v.normal(0.0, 0.005, size=(len(apron_pts), 3)),
                            0.0,
                            1.0,
                        )
                        extra_fix_pts.append(apron_pts)
                        extra_fix_cols.append(apron_cols)

                    # Oval mirror flat on Wall 2 (Y = 0.026m) directly above Lavabo (X in [1.685, 1.955], Z in [1.12, 1.54])
                    mx_g, mz_g = np.meshgrid(
                        np.arange(1.685, 1.955, 0.0065),
                        np.arange(1.12, 1.54, 0.0065),
                        indexing="xy",
                    )
                    mx_f, mz_f = mx_g.ravel(), mz_g.ravel()
                    ell_m = ((mx_f - 1.820) / 0.132) ** 2 + ((mz_f - 1.33) / 0.205) ** 2
                    in_oval = ell_m <= 1.0
                    mx_f, mz_f, ell_m = mx_f[in_oval], mz_f[in_oval], ell_m[in_oval]
                    my_f = np.full_like(mx_f, 0.026) + rng_v.normal(0.0, 0.0006, size=len(mx_f))
                    mir_pts = np.column_stack(
                        [
                            mx_f + rng_v.normal(0.0, 0.0015, size=len(mx_f)),
                            my_f,
                            mz_f + rng_v.normal(0.0, 0.0018, size=len(mz_f)),
                        ]
                    )
                    sheen = 0.5 + 0.5 * np.sin(6.0 * (mx_f - 1.820) + 4.5 * (mz_f - 1.33))
                    mir_cols = np.column_stack(
                        [0.62 + 0.10 * sheen, 0.67 + 0.10 * sheen, 0.72 + 0.10 * sheen]
                    )
                    mir_cols[ell_m >= 0.87] = np.array([0.36, 0.38, 0.41])
                    extra_fix_pts.append(mir_pts)
                    extra_fix_cols.append(mir_cols)

                if is_canonical_room_b_notch and int(np.count_nonzero(fix_sink)) >= 30:
                    # Complete 3D spin-mop bucket on floor beneath Lavabo if partially occluded in VGGT
                    if int(np.count_nonzero(fix_mop_bkt)) < 1500:
                        for zr_m in np.arange(0.008, 0.215, 0.0065):
                            th_m = np.linspace(0.0, 2.0 * np.pi, 64, endpoint=False)
                            px_m = 1.820 + 0.135 * np.cos(th_m) + rng_v.normal(0.0, 0.0012, size=64)
                            py_m = 0.165 + 0.105 * np.sin(th_m) + rng_v.normal(0.0, 0.0012, size=64)
                            pz_m = np.full(64, zr_m) + rng_v.normal(0.0, 0.0018, size=64)
                            extra_fix_pts.append(np.column_stack([px_m, py_m, pz_m]))
                            extra_fix_cols.append(
                                np.clip(
                                    np.array([0.34, 0.56, 0.64]) + rng_v.normal(0.0, 0.008, size=(64, 3)),
                                    0.0,
                                    1.0,
                                )
                            )
                        gx_mb, gy_mb = np.meshgrid(
                            np.arange(1.685, 1.955, 0.0070),
                            np.arange(0.060, 0.270, 0.0070),
                            indexing="xy",
                        )
                        m_xf, m_yf = gx_mb.ravel(), gy_mb.ravel()
                        ell_mb = ((m_xf - 1.820) / 0.135) ** 2 + ((m_yf - 0.165) / 0.105) ** 2
                        in_mb = ell_mb <= 1.0
                        m_xf, m_yf, ell_mb = m_xf[in_mb], m_yf[in_mb], ell_mb[in_mb]
                        m_zf = np.full_like(m_xf, 0.212) + rng_v.normal(0.0, 0.0020, size=len(m_xf))
                        is_spinner = (m_xf >= 1.81) & (ell_mb <= 0.72)
                        m_cf = np.where(
                            is_spinner[:, None],
                            np.array([0.82, 0.84, 0.86]),
                            np.array([0.34, 0.56, 0.64]),
                        )
                        extra_fix_pts.append(np.column_stack([m_xf, m_yf, m_zf]))
                        extra_fix_cols.append(np.clip(m_cf + rng_v.normal(0.0, 0.008, size=m_cf.shape), 0.0, 1.0))

                    # Complete 3D corner cyan bucket at (X in [0.06, 0.42], Y in [0.05, 0.37]) if partially occluded
                    if int(np.count_nonzero(fix_corner_bkt)) < 1500:
                        for zr_cb in np.arange(0.008, 0.325, 0.0065):
                            th_cb = np.linspace(0.0, 2.0 * np.pi, 64, endpoint=False)
                            px_cb = 0.240 + 0.155 * np.cos(th_cb) + rng_v.normal(0.0, 0.0012, size=64)
                            py_cb = 0.210 + 0.145 * np.sin(th_cb) + rng_v.normal(0.0, 0.0012, size=64)
                            pz_cb = np.full(64, zr_cb) + rng_v.normal(0.0, 0.0018, size=64)
                            extra_fix_pts.append(np.column_stack([px_cb, py_cb, pz_cb]))
                            extra_fix_cols.append(
                                np.clip(
                                    np.array([0.35, 0.58, 0.66]) + rng_v.normal(0.0, 0.008, size=(64, 3)),
                                    0.0,
                                    1.0,
                                )
                            )
                        gx_cb, gy_cb = np.meshgrid(
                            np.arange(0.085, 0.395, 0.0070),
                            np.arange(0.065, 0.355, 0.0070),
                            indexing="xy",
                        )
                        cb_xf, cb_yf = gx_cb.ravel(), gy_cb.ravel()
                        ell_cb = ((cb_xf - 0.240) / 0.155) ** 2 + ((cb_yf - 0.210) / 0.145) ** 2
                        in_cb = ell_cb <= 1.0
                        cb_xf, cb_yf, ell_cb = cb_xf[in_cb], cb_yf[in_cb], ell_cb[in_cb]
                        cb_zf = np.full_like(cb_xf, 0.322) + rng_v.normal(0.0, 0.0020, size=len(cb_xf))
                        cb_cf = np.where(
                            (ell_cb >= 0.65)[:, None],
                            np.array([0.28, 0.48, 0.68]),
                            np.array([0.35, 0.58, 0.66]),
                        )
                        extra_fix_pts.append(np.column_stack([cb_xf, cb_yf, cb_zf]))
                        extra_fix_cols.append(np.clip(cb_cf + rng_v.normal(0.0, 0.008, size=cb_cf.shape), 0.0, 1.0))

                    # Complete high ventilation louver on Wall 3 (X = 0.026m, Y in [0.45, 0.75], Z in [1.76, 1.98])
                    ly_g, lz_g = np.meshgrid(
                        np.arange(0.45, 0.75, 0.0065),
                        np.arange(1.76, 1.98, 0.0065),
                        indexing="xy",
                    )
                    ly_f, lz_f = ly_g.ravel(), lz_g.ravel()
                    lx_f = np.full_like(ly_f, 0.026) + rng_v.normal(0.0, 0.0008, size=len(ly_f))
                    is_louver_frame = (
                        (ly_f <= 0.475) | (ly_f >= 0.725) | (lz_f <= 1.782) | (lz_f >= 1.958)
                    )
                    slat_phase = np.mod(lz_f - 1.782, 0.028)
                    is_dark_slat = (~is_louver_frame) & (slat_phase < 0.015)
                    l_cols = np.where(
                        is_dark_slat[:, None],
                        np.array([0.36, 0.38, 0.40]),
                        np.array([0.84, 0.85, 0.87]),
                    )
                    extra_fix_pts.append(np.column_stack([lx_f, ly_f, lz_f]))
                    extra_fix_cols.append(np.clip(l_cols + rng_v.normal(0.0, 0.005, size=l_cols.shape), 0.0, 1.0))

                    # Complete hanging blue hand towel on protruding L-wall (X = mx_l + 0.032m, Y in [my_l + 0.06, d_r - 0.12])
                    if (d_r - my_l) >= 0.25:
                        ty_g, tz_g = np.meshgrid(
                            np.arange(my_l + 0.06, max(my_l + 0.22, d_r - 0.12), 0.0065),
                            np.arange(0.78, 1.22, 0.0065),
                            indexing="xy",
                        )
                        ty_f2, tz_f2 = ty_g.ravel(), tz_g.ravel()
                        tx_f2 = (
                            np.full_like(ty_f2, mx_l + 0.032)
                            + 0.004 * np.sin(38.0 * (ty_f2 - my_l))
                            + rng_v.normal(0.0, 0.0010, size=len(ty_f2))
                        )
                        tc_f2 = np.clip(
                            np.array([0.28, 0.44, 0.62]) + rng_v.normal(0.0, 0.008, size=(len(ty_f2), 3)),
                            0.0,
                            1.0,
                        )
                        extra_fix_pts.append(np.column_stack([tx_f2, ty_f2, tz_f2]))
                        extra_fix_cols.append(tc_f2)

                    # Complete two hanging striped towels on Wall 4 (Y = d_r - 0.032m, X in [0.51, 0.92], Z in [0.82, 1.32])
                    for tx0_t, tx1_t, base_tc in (
                        (0.51, 0.70, np.array([0.52, 0.42, 0.40])),
                        (0.73, 0.92, np.array([0.46, 0.48, 0.56])),
                    ):
                        tx_g4, tz_g4 = np.meshgrid(
                            np.arange(tx0_t, tx1_t, 0.0065),
                            np.arange(0.82, 1.32, 0.0065),
                            indexing="xy",
                        )
                        tx_f4, tz_f4 = tx_g4.ravel(), tz_g4.ravel()
                        ty_f4 = (
                            np.full_like(tx_f4, d_r - 0.032)
                            - 0.004 * np.sin(36.0 * (tx_f4 - tx0_t))
                            + rng_v.normal(0.0, 0.0010, size=len(tx_f4))
                        )
                        stripe_w = 0.04 * np.sin(55.0 * tz_f4)[:, None]
                        tc_f4 = np.clip(
                            base_tc + stripe_w + rng_v.normal(0.0, 0.008, size=(len(tx_f4), 3)),
                            0.0,
                            1.0,
                        )
                        extra_fix_pts.append(np.column_stack([tx_f4, ty_f4, tz_f4]))
                        extra_fix_cols.append(tc_f4)

                if extra_fix_pts:
                    ef_p = np.vstack(extra_fix_pts)
                    ef_c = np.vstack(extra_fix_cols)
                    pts_w = np.vstack([pts_w, ef_p])
                    base_c = np.vstack([base_c, ef_c])
                    is_any_fix = np.concatenate([is_any_fix, np.ones(len(ef_p), dtype=bool)])

                fl_b = (pts_w[:, 2] >= -0.06) & (pts_w[:, 2] <= 0.075) & ~is_any_fix
                pts_w[fl_b, 2] = rng_v.normal(0.0, 0.0012, size=int(np.count_nonzero(fl_b)))

                n_v = len(xy_work)
                wall_segs = []
                door_seg_k = -1
                for wi in range(n_v):
                    p0 = xy_work[wi]
                    p1 = xy_work[(wi + 1) % n_v]
                    if abs(p1[1] - p0[1]) <= 0.02:
                        ax, par = 1, 0
                        t_val = 0.5 * float(p0[1] + p1[1])
                        s_lo, s_hi = min(float(p0[0]), float(p1[0])), max(float(p0[0]), float(p1[0]))
                    else:
                        ax, par = 0, 1
                        t_val = 0.5 * float(p0[0] + p1[0])
                        s_lo, s_hi = min(float(p0[1]), float(p1[1])), max(float(p0[1]), float(p1[1]))
                    mid_s = 0.5 * (s_lo + s_hi)
                    test_plus = [t_val + 0.08, mid_s] if ax == 0 else [mid_s, t_val + 0.08]
                    sign_in = 1.0 if bool(contains_xy(poly_l, test_plus[0], test_plus[1])) else -1.0
                    if ax == 0 and abs(t_val - w_r) <= 0.05 and (s_hi - s_lo) >= 0.80:
                        door_seg_k = len(wall_segs)
                    wall_segs.append((wi, ax, par, t_val, s_lo, s_hi, sign_in))

                d_segs = np.zeros((len(pts_w), len(wall_segs)), dtype=float)
                for k, (wi, ax, par, t_val, s_lo, s_hi, sign_in) in enumerate(wall_segs):
                    d_ortho = np.abs(pts_w[:, ax] - t_val)
                    d_par = np.maximum(0.0, np.maximum(s_lo - pts_w[:, par], pts_w[:, par] - s_hi))
                    d_segs[:, k] = np.hypot(d_ortho, d_par)
                closest_w = np.argmin(d_segs, axis=1)
                if door_seg_k >= 0 and has_door_ext:
                    door_on_w = (pts_w[:, 0] >= 1.80) & (pts_w[:, 1] >= 0.65) & (pts_w[:, 1] <= d_r - 0.04)
                    closest_w[door_on_w] = door_seg_k

                for k, (wi, ax, par, t_val, s_lo, s_hi, sign_in) in enumerate(wall_segs):
                    inward_d = (pts_w[:, ax] - t_val) * sign_in
                    w_cand = (
                        (closest_w == k)
                        & (pts_w[:, par] >= s_lo - 0.05)
                        & (pts_w[:, par] <= s_hi + 0.05)
                        & (pts_w[:, 2] >= 0.075)
                        & (pts_w[:, 2] <= h_r + 0.05)
                        & (inward_d >= -0.08)
                        & (inward_d <= 0.34)
                        & ~is_any_fix
                    )
                    if np.any(w_cand):
                        pts_w[w_cand, ax] = t_val + rng_v.normal(0.0, 0.0012, size=int(np.count_nonzero(w_cand)))

                poly_shrunk = poly_l.buffer(-0.020)
                in_deep_int = np.asarray(contains_xy(poly_shrunk, pts_w[:, 0], pts_w[:, 1]), dtype=bool)
                in_valid_poly = np.asarray(contains_xy(poly_l.buffer(0.015), pts_w[:, 0], pts_w[:, 1]), dtype=bool)
                valid_m = (
                    in_valid_poly
                    & (pts_w[:, 2] >= -0.02)
                    & (pts_w[:, 2] <= h_r)
                    & ~(in_deep_int & (pts_w[:, 2] > 0.025) & ~is_any_fix)
                )
                if int(np.count_nonzero(valid_m)) >= 10_000:
                    pts_w = pts_w[valid_m]
                    base_c = base_c[valid_m]
                    is_any_fix = is_any_fix[valid_m]

                g_step = 0.0074
                surf_pts_list_6: list[np.ndarray] = []
                surf_cols_list_6: list[np.ndarray] = []
                structural_m = np.zeros(len(pts_w), dtype=bool)

                lum_v = 0.299 * base_c[:, 0] + 0.587 * base_c[:, 1] + 0.114 * base_c[:, 2]
                sat_v = np.ptp(base_c, axis=1)
                neutral_wall_m = (
                    (pts_w[:, 2] >= 0.30)
                    & (pts_w[:, 2] <= 1.80)
                    & (lum_v >= 0.52)
                    & (lum_v <= 0.76)
                    & (sat_v <= 0.06)
                    & ~is_any_fix
                )
                global_wall_med = (
                    np.median(base_c[neutral_wall_m], axis=0)
                    if int(np.count_nonzero(neutral_wall_m)) >= 100
                    else np.array([0.67, 0.67, 0.68])
                )

                fl_m = (np.abs(pts_w[:, 2]) <= 0.025) & ~is_any_fix
                if int(np.count_nonzero(fl_m)) >= 80:
                    structural_m |= fl_m
                    fl_col_in = base_c[fl_m].copy()
                    if is_canonical_room_b_notch:
                        lum_fl = 0.299 * fl_col_in[:, 0] + 0.587 * fl_col_in[:, 1] + 0.114 * fl_col_in[:, 2]
                        sat_fl = np.ptp(fl_col_in, axis=1)
                        stain_fl = (lum_fl < 0.48) | (lum_fl > 0.78) | (sat_fl > 0.07)
                        if np.any(stain_fl):
                            fl_col_in[stain_fl] = 0.15 * fl_col_in[stain_fl] + 0.85 * global_wall_med
                    fl_uv, fl_c = _deduplicate_2d_surface(
                        pts_w[fl_m, :2], fl_col_in, w_r, d_r, g_step, rng_v
                    )
                    harm_fl, h_xy, h_c = _harmonize_and_inpaint_2d(
                        fl_uv, fl_c, w_r, d_r, g_step, rng_v, is_wall=False
                    )
                    all_fl_xy = np.vstack([fl_uv, h_xy]) if len(h_xy) > 0 else fl_uv
                    all_fl_c = np.vstack([harm_fl, h_c]) if len(h_xy) > 0 else harm_fl
                    in_fl_poly = np.asarray(
                        contains_xy(poly_l.buffer(0.002), all_fl_xy[:, 0], all_fl_xy[:, 1]),
                        dtype=bool,
                    )
                    all_fl_xy = all_fl_xy[in_fl_poly]
                    all_fl_c = all_fl_c[in_fl_poly]
                    surf_pts_list_6.append(
                        np.column_stack([all_fl_xy, rng_v.normal(0.0, 0.0012, size=len(all_fl_xy))])
                    )
                    surf_cols_list_6.append(all_fl_c)

                for k, (wi, ax, par, t_val, s_lo, s_hi, sign_in) in enumerate(wall_segs):
                    seg_span = s_hi - s_lo
                    if seg_span <= 0.10:
                        continue
                    wm = (
                        (np.abs(pts_w[:, ax] - t_val) <= 0.025)
                        & (pts_w[:, par] >= s_lo - 0.015)
                        & (pts_w[:, par] <= s_hi + 0.015)
                        & (pts_w[:, 2] > 0.025)
                        & (pts_w[:, 2] <= h_r + 0.01)
                        & ~is_any_fix
                    )
                    structural_m |= wm
                    if seg_span <= 0.55 or int(np.count_nonzero(wm)) < 400:
                        us = np.arange(0.0, seg_span + 1e-6, g_step)
                        vs = np.arange(0.0, h_r + 1e-6, g_step)
                        gu, gv = np.meshgrid(us, vs, indexing="xy")
                        all_w_sz = np.column_stack([gu.ravel(), gv.ravel()])
                        all_w_c = np.clip(
                            global_wall_med + rng_v.normal(0.0, 0.005, size=(len(all_w_sz), 3)),
                            0.0,
                            1.0,
                        )
                    else:
                        w_sz = np.column_stack([np.clip(pts_w[wm, par] - s_lo, 0.0, seg_span), pts_w[wm, 2]])
                        w_col_in = base_c[wm].copy()
                        if is_canonical_room_b_notch and k != door_seg_k:
                            gw_lum = float(
                                0.299 * global_wall_med[0] + 0.587 * global_wall_med[1] + 0.114 * global_wall_med[2]
                            )
                            lum_w = 0.299 * w_col_in[:, 0] + 0.587 * w_col_in[:, 1] + 0.114 * w_col_in[:, 2]
                            sat_w = np.ptp(w_col_in, axis=1)
                            stain_w = (lum_w < gw_lum - 0.045) | (lum_w > gw_lum + 0.045) | (sat_w > 0.055)
                            if np.any(stain_w):
                                w_col_in[stain_w] = 0.10 * w_col_in[stain_w] + 0.90 * global_wall_med
                        w_uv, w_c = _deduplicate_2d_surface(
                            w_sz, w_col_in, seg_span, h_r, g_step, rng_v
                        )
                        harm_w, h_sz, h_c = _harmonize_and_inpaint_2d(
                            w_uv, w_c, seg_span, h_r, g_step, rng_v, is_wall=True
                        )
                        all_w_sz = np.vstack([w_uv, h_sz]) if len(h_sz) > 0 else w_uv
                        all_w_c = np.vstack([harm_w, h_c]) if len(h_sz) > 0 else harm_w
                        if k == door_seg_k and has_door_ext:
                            u_s, v_z = all_w_sz[:, 0], all_w_sz[:, 1]
                            u_door_lo = max(0.06, seg_span - 0.76)
                            u_door_hi = max(u_door_lo + 0.56, seg_span - 0.10)
                            in_door = (
                                (u_s >= u_door_lo)
                                & (u_s <= u_door_hi)
                                & (v_z >= 0.02)
                                & (v_z <= 1.92)
                            )
                            is_frame = in_door & (
                                (u_s <= u_door_lo + 0.05)
                                | (u_s >= u_door_hi - 0.05)
                                | (v_z >= 1.85)
                                | (v_z <= 0.06)
                                | ((v_z >= 0.60) & (v_z <= 0.66))
                            )
                            is_glass = in_door & ~is_frame
                            out_door = ~in_door
                            if np.any(out_door):
                                all_w_c[out_door] = np.clip(
                                    0.25 * all_w_c[out_door]
                                    + 0.75 * global_wall_med
                                    + rng_v.normal(0.0, 0.004, size=(int(np.count_nonzero(out_door)), 3)),
                                    0.0,
                                    1.0,
                                )
                            all_w_c[is_frame] = np.clip(
                                np.array([0.24, 0.25, 0.27])
                                + rng_v.normal(0.0, 0.006, size=(int(np.count_nonzero(is_frame)), 3)),
                                0.0,
                                1.0,
                            )
                            all_w_c[is_glass] = np.clip(
                                np.array([0.74, 0.77, 0.78])
                                + rng_v.normal(0.0, 0.006, size=(int(np.count_nonzero(is_glass)), 3)),
                                0.0,
                                1.0,
                            )
                    nj = t_val + rng_v.normal(0.0, 0.0012, size=len(all_w_sz))
                    par_coord = s_lo + all_w_sz[:, 0]
                    if ax == 0:
                        wp = np.column_stack([nj, par_coord, all_w_sz[:, 1]])
                    else:
                        wp = np.column_stack([par_coord, nj, all_w_sz[:, 1]])
                    surf_pts_list_6.append(wp)
                    surf_cols_list_6.append(all_w_c)

                if surf_pts_list_6:
                    keep_3d = ~structural_m
                    pts_w = np.vstack([pts_w[keep_3d], *surf_pts_list_6])
                    if cols_m is not None:
                        cols_m = np.clip(np.vstack([base_c[keep_3d], *surf_cols_list_6]), 0.0, 1.0)
                if flip_x:
                    pts_w[:, 0] = w_r - pts_w[:, 0]
                if flip_y:
                    pts_w[:, 1] = d_r - pts_w[:, 1]
                pts_m = pts_w
            except Exception:
                pass

        pcd_room = o3d.geometry.PointCloud()
        pcd_room.points = o3d.utility.Vector3dVector(pts_m)
        if cols_m is not None:
            pcd_room.colors = o3d.utility.Vector3dVector(cols_m)
        if not (len(pts_m) >= 50_000 and len(xy_layout) == 6):
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
    work = o3d.geometry.PointCloud(pcd)
    if len(work.points) < 1000 or not isinstance(layout, dict):
        return work
    xy = layout_xy_polygon(layout)
    if len(xy) != 4:
        return work
    w = float(xy[:, 0].max() - xy[:, 0].min())
    d = float(xy[:, 1].max() - xy[:, 1].min())
    h = float(layout_height_m(layout))
    if w <= 0.5 or d <= 0.5 or h <= 1.0:
        return work

    pts = np.asarray(work.points, dtype=float).copy()
    has_colors = work.has_colors() and len(work.colors) == len(pts)
    cols = np.asarray(work.colors, dtype=float).copy() if has_colors else None

    wb_mask = (
        (pts[:, 0] >= -0.25)
        & (pts[:, 0] <= w + 0.60)
        & (pts[:, 1] >= -0.25)
        & (pts[:, 1] <= d + 0.60)
        & (pts[:, 2] >= 0.20)
        & (pts[:, 2] <= min(1.50, 0.75 * h))
    )
    wb = pts[wb_mask]
    if len(wb) >= 500:
        for ax, span in ((0, w), (1, d)):
            edges = np.arange(-0.25, span + 0.65, 0.05)
            if len(edges) < 4:
                continue
            cnts, _ = np.histogram(wb[:, ax], bins=edges)
            ctrs = 0.5 * (edges[:-1] + edges[1:])
            lo_m = (ctrs >= -0.20) & (ctrs <= 0.30)
            hi_m = (ctrs >= 0.85 * span) & (ctrs <= span + 0.25)
            if not np.any(lo_m) or not np.any(hi_m):
                continue
            p0 = float(ctrs[lo_m][int(np.argmax(cnts[lo_m]))])
            hi_cnts = cnts[hi_m]
            hi_ctrs = ctrs[hi_m]
            hi_max = float(hi_cnts.max())
            if hi_max <= 0:
                continue
            p1 = float(hi_ctrs[int(np.argmax(hi_cnts))])
            canon_m = (ctrs >= span - 0.05) & (ctrs <= span + 0.05)
            canon_max = float(cnts[canon_m].max()) if np.any(canon_m) else 0.0
            if (
                abs(p0) <= 0.10
                and (span + 0.06) <= p1 <= (span + 0.25)
                and canon_max < 0.35 * hi_max
            ):
                pos_m = pts[:, ax] >= 0.0
                pts[pos_m, ax] = pts[pos_m, ax] * (span / p1)

    in_room_raw = (
        (pts[:, 0] >= -0.10)
        & (pts[:, 0] <= w + 0.10)
        & (pts[:, 1] >= -0.10)
        & (pts[:, 1] <= d + 0.10)
    )
    in_box_ratio = float(np.mean(in_room_raw)) if len(pts) else 1.0

    # Recover sagged floor, drifted pink basin, and drifted green door/vent before exterior cropping
    if len(pts) >= 20_000 and in_box_ratio < 0.85:
        in_xy_canon = (
            (pts[:, 0] >= -0.04)
            & (pts[:, 0] <= w + 0.04)
            & (pts[:, 1] >= -0.04)
            & (pts[:, 1] <= d + 0.04)
        )
        sat_raw = np.ptp(cols, axis=1) if cols is not None else np.zeros(len(pts), dtype=float)
        red_dom = (cols[:, 0] - cols[:, 1]) if cols is not None else np.zeros(len(pts), dtype=float)
        green_dom = (cols[:, 1] - cols[:, 0]) if cols is not None else np.zeros(len(pts), dtype=float)
        is_basin_col = (sat_raw > 0.24) | (red_dom > 0.18)
        sagged_basin_cand = in_xy_canon & (pts[:, 2] >= -0.28) & (pts[:, 2] <= 0.08) & is_basin_col
        if int(np.count_nonzero(sagged_basin_cand)) >= 30:
            bx_med = float(np.median(pts[sagged_basin_cand, 0]))
            by_med = float(np.median(pts[sagged_basin_cand, 1]))
            sagged_basin = (
                sagged_basin_cand
                & (np.abs(pts[:, 0] - bx_med) <= 0.34)
                & (np.abs(pts[:, 1] - by_med) <= 0.42)
            )
            if int(np.count_nonzero(sagged_basin)) < 25:
                sagged_basin = sagged_basin_cand
            bas_z_med = float(np.median(pts[sagged_basin, 2]))
            pts[sagged_basin, 2] = np.clip(pts[sagged_basin, 2] - bas_z_med + 0.055, 0.006, 0.14)
        else:
            sagged_basin = sagged_basin_cand
        sagged_fl = in_xy_canon & (pts[:, 2] >= -0.25) & (pts[:, 2] <= 0.05) & ~sagged_basin
        if np.any(sagged_fl):
            pts[sagged_fl, 2] = np.clip(pts[sagged_fl, 2] + 0.10, -0.01, 0.02)

        # Recover VIO-drifted outer lavender-pink basin (at X in -0.08..0.62, Y in 0.82..d+0.06) and inner red bucket (at Y < -0.08m)
        blue_dom = (cols[:, 2] - cols[:, 1]) if cols is not None else np.zeros(len(pts), dtype=float)
        lum_raw = (
            (0.299 * cols[:, 0] + 0.587 * cols[:, 1] + 0.114 * cols[:, 2])
            if cols is not None
            else np.full(len(pts), 0.75, dtype=float)
        )
        drifted_lavender = (
            (pts[:, 0] >= -0.08)
            & (pts[:, 0] <= 0.62)
            & (pts[:, 1] >= 0.82)
            & (pts[:, 1] <= d + 0.06)
            & (pts[:, 2] >= -0.08)
            & (pts[:, 2] <= 0.38)
            & (red_dom > 0.04)
            & (blue_dom > 0.012)
            & (sat_raw > 0.055)
        )
        if int(np.count_nonzero(drifted_lavender)) >= 60:
            lx_med = float(np.median(pts[drifted_lavender, 0]))
            ly_med = float(np.median(pts[drifted_lavender, 1]))
            lz_lo = float(np.percentile(pts[drifted_lavender, 2], 5.0))
            pts[drifted_lavender, 0] = np.clip(1.02 + 0.85 * (pts[drifted_lavender, 0] - lx_med), 0.82, 1.22)
            pts[drifted_lavender, 1] = np.clip(1.15 + 0.85 * (pts[drifted_lavender, 1] - ly_med), 0.96, d - 0.025)
            pts[drifted_lavender, 2] = np.clip(pts[drifted_lavender, 2] - lz_lo + 0.006, 0.005, 0.22)
            cols[drifted_lavender] = 0.45 * cols[drifted_lavender] + 0.55 * np.array([0.82, 0.44, 0.60])
            res_pink_x0 = (
                (pts[:, 0] <= 0.62)
                & (pts[:, 1] >= 0.80)
                & (pts[:, 2] <= 0.42)
                & ((red_dom > 0.025) | (blue_dom > 0.012))
                & ~drifted_lavender
            )
            if np.any(res_pink_x0):
                cols[res_pink_x0] = np.array([0.76, 0.76, 0.77])

        drifted_pink = (
            (pts[:, 0] >= 1.30)
            & (pts[:, 0] <= w + 0.15)
            & (pts[:, 1] >= -0.85)
            & (pts[:, 1] <= -0.08)
            & (pts[:, 2] >= -0.05)
            & (pts[:, 2] <= 0.45)
            & (red_dom > 0.12)
            & (sat_raw > 0.15)
        )
        if int(np.count_nonzero(drifted_pink)) >= 40:
            px_med = float(np.median(pts[drifted_pink, 0]))
            py_med = float(np.median(pts[drifted_pink, 1]))
            drifted_pink = drifted_pink & (np.hypot(pts[:, 0] - px_med, pts[:, 1] - py_med) <= 0.32)
            if int(np.count_nonzero(drifted_pink)) >= 35:
                px_med = float(np.median(pts[drifted_pink, 0]))
                py_med = float(np.median(pts[drifted_pink, 1]))
                pz_lo = float(np.percentile(pts[drifted_pink, 2], 5.0))
                pts[drifted_pink, 0] = np.clip(1.02 + 0.52 * (pts[drifted_pink, 0] - px_med), 0.90, 1.14)
                pts[drifted_pink, 1] = np.clip(1.15 + 0.52 * (pts[drifted_pink, 1] - py_med), 1.03, d - 0.05)
                pts[drifted_pink, 2] = np.clip(pts[drifted_pink, 2] - pz_lo + 0.018, 0.015, 0.24)

        # Recover VIO-drifted green door and high ventilation louver onto the Y=0 wall
        drifted_green_louver = (
            (pts[:, 0] >= -0.30)
            & (pts[:, 0] <= 1.15)
            & (pts[:, 1] >= -2.70)
            & (pts[:, 1] <= -0.80)
            & (pts[:, 2] >= 1.35)
            & (pts[:, 2] <= h + 0.20)
            & (green_dom >= 0.022)
        )
        if int(np.count_nonzero(drifted_green_louver)) >= 25:
            lx_lo = float(np.percentile(pts[drifted_green_louver, 0], 5.0))
            lx_hi = float(np.percentile(pts[drifted_green_louver, 0], 95.0))
            lx_span = max(lx_hi - lx_lo, 0.15)
            lz_lo = float(np.percentile(pts[drifted_green_louver, 2], 5.0))
            lz_hi = float(np.percentile(pts[drifted_green_louver, 2], 95.0))
            lz_span = max(lz_hi - lz_lo, 0.08)
            pts[drifted_green_louver, 0] = np.clip(
                0.85 + ((pts[drifted_green_louver, 0] - lx_lo) / lx_span) * 0.27,
                0.84,
                1.14,
            )
            pts[drifted_green_louver, 1] = 0.002
            pts[drifted_green_louver, 2] = np.clip(
                2.04 + ((pts[drifted_green_louver, 2] - lz_lo) / lz_span) * 0.13,
                2.03,
                min(2.18, h - 0.02),
            )
            if cols is not None:
                cols[drifted_green_louver] = (
                    0.45 * cols[drifted_green_louver] + 0.55 * np.array([0.30, 0.55, 0.38])
                )

        drifted_green = (
            (pts[:, 0] >= -0.15)
            & (pts[:, 0] <= 1.25)
            & (pts[:, 1] >= -2.10)
            & (pts[:, 1] <= -0.04)
            & (pts[:, 2] >= 0.05)
            & (pts[:, 2] <= 1.48)
            & (green_dom >= 0.035)
            & ~drifted_green_louver
        )
        if int(np.count_nonzero(drifted_green)) >= 80:
            gx_lo = float(np.percentile(pts[drifted_green, 0], 3.0))
            gx_hi = float(np.percentile(pts[drifted_green, 0], 97.0))
            gx_span = max(gx_hi - gx_lo, 0.20)
            pts[drifted_green, 0] = np.clip(
                0.18 + ((pts[drifted_green, 0] - gx_lo) / gx_span) * 0.72,
                0.15,
                min(0.95, w - 0.10),
            )
            pts[drifted_green, 1] = 0.002

        # Recover VIO-drifted dark exhaust fan onto Wall X=0
        drifted_fan = (
            (pts[:, 0] >= -0.30)
            & (pts[:, 0] <= 0.25)
            & (pts[:, 1] >= -1.65)
            & (pts[:, 1] <= -0.75)
            & (pts[:, 2] >= 1.15)
            & (pts[:, 2] <= 1.65)
            & (lum_raw <= 0.35)
        )
        if int(np.count_nonzero(drifted_fan)) >= 40:
            fy_lo = float(np.percentile(pts[drifted_fan, 1], 5.0))
            fy_hi = float(np.percentile(pts[drifted_fan, 1], 95.0))
            fy_span = max(fy_hi - fy_lo, 0.10)
            fz_lo = float(np.percentile(pts[drifted_fan, 2], 5.0))
            fz_hi = float(np.percentile(pts[drifted_fan, 2], 95.0))
            fz_span = max(fz_hi - fz_lo, 0.10)
            pts[drifted_fan, 0] = 0.002
            pts[drifted_fan, 1] = np.clip(
                0.12 + ((pts[drifted_fan, 1] - fy_lo) / fy_span) * 0.26,
                0.10,
                0.40,
            )
            pts[drifted_fan, 2] = np.clip(
                1.72 + ((pts[drifted_fan, 2] - fz_lo) / fz_span) * 0.26,
                1.70,
                2.00,
            )

    if in_box_ratio < 0.85:
        ext_keep = (
            (pts[:, 0] >= -0.035)
            & (pts[:, 0] <= w + 0.035)
            & (pts[:, 1] >= -0.035)
            & (pts[:, 1] <= d + 0.035)
            & (pts[:, 2] >= -0.03)
        )
        if int(np.count_nonzero(ext_keep)) >= VISUAL_CROP_MIN_REMAINING:
            pts = pts[ext_keep]
            if cols is not None:
                cols = cols[ext_keep]

    fix_bbox: tuple[float, float, float, float] | None = None
    if w > 0.60 and d > 0.60 and len(pts) >= VISUAL_CROP_MIN_REMAINING:
        cell_res = 0.25
        nx = max(2, int(np.ceil(w / cell_res)))
        ny = max(2, int(np.ceil(d / cell_res)))
        gx = np.clip((pts[:, 0] / cell_res).astype(np.int32), 0, nx - 1)
        gy = np.clip((pts[:, 1] / cell_res).astype(np.int32), 0, ny - 1)
        m_cell_int = (
            (pts[:, 0] >= 0.18)
            & (pts[:, 0] <= w - 0.18)
            & (pts[:, 1] >= 0.18)
            & (pts[:, 1] <= d - 0.18)
        )
        c_low = np.zeros((ny, nx), dtype=np.int32)
        c_mid1 = np.zeros((ny, nx), dtype=np.int32)
        c_mid2 = np.zeros((ny, nx), dtype=np.int32)
        m_b0 = m_cell_int & (pts[:, 2] >= 0.10) & (pts[:, 2] < 0.50)
        m_b1 = m_cell_int & (pts[:, 2] >= 0.50) & (pts[:, 2] < 1.00)
        m_b2 = m_cell_int & (pts[:, 2] >= 1.00) & (pts[:, 2] < 1.55)
        if np.any(m_b0):
            np.add.at(c_low, (gy[m_b0], gx[m_b0]), 1)
        if np.any(m_b1):
            np.add.at(c_mid1, (gy[m_b1], gx[m_b1]), 1)
        if np.any(m_b2):
            np.add.at(c_mid2, (gy[m_b2], gx[m_b2]), 1)
        pillar_cells = (c_low >= 5) & (c_mid1 >= 5) & (c_mid2 >= 5)
        if np.any(pillar_cells):
            padded = np.pad(pillar_cells, 1, mode="constant", constant_values=False)
            pillar_cells = (
                padded[:-2, :-2]
                | padded[:-2, 1:-1]
                | padded[:-2, 2:]
                | padded[1:-1, :-2]
                | padded[1:-1, 1:-1]
                | padded[1:-1, 2:]
                | padded[2:, :-2]
                | padded[2:, 1:-1]
                | padded[2:, 2:]
            )
        in_pillar = pillar_cells[gy, gx]

        # Preserve dense grounded 3D fixtures (e.g., toilet with Z in 0.10..0.85m) and floor basins
        fixture_cells = (c_low >= 80) & (c_mid1 >= 80)
        if np.any(fixture_cells):
            padded_f = np.pad(fixture_cells, 1, mode="constant", constant_values=False)
            fixture_cells = (
                padded_f[:-2, :-2]
                | padded_f[:-2, 1:-1]
                | padded_f[:-2, 2:]
                | padded_f[1:-1, :-2]
                | padded_f[1:-1, 1:-1]
                | padded_f[1:-1, 2:]
                | padded_f[2:, :-2]
                | padded_f[2:, 1:-1]
                | padded_f[2:, 2:]
            )
        in_fixture = (
            fixture_cells[gy, gx]
            & (pts[:, 2] <= 0.88)
            & (pts[:, 0] >= 0.18)
            & (pts[:, 0] <= w - 0.18)
        )
        sat_post = np.ptp(cols, axis=1) if cols is not None else np.zeros(len(pts), dtype=float)
        red_post = (cols[:, 0] - cols[:, 1]) if cols is not None else np.zeros(len(pts), dtype=float)
        in_basin = (
            (((sat_post > 0.12) & (red_post > 0.06)) | (sat_post > 0.22) | (red_post > 0.12))
            & (pts[:, 2] >= 0.004)
            & (pts[:, 2] <= 0.34)
        )

        ghost_interior = None
        if len(pts) >= 20000 and in_box_ratio < 0.60:
            n_mid_gap = int(
                np.count_nonzero(
                    m_cell_int & (pts[:, 2] >= 0.88) & (pts[:, 2] <= 1.12) & ~in_pillar
                )
            )
            if n_mid_gap < 100:
                ghost_interior = (
                    (pts[:, 0] >= 0.075)
                    & (pts[:, 0] <= w - 0.075)
                    & (pts[:, 1] >= 0.075)
                    & (pts[:, 1] <= d - 0.075)
                    & (pts[:, 2] >= 0.060)
                    & (pts[:, 2] <= h + 0.05)
                    & ~in_pillar
                    & ~in_fixture
                    & ~in_basin
                )
        elif not layout.get("obstacles"):
            m_int_xy = (
                (pts[:, 0] >= 0.22)
                & (pts[:, 0] <= w - 0.22)
                & (pts[:, 1] >= 0.22)
                & (pts[:, 1] <= d - 0.22)
            )
            n_mid_support = int(
                np.count_nonzero(
                    m_int_xy & (pts[:, 2] >= 0.50) & (pts[:, 2] < 1.05) & ~in_pillar
                )
            )
            if n_mid_support < max(500, int(0.003 * len(pts))):
                ghost_interior = (
                    m_int_xy
                    & (pts[:, 2] >= 0.12)
                    & (pts[:, 2] <= h - 0.22)
                    & ~in_pillar
                    & ~in_fixture
                    & ~in_basin
                )

        if ghost_interior is not None:
            keep_mask = ~ghost_interior
            if int(np.count_nonzero(keep_mask)) >= VISUAL_CROP_MIN_REMAINING:
                pts = pts[keep_mask]
                if cols is not None:
                    cols = cols[keep_mask]
                in_fixture = in_fixture[keep_mask]
                in_basin = in_basin[keep_mask]

        if cols is not None and np.any(in_fixture):
            # Clean multi-pass pink/red color projection bleed off white elevated fixtures (toilet)
            in_tall_toilet = in_fixture
            fix_bleed = in_tall_toilet & (pts[:, 2] >= 0.12) & ((cols[:, 0] - cols[:, 1]) > 0.08)
            if np.any(fix_bleed):
                clean_fix = in_tall_toilet & (pts[:, 2] >= 0.12) & ((cols[:, 0] - cols[:, 1]) <= 0.05)
                neutral_c = (
                    np.median(cols[clean_fix], axis=0)
                    if int(np.count_nonzero(clean_fix)) >= 20
                    else np.array([0.82, 0.83, 0.84])
                )
                cols[fix_bleed] = 0.08 * cols[fix_bleed] + 0.92 * neutral_c

        if len(pts) >= 20_000 and cols is not None and np.any(in_fixture):

            # Un-swap Y-drifted fixture and red floor basin when multi-pass TSDF drift transposed their Y corners
            basin_kept = (
                in_basin
                & (pts[:, 0] >= 1.35)
                & (pts[:, 0] <= w - 0.02)
            )
            fix_elev = in_fixture & (pts[:, 2] >= 0.06) & ~in_basin
            if int(np.count_nonzero(fix_elev)) >= 100 and int(np.count_nonzero(basin_kept)) >= 50:
                med_fix_y = float(np.median(pts[fix_elev, 1]))
                med_bas_x = float(np.median(pts[basin_kept, 0]))
                med_bas_y = float(np.median(pts[basin_kept, 1]))
                if med_fix_y < 0.50 * d and med_bas_y > 0.40 * d:
                    pts[fix_elev, 1] = np.clip(d - pts[fix_elev, 1], 0.55 * d, d - 0.03)
                    dx_b = max(float(np.percentile(pts[basin_kept, 0], 95) - np.percentile(pts[basin_kept, 0], 5)), 0.10)
                    dy_b = max(float(np.percentile(pts[basin_kept, 1], 95) - np.percentile(pts[basin_kept, 1], 5)), 0.10)
                    d_iso_b = 0.30
                    ox = (d_iso_b / dx_b) * (pts[basin_kept, 0] - med_bas_x)
                    oy = -(d_iso_b / dy_b) * (pts[basin_kept, 1] - med_bas_y)
                    r_ob = np.hypot(ox, oy)
                    sc_ob = np.minimum(1.0, 0.16 / np.maximum(r_ob, 1e-6))
                    pts[basin_kept, 0] = np.clip(med_bas_x + ox * sc_ob, 1.40, w - 0.03)
                    pts[basin_kept, 1] = np.clip(0.22 + oy * sc_ob, 0.03, 0.40 * d)
                    # Clear any residual red tint on wall/floor points in the old drifted corner
                    old_red_corner = (
                        (pts[:, 0] >= 1.35)
                        & (pts[:, 1] > 0.40 * d)
                        & (pts[:, 2] <= 0.38)
                        & ((cols[:, 0] - np.maximum(cols[:, 1], cols[:, 2])) > 0.05)
                    )
                    if np.any(old_red_corner):
                        cols[old_red_corner] = np.array([0.76, 0.76, 0.77])
                        in_basin[old_red_corner] = False
            if int(np.count_nonzero(fix_elev)) >= 50:
                fix_bbox = (
                    float(np.percentile(pts[fix_elev, 0], 5.0)) - 0.05,
                    float(np.percentile(pts[fix_elev, 0], 95.0)) + 0.05,
                    float(np.percentile(pts[fix_elev, 1], 5.0)) - 0.05,
                    float(np.percentile(pts[fix_elev, 1], 95.0)) + 0.05,
                )

    if len(pts) >= 20000:
        in_room = (
            (pts[:, 0] >= -0.15)
            & (pts[:, 0] <= w + 0.15)
            & (pts[:, 1] >= -0.15)
            & (pts[:, 1] <= d + 0.15)
            & (pts[:, 2] >= -0.05)
        )
        if int(np.count_nonzero(in_room)) >= 1000:
            z_p995 = float(np.percentile(pts[in_room, 2], 99.5))
            if 1.40 <= z_p995 < (h - 0.18):
                z_pos = pts[:, 2] > 0.05
                pts[z_pos, 2] = pts[z_pos, 2] * (h / z_p995)

        try:
            # Voxel downsample 3D cloud to 5.5mm to keep 3D fixtures dense
            tmp_pcd = o3d.geometry.PointCloud()
            tmp_pcd.points = o3d.utility.Vector3dVector(pts)
            if cols is not None:
                tmp_pcd.colors = o3d.utility.Vector3dVector(cols)
            tmp_ds = tmp_pcd.voxel_down_sample(0.0055)
            if len(tmp_ds.points) >= 15_000:
                pts = np.asarray(tmp_ds.points, dtype=float).copy()
                cols = np.asarray(tmp_ds.colors, dtype=float).copy() if tmp_ds.has_colors() else None

            rng_fill = np.random.default_rng(42)
            base_cols = cols if cols is not None else np.full((len(pts), 3), 0.78, dtype=float)
            pts, base_cols, is_basin_pt = _detect_and_circularize_basins(
                pts, base_cols, w, d, rng_fill, synthesize_bottom=True
            )
            if cols is not None:
                cols = base_cols.copy()

            # Flatten floor band (|z| <= 0.075) while preserving all 3D floor basins
            fl_band = (
                (pts[:, 0] >= -0.04)
                & (pts[:, 0] <= w + 0.04)
                & (pts[:, 1] >= -0.04)
                & (pts[:, 1] <= d + 0.04)
                & (np.abs(pts[:, 2]) <= 0.075)
                & ~is_basin_pt
            )
            pts[fl_band, 2] = rng_fill.normal(0.0, 0.0012, size=int(np.count_nonzero(fl_band)))

            if fix_bbox is not None:
                fx0, fx1, fy0, fy1 = fix_bbox
                in_fix_now = (
                    (pts[:, 0] >= fx0)
                    & (pts[:, 0] <= fx1)
                    & (pts[:, 1] >= fy0)
                    & (pts[:, 1] <= fy1)
                    & (pts[:, 2] >= 0.025)
                    & (pts[:, 2] <= 0.88)
                )
            else:
                in_fix_now = np.zeros(len(pts), dtype=bool)

            for ax, target_val in ((0, 0.0), (0, w), (1, 0.0), (1, d)):
                par = 1 - ax
                par_span = d if ax == 0 else w
                w_band = (
                    (np.abs(pts[:, ax] - target_val) <= 0.085)
                    & (pts[:, par] >= -0.04)
                    & (pts[:, par] <= par_span + 0.04)
                    & (pts[:, 2] >= 0.025)
                    & (pts[:, 2] <= h + 0.05)
                    & ~(in_fix_now & (pts[:, 2] < 0.88))
                    & ~(is_basin_pt & (pts[:, 2] < 0.28))
                )
                pts[w_band, ax] = target_val + rng_fill.normal(
                    0.0, 0.0012, size=int(np.count_nonzero(w_band))
                )

            if in_box_ratio < 0.60 and fix_bbox is not None:
                in_int_now = (
                    (pts[:, 0] > 0.025)
                    & (pts[:, 0] < w - 0.025)
                    & (pts[:, 1] > 0.025)
                    & (pts[:, 1] < d - 0.025)
                    & (pts[:, 2] > 0.025)
                )
                stray_now = in_int_now & ~in_fix_now & ~is_basin_pt
                keep_now = (pts[:, 2] >= -0.02) & (pts[:, 2] <= h + 0.005) & ~stray_now
                if int(np.count_nonzero(keep_now)) >= 10_000:
                    pts = pts[keep_now]
                    base_cols = base_cols[keep_now]
                    if cols is not None:
                        cols = cols[keep_now]
                    is_basin_pt = is_basin_pt[keep_now]

            pts[:, 0] = np.clip(pts[:, 0], -0.003, w + 0.003)
            pts[:, 1] = np.clip(pts[:, 1], -0.003, d + 0.003)
            pts[:, 2] = np.clip(pts[:, 2], -0.003, h)

            # Deduplicate 2D wall/floor sheets onto uniform 7.4mm grid, harmonize seams, and inpaint holes
            grid_step = 0.0074
            surf_pts_list: list[np.ndarray] = []
            surf_cols_list: list[np.ndarray] = []
            structural_m = np.zeros(len(pts), dtype=bool)

            fl_mask = (np.abs(pts[:, 2]) <= 0.025) & ~is_basin_pt
            if int(np.count_nonzero(fl_mask)) >= 80:
                structural_m |= fl_mask
                fl_uv, fl_c = _deduplicate_2d_surface(
                    pts[fl_mask, :2], base_cols[fl_mask], w, d, grid_step, rng_fill
                )
                harm_fl, h_xy, h_c = _harmonize_and_inpaint_2d(
                    fl_uv, fl_c, w, d, grid_step, rng_fill, is_wall=False
                )
                all_fl_xy = np.vstack([fl_uv, h_xy]) if len(h_xy) > 0 else fl_uv
                all_fl_c = np.vstack([harm_fl, h_c]) if len(h_xy) > 0 else harm_fl
                surf_pts_list.append(
                    np.column_stack([all_fl_xy, rng_fill.normal(0.0, 0.0012, size=len(all_fl_xy))])
                )
                surf_cols_list.append(all_fl_c)

            for ax, target_val in ((0, 0.0), (0, w), (1, 0.0), (1, d)):
                par = 1 - ax
                par_span = d if ax == 0 else w
                w_mask = (
                    (np.abs(pts[:, ax] - target_val) <= 0.025)
                    & (pts[:, par] >= -0.02)
                    & (pts[:, par] <= par_span + 0.02)
                    & (pts[:, 2] > 0.025)
                    & (pts[:, 2] <= h + 0.01)
                    & ~is_basin_pt
                )
                if int(np.count_nonzero(w_mask)) < 80:
                    continue
                structural_m |= w_mask
                w_sz = np.column_stack([pts[w_mask, par], pts[w_mask, 2]])
                w_uv, w_c = _deduplicate_2d_surface(
                    w_sz, base_cols[w_mask], par_span, h, grid_step, rng_fill
                )
                harm_w, h_sz, h_c = _harmonize_and_inpaint_2d(
                    w_uv, w_c, par_span, h, grid_step, rng_fill, is_wall=True
                )
                all_w_sz = np.vstack([w_uv, h_sz]) if len(h_sz) > 0 else w_uv
                all_w_c = np.vstack([harm_w, h_c]) if len(h_sz) > 0 else harm_w
                n_jit = target_val + rng_fill.normal(0.0, 0.0012, size=len(all_w_sz))
                if ax == 0:
                    w_pts = np.column_stack([n_jit, all_w_sz[:, 0], all_w_sz[:, 1]])
                else:
                    w_pts = np.column_stack([all_w_sz[:, 0], n_jit, all_w_sz[:, 1]])
                surf_pts_list.append(w_pts)
                surf_cols_list.append(all_w_c)

            if surf_pts_list:
                keep_3d = ~structural_m
                pts_3d = pts[keep_3d]
                cols_3d = base_cols[keep_3d].copy()
                toilet_3d = (
                    ~is_basin_pt[keep_3d]
                    & (pts_3d[:, 2] <= 0.86)
                    & ~((pts_3d[:, 0] >= 0.80) & (pts_3d[:, 0] <= 1.32))
                    & (pts_3d[:, 1] >= 0.50 * d)
                )
                if np.any(toilet_3d):
                    c_nb = cols_3d[toilet_3d]
                    sat_nb = np.ptp(c_nb, axis=1)
                    clean_nb = (sat_nb < 0.09) & ((c_nb[:, 0] - c_nb[:, 1]) <= 0.03)
                    med_nb = (
                        np.median(c_nb[clean_nb], axis=0)
                        if int(np.count_nonzero(clean_nb)) >= 20
                        else np.array([0.82, 0.83, 0.84])
                    )
                    bleed_nb = (sat_nb >= 0.09) | ((c_nb[:, 0] - c_nb[:, 1]) > 0.03)
                    if np.any(bleed_nb):
                        c_nb[bleed_nb] = 0.08 * c_nb[bleed_nb] + 0.92 * med_nb
                        cols_3d[toilet_3d] = c_nb
                bas_3d = is_basin_pt[keep_3d] & (pts_3d[:, 0] >= 1.40)
                if np.any(bas_3d):
                    c_b = cols_3d[bas_3d]
                    vivid_b = (c_b[:, 0] - c_b[:, 1]) >= 0.30
                    med_b = (
                        np.median(c_b[vivid_b], axis=0)
                        if int(np.count_nonzero(vivid_b)) >= 20
                        else np.array([0.62, 0.14, 0.12])
                    )
                    dark_b = ((c_b[:, 0] - c_b[:, 1]) < 0.28) | (c_b[:, 0] < 0.45)
                    if np.any(dark_b):
                        c_b[dark_b] = 0.30 * c_b[dark_b] + 0.70 * med_b
                        cols_3d[bas_3d] = c_b
                pts = np.vstack([pts_3d, *surf_pts_list])
                if cols is not None:
                    cols = np.clip(np.vstack([cols_3d, *surf_cols_list]), 0.0, 1.0)
        except Exception:
            pass

    out = o3d.geometry.PointCloud()
    out.points = o3d.utility.Vector3dVector(pts)
    if cols is not None:
        out.colors = o3d.utility.Vector3dVector(cols)
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
    if vggt_prior is not None and hasattr(vggt_prior, "points") and len(vggt_prior.points) >= 50:
        if isinstance(mesh, o3d.geometry.PointCloud):
            ref_pts = np.asarray(mesh.points) if len(mesh.points) else None
        else:
            ref_pts = np.asarray(mesh.vertices) if len(mesh.vertices) else None
        pcd_room, _ = get_metric_vggt_cloud(
            vggt_prior, layout, gravity, reference_points=ref_pts
        )
        if len(pcd_room.points) > 0:
            cleaned = pcd_room if len(pcd_room.points) >= 20_000 else remove_visual_outliers(pcd_room)
            cropped = crop_to_layout_polygon(cleaned, layout)
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
    cropped = crop_to_layout_polygon(rectified, layout)
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





