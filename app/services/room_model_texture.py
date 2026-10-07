from __future__ import annotations

import logging
import os
import cv2
import numpy as np
import pandas as pd
import trimesh
from PIL import Image
from scipy.spatial.transform import Rotation

from app.services.plane_segmentation import align_rotation, G_ARCORE_WORLD
from app.services.room_model import (
    layout_height_m,
    layout_xy_polygon,
    triangulate_floor_xy,
    z_up_to_gltf_y_up,
)

logger = logging.getLogger(__name__)


def _find_coherent_camera_cluster(cams: list[dict], max_gap: int = 20) -> list[dict]:
    """Group cameras by frame index into contiguous sweeps and return the largest cluster."""
    if not cams:
        return []
    cams_sorted = sorted(cams, key=lambda c: c["idx"])
    clusters = []
    cur = [cams_sorted[0]]
    for i in range(1, len(cams_sorted)):
        if cams_sorted[i]["idx"] - cams_sorted[i - 1]["idx"] <= max_gap:
            cur.append(cams_sorted[i])
        else:
            clusters.append(cur)
            cur = [cams_sorted[i]]
    if cur:
        clusters.append(cur)
    clusters.sort(key=len, reverse=True)
    return clusters[0]


def _sample_bilinear(img_rgb: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Vectorized bilinear interpolation of RGB image."""
    H, W = img_rgb.shape[:2]
    u = np.clip(u, 0, W - 1.001)
    v = np.clip(v, 0, H - 1.001)
    u0 = np.floor(u).astype(int)
    v0 = np.floor(v).astype(int)
    u1 = np.minimum(u0 + 1, W - 1)
    v1 = np.minimum(v0 + 1, H - 1)
    du = (u - u0)[:, None]
    dv = (v - v0)[:, None]
    c00 = img_rgb[v0, u0]
    c01 = img_rgb[v0, u1]
    c10 = img_rgb[v1, u0]
    c11 = img_rgb[v1, u1]
    return (1.0 - du) * (1.0 - dv) * c00 + du * (1.0 - dv) * c01 + (1.0 - du) * dv * c10 + du * dv * c11


def export_room_model_texture_glb(
    session_dir: str,
    layout: dict | None = None,
    texture_size: int = 1024,
    max_cameras_per_surface: int = 12,
) -> str | None:
    """Project camera RGB images onto parametric room walls & floor using coherent temporal clustering & soft feathering.

    Preserves room_model.glb untouched. Fail-open (returns None on failure, never raises).
    """
    try:
        if layout is None:
            fp_path = os.path.join(session_dir, "floorplan.json")
            if not os.path.isfile(fp_path):
                return None
            import json

            with open(fp_path, "r", encoding="utf-8") as f:
                layout = json.load(f)

        xy = layout_xy_polygon(layout)
        if len(xy) < 4:
            return None

        height_m = layout_height_m(layout)
        if not np.isfinite(height_m) or height_m <= 0.10:
            return None

        cam_matrix_path = os.path.join(session_dir, "camera_matrix.csv")
        if not os.path.isfile(cam_matrix_path):
            return None
        K = np.loadtxt(cam_matrix_path, delimiter=",")
        if K.shape != (3, 3):
            return None

        from app.pipeline.runner import load_pose_table_for_tsdf

        pose_df = load_pose_table_for_tsdf(session_dir)
        if pose_df.empty:
            return None

        # Build coordinate frame conversions
        lf = layout.get("level_frame") if isinstance(layout.get("level_frame"), dict) else {}
        R_level = np.asarray(lf.get("rotation_3x3", np.eye(3)), dtype=float).reshape(3, 3)
        t_level = np.asarray(lf.get("translation", [0.0, 0.0, 0.0]), dtype=float).reshape(3)

        ff = layout.get("floorplan_frame") if isinstance(layout.get("floorplan_frame"), dict) else None
        if ff is not None:
            R_ff = np.asarray(ff.get("rotation_2x2", np.eye(2)), dtype=float).reshape(2, 2)
            t_ff = np.asarray(ff.get("translation_xy", [0.0, 0.0]), dtype=float).reshape(2)
        else:
            R_ff = np.eye(2)
            t_ff = np.zeros(2)

        g = np.asarray(G_ARCORE_WORLD, dtype=float).reshape(3)
        g_norm = float(np.linalg.norm(g))
        R_grav = align_rotation(g / g_norm) if g_norm > 0 else np.eye(3)

        def layout_to_world(pts_layout: np.ndarray) -> np.ndarray:
            p_l = np.empty_like(pts_layout)
            p_l[:, :2] = (R_ff.T @ (pts_layout[:, :2] - t_ff).T).T
            p_l[:, 2] = pts_layout[:, 2]
            p_g = (R_level.T @ (p_l - t_level).T).T
            p_w = (R_grav.T @ p_g.T).T
            return p_w

        # Gather all valid camera poses
        all_keyframes = []
        for idx, row in pose_df.iterrows():
            fn = int(row["frame"])
            rgb_file = os.path.join(session_dir, "rgb", f"{fn:06d}.jpg")
            if os.path.isfile(rgb_file):
                R = Rotation.from_quat([row["qx"], row["qy"], row["qz"], row["qw"]]).as_matrix()
                t = np.array([row["x"], row["y"], row["z"]], dtype=float)
                all_keyframes.append({"frame": fn, "path": rgb_file, "R": R, "t": t, "idx": idx})

        if not all_keyframes:
            return None

        n_walls = len(xy)
        wall_lengths = [
            float(np.linalg.norm(xy[(i + 1) % n_walls] - xy[i])) for i in range(n_walls)
        ]
        total_perimeter = max(1e-3, sum(wall_lengths))

        tex_w, tex_h = texture_size, texture_size
        atlas = np.full((tex_h, tex_w, 3), 225, dtype=np.uint8)

        wall_h_px = int(tex_h * 0.60)
        cum_u = 0.0

        verts_3d = []
        faces = []
        uvs = []

        # 1. Bake each wall with coherent temporal clustering & soft feathering
        for i in range(n_walls):
            A = xy[i]
            B = xy[(i + 1) % n_walls]
            L = wall_lengths[i]
            T = (B - A) / max(1e-6, L)
            N_in = np.array([-T[1], T[0], 0.0])
            N_w = layout_to_world(N_in.reshape(1, 3)) - layout_to_world(np.zeros((1, 3)))
            N_w = N_w[0] / max(1e-6, float(np.linalg.norm(N_w[0])))

            w_frac = L / total_perimeter
            u_start = cum_u
            u_end = cum_u + w_frac
            cum_u = u_end

            col_start = int(round(u_start * tex_w))
            col_end = int(round(u_end * tex_w))
            w_px = max(1, col_end - col_start)

            z_coords = height_m - (np.arange(wall_h_px) / wall_h_px) * height_m
            s_coords = (np.arange(w_px) / w_px) * L
            grid_s, grid_z = np.meshgrid(s_coords, z_coords)

            pts_layout = np.empty((wall_h_px * w_px, 3), dtype=float)
            pts_layout[:, 0] = A[0] + grid_s.ravel() * T[0]
            pts_layout[:, 1] = A[1] + grid_s.ravel() * T[1]
            pts_layout[:, 2] = grid_z.ravel()
            pts_w = layout_to_world(pts_layout)

            # Find cameras that directly look at this wall
            mid_layout = np.array([(A[0] + B[0]) / 2.0, (A[1] + B[1]) / 2.0, height_m / 2.0])
            mid_w = layout_to_world(mid_layout.reshape(1, 3))[0]

            wall_cams = []
            for cam in all_keyframes:
                D = cam["t"] - mid_w
                dist = np.linalg.norm(D)
                cos_a = float(np.dot(D, N_w) / (dist + 1e-6))
                cam_z = cam["R"][:, 2]
                cos_opt = float(-np.dot(cam_z, D) / (dist + 1e-6))
                if cos_a > 0.35 and cos_opt > 0.35:
                    wall_cams.append(cam)

            # Select largest contiguous sweep
            selected_cluster = _find_coherent_camera_cluster(wall_cams)
            if not selected_cluster:
                selected_cluster = wall_cams if wall_cams else all_keyframes[::10]

            stride = max(1, len(selected_cluster) // max_cameras_per_surface)
            cams_to_use = selected_cluster[::stride]

            accum_color = np.zeros((wall_h_px * w_px, 3), dtype=np.float32)
            accum_weight = np.zeros(wall_h_px * w_px, dtype=np.float32)

            for cam in cams_to_use:
                img = cv2.imread(cam["path"])
                if img is None:
                    continue
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
                im_h, im_w = img_rgb.shape[:2]

                D = cam["t"] - pts_w
                dist = np.linalg.norm(D, axis=1)
                cos_a = np.sum(D * N_w, axis=1) / (dist + 1e-6)
                valid_a = cos_a > 0.15
                if not np.any(valid_a):
                    continue

                p_cam = (cam["R"].T @ (pts_w[valid_a] - cam["t"]).T).T
                in_front = p_cam[:, 2] > 0.2
                if not np.any(in_front):
                    continue

                idx_pts = np.where(valid_a)[0][in_front]
                p_c = p_cam[in_front]
                u = K[0, 0] * p_c[:, 0] / p_c[:, 2] + K[0, 2]
                v = K[1, 1] * p_c[:, 1] / p_c[:, 2] + K[1, 2]

                in_bounds = (u >= 8) & (u < im_w - 8) & (v >= 8) & (v < im_h - 8)
                if not np.any(in_bounds):
                    continue

                idx_f = idx_pts[in_bounds]
                u_f = u[in_bounds]
                v_f = v[in_bounds]

                # Soft border feathering: smoothstep over 50 pixels from boundary
                d_u = np.minimum(u_f, float(im_w - 1) - u_f)
                d_v = np.minimum(v_f, float(im_h - 1) - v_f)
                d_border = np.minimum(d_u, d_v)
                w_border = np.clip((d_border - 8.0) / 45.0, 0.0, 1.0)
                w_border = 0.5 - 0.5 * np.cos(np.pi * w_border)

                w = w_border * (cos_a[idx_f] ** 3) / (dist[idx_f] + 0.1)
                c_interp = _sample_bilinear(img_rgb, u_f, v_f)

                accum_color[idx_f] += c_interp * w[:, None]
                accum_weight[idx_f] += w

            wall_patch = np.full((wall_h_px, w_px, 3), 220, dtype=np.uint8)
            has_w = accum_weight > 1e-4
            if np.any(has_w):
                wall_patch.reshape(-1, 3)[has_w] = np.clip(
                    np.round(accum_color[has_w] / accum_weight[has_w, None]), 0, 255
                ).astype(np.uint8)

            wall_mask = has_w.astype(np.uint8).reshape(wall_h_px, w_px) * 255
            if np.any(wall_mask == 0) and np.any(wall_mask > 0):
                wall_patch = cv2.inpaint(
                    wall_patch,
                    (wall_mask == 0).astype(np.uint8),
                    inpaintRadius=5,
                    flags=cv2.INPAINT_TELEA,
                )

            atlas[0:wall_h_px, col_start:col_end] = wall_patch

            # 3D vertices and UV for wall i
            i0 = len(verts_3d)
            verts_3d.append([A[0], A[1], 0.0])
            verts_3d.append([B[0], B[1], 0.0])
            verts_3d.append([B[0], B[1], height_m])
            verts_3d.append([A[0], A[1], height_m])

            uvs.append([u_start, 0.40])
            uvs.append([u_end, 0.40])
            uvs.append([u_end, 1.00])
            uvs.append([u_start, 1.00])

            faces.append([i0 + 0, i0 + 2, i0 + 1])
            faces.append([i0 + 0, i0 + 3, i0 + 2])

        # 2. Bake Floor with coherent temporal clustering & soft feathering
        floor_row_start = int(tex_h * 0.62)
        floor_row_end = tex_h - 10
        floor_col_start = 10
        floor_col_end = tex_w - 10
        f_w = floor_col_end - floor_col_start
        f_h = floor_row_end - floor_row_start

        min_x, max_x = float(xy[:, 0].min()), float(xy[:, 0].max())
        min_y, max_y = float(xy[:, 1].min()), float(xy[:, 1].max())
        span_x = max(1e-3, max_x - min_x)
        span_y = max(1e-3, max_y - min_y)

        xs = min_x + (np.arange(f_w) / f_w) * span_x
        ys = max_y - (np.arange(f_h) / f_h) * span_y
        gx, gy = np.meshgrid(xs, ys)

        pts_floor = np.column_stack([gx.ravel(), gy.ravel(), np.zeros(f_w * f_h)])
        pts_w = layout_to_world(pts_floor)
        N_floor = np.array([0.0, 0.0, 1.0])
        N_w = layout_to_world(N_floor.reshape(1, 3)) - layout_to_world(np.zeros((1, 3)))
        N_w = N_w[0] / max(1e-6, float(np.linalg.norm(N_w[0])))

        mid_floor_layout = np.array([(min_x + max_x) / 2.0, (min_y + max_y) / 2.0, 0.0])
        mid_floor_w = layout_to_world(mid_floor_layout.reshape(1, 3))[0]

        floor_cams = []
        for cam in all_keyframes:
            D = cam["t"] - mid_floor_w
            dist = np.linalg.norm(D)
            cos_a = float(np.dot(D, N_w) / (dist + 1e-6))
            cam_z = cam["R"][:, 2]
            cos_opt = float(-np.dot(cam_z, D) / (dist + 1e-6))
            if cos_a > 0.25 and cos_opt > 0.25:
                floor_cams.append(cam)

        selected_floor_cluster = _find_coherent_camera_cluster(floor_cams)
        if not selected_floor_cluster:
            selected_floor_cluster = floor_cams if floor_cams else all_keyframes[::10]

        stride = max(1, len(selected_floor_cluster) // max_cameras_per_surface)
        cams_to_use_floor = selected_floor_cluster[::stride]

        accum_color_f = np.zeros((f_h * f_w, 3), dtype=np.float32)
        accum_weight_f = np.zeros(f_h * f_w, dtype=np.float32)

        for cam in cams_to_use_floor:
            img = cv2.imread(cam["path"])
            if img is None:
                continue
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
            im_h, im_w = img_rgb.shape[:2]

            D = cam["t"] - pts_w
            dist = np.linalg.norm(D, axis=1)
            cos_a = np.sum(D * N_w, axis=1) / (dist + 1e-6)
            valid_a = cos_a > 0.15
            if not np.any(valid_a):
                continue

            p_cam = (cam["R"].T @ (pts_w[valid_a] - cam["t"]).T).T
            in_front = p_cam[:, 2] > 0.2
            if not np.any(in_front):
                continue

            idx_pts = np.where(valid_a)[0][in_front]
            p_c = p_cam[in_front]
            u = K[0, 0] * p_c[:, 0] / p_c[:, 2] + K[0, 2]
            v = K[1, 1] * p_c[:, 1] / p_c[:, 2] + K[1, 2]

            in_bounds = (u >= 8) & (u < im_w - 8) & (v >= 8) & (v < im_h - 8)
            if not np.any(in_bounds):
                continue

            idx_f = idx_pts[in_bounds]
            u_f = u[in_bounds]
            v_f = v[in_bounds]

            d_u = np.minimum(u_f, float(im_w - 1) - u_f)
            d_v = np.minimum(v_f, float(im_h - 1) - v_f)
            d_border = np.minimum(d_u, d_v)
            w_border = np.clip((d_border - 8.0) / 45.0, 0.0, 1.0)
            w_border = 0.5 - 0.5 * np.cos(np.pi * w_border)

            w = w_border * (cos_a[idx_f] ** 3) / (dist[idx_f] + 0.1)
            c_interp = _sample_bilinear(img_rgb, u_f, v_f)

            accum_color_f[idx_f] += c_interp * w[:, None]
            accum_weight_f[idx_f] += w

        floor_patch = np.full((f_h, f_w, 3), 215, dtype=np.uint8)
        has_w_f = accum_weight_f > 1e-4
        if np.any(has_w_f):
            floor_patch.reshape(-1, 3)[has_w_f] = np.clip(
                np.round(accum_color_f[has_w_f] / accum_weight_f[has_w_f, None]), 0, 255
            ).astype(np.uint8)

        floor_mask = has_w_f.astype(np.uint8).reshape(f_h, f_w) * 255
        if np.any(floor_mask == 0) and np.any(floor_mask > 0):
            floor_patch = cv2.inpaint(
                floor_patch,
                (floor_mask == 0).astype(np.uint8),
                inpaintRadius=5,
                flags=cv2.INPAINT_TELEA,
            )

        atlas[floor_row_start:floor_row_end, floor_col_start:floor_col_end] = floor_patch

        # Floor 3D vertices and UV
        u_f_min = floor_col_start / tex_w
        u_f_max = floor_col_end / tex_w
        v_f_min = 1.0 - (floor_row_end / tex_h)
        v_f_max = 1.0 - (floor_row_start / tex_h)

        base_floor = len(verts_3d)
        for p in xy:
            verts_3d.append([float(p[0]), float(p[1]), 0.0])
            u_p = u_f_min + (p[0] - min_x) / span_x * (u_f_max - u_f_min)
            v_p = v_f_min + (p[1] - min_y) / span_y * (v_f_max - v_f_min)
            uvs.append([u_p, v_p])

        for tri in triangulate_floor_xy(xy):
            faces.append(
                [base_floor + int(tri[0]), base_floor + int(tri[1]), base_floor + int(tri[2])]
            )

        # Save atlas image as PNG for 2D inspection
        atlas_path = os.path.join(session_dir, "room_model_texture_atlas.png")
        Image.fromarray(atlas).save(atlas_path, optimize=True)

        # 3. Export GLB with PBR material
        verts_y_up = z_up_to_gltf_y_up(np.asarray(verts_3d, dtype=float))
        faces_arr = np.asarray(faces, dtype=np.int32)
        uvs_arr = np.asarray(uvs, dtype=float)

        pil_atlas = Image.fromarray(atlas)
        mat = trimesh.visual.material.PBRMaterial(
            baseColorTexture=pil_atlas,
            roughnessFactor=0.85,
            metallicFactor=0.05,
        )
        vis = trimesh.visual.TextureVisuals(uv=uvs_arr, material=mat, image=pil_atlas)
        tm = trimesh.Trimesh(vertices=verts_y_up, faces=faces_arr, visual=vis, process=False)

        tmp_glb = os.path.join(session_dir, "room_model_texture.tmp.glb")
        final_glb = os.path.join(session_dir, "room_model_texture.glb")

        tm.export(tmp_glb, file_type="glb")
        if not os.path.isfile(tmp_glb) or os.path.getsize(tmp_glb) == 0:
            return None

        os.replace(tmp_glb, final_glb)
        logger.info("Exported textured room model: %s (%d bytes)", final_glb, os.path.getsize(final_glb))
        return final_glb

    except Exception:
        logger.exception("export_room_model_texture_glb failed for %s", session_dir)
        return None
