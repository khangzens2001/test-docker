"""VGGT Local GPU Inference Service.

Extracts keyframes from mobile session and runs facebook/VGGT-1B locally on GPU
in FP16 (optimized for NVIDIA RTX 2060 12GB VRAM). Produces vggt/cameras.json
and vggt/project_point_cloud.npz for downstream Manhattan topology prior.
"""
from __future__ import annotations

import contextlib
import gc
import json
import logging
import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import sys
import threading
import time
from typing import Optional

import numpy as np
import torch

from app.services.vggt_prior import N_VGGT_MAX
try:
    from scripts.export_vggt_keyframes import export_vggt_keyframes
except ImportError:
    from export_vggt_keyframes import export_vggt_keyframes

logger = logging.getLogger(__name__)

# Ensure vggt-space is in sys.path if available
DEFAULT_VGGT_SPACE = "/app/vggt-space" if os.path.isdir("/app/vggt-space") else "/home/zenzen2411/development/projects/test-docker/vggt-space"
VGGT_SPACE_DIR = os.getenv("VGGT_SPACE_DIR", DEFAULT_VGGT_SPACE)
if VGGT_SPACE_DIR and os.path.isdir(VGGT_SPACE_DIR) and VGGT_SPACE_DIR not in sys.path:
    sys.path.append(VGGT_SPACE_DIR)


def _patch_vggt_pos_embed(model):
    """Patch DPTHead _apply_pos_embed so that generated grid matches input dtype (FP16)."""
    try:
        from vggt.heads.utils import create_uv_grid, position_grid_to_embed

        def _patched_apply_pos_embed(self, x: torch.Tensor, W: int, H: int, ratio: float = 0.1) -> torch.Tensor:
            patch_w = x.shape[-1]
            patch_h = x.shape[-2]
            pos_embed = create_uv_grid(patch_w, patch_h, aspect_ratio=W / H, dtype=x.dtype, device=x.device)
            pos_embed = position_grid_to_embed(pos_embed, x.shape[1])
            pos_embed = pos_embed.to(x.dtype) * ratio
            pos_embed = pos_embed.permute(2, 0, 1)[None].expand(x.shape[0], -1, -1, -1)
            return x + pos_embed

        if hasattr(model, "point_head") and model.point_head is not None:
            model.point_head._apply_pos_embed = _patched_apply_pos_embed.__get__(model.point_head)
    except Exception as e:
        logger.warning("Could not patch VGGT _apply_pos_embed: %s", e)


_vggt_lock = threading.Lock()


@contextlib.contextmanager
def _enable_feature_diet(model):
    """Context manager to optimize aggregator forward to offload intermediate tokens to CPU (saves ~4.3GB VRAM).
    Thread-safe and guarantees restoration of original forward method upon exit.
    """
    with _vggt_lock:
        if not hasattr(model, "aggregator") or not hasattr(model.aggregator, "forward"):
            yield
            return

        orig_aggregator_forward = model.aggregator.forward
        from vggt.models.aggregator import slice_expand_and_flatten

        def _optimized_aggregator_forward(imgs):
            B, S, C_in, H, W = imgs.shape
            imgs_norm = (imgs - model.aggregator._resnet_mean) / model.aggregator._resnet_std
            imgs_norm = imgs_norm.view(B * S, C_in, H, W)
            patch_tokens = model.aggregator.patch_embed(imgs_norm)
            if isinstance(patch_tokens, dict):
                patch_tokens = patch_tokens["x_norm_patchtokens"]
            _, P, C = patch_tokens.shape

            cam_token = slice_expand_and_flatten(model.aggregator.camera_token, B, S)
            reg_token = slice_expand_and_flatten(model.aggregator.register_token, B, S)
            tkns = torch.cat([cam_token, reg_token, patch_tokens], dim=1)

            p = None
            if getattr(model.aggregator, "rope", None) is not None:
                p = model.aggregator.position_getter(
                    B * S, H // model.aggregator.patch_size, W // model.aggregator.patch_size, device=imgs.device
                )

            if getattr(model.aggregator, "patch_start_idx", 0) > 0:
                p = p + 1
                p_special = torch.zeros(B * S, model.aggregator.patch_start_idx, 2, device=imgs.device, dtype=p.dtype)
                p = torch.cat([p_special, p], dim=1)

            _, P, C = tkns.shape
            f_idx = 0
            g_idx = 0
            out_list = []
            needed_layers = set(getattr(getattr(model, "point_head", None), "intermediate_layer_idx", [4, 11, 17, 23])) | {
                model.aggregator.aa_block_num - 1
            }

            for _ in range(model.aggregator.aa_block_num):
                for attn_t in model.aggregator.aa_order:
                    if attn_t == "frame":
                        tkns, f_idx, f_inter = model.aggregator._process_frame_attention(
                            tkns, B, S, P, C, f_idx, pos=p
                        )
                    elif attn_t == "global":
                        tkns, g_idx, g_inter = model.aggregator._process_global_attention(
                            tkns, B, S, P, C, g_idx, pos=p
                        )

                for i in range(len(f_inter)):
                    layer_idx = len(out_list)
                    if layer_idx in needed_layers:
                        concat_inter = torch.cat([f_inter[i], g_inter[i]], dim=-1)
                        out_list.append(concat_inter.cpu())
                    else:
                        out_list.append(None)
                del f_inter, g_inter

            return out_list, getattr(model.aggregator, "patch_start_idx", 0)

        model.aggregator.forward = _optimized_aggregator_forward
        try:
            yield
        finally:
            if hasattr(model, "aggregator"):
                model.aggregator.forward = orig_aggregator_forward


def _load_and_preprocess_vggt_images(
    img_paths: list[str],
    session_dir: str,
    load_and_preprocess_images_fn,
) -> tuple[torch.Tensor, bool]:
    """Load keyframe images for VGGT, rotating to upright display orientation when needed."""
    rot_deg = 0
    sess_manifest = os.path.join(session_dir, "manifest.json")
    if os.path.isfile(sess_manifest):
        try:
            with open(sess_manifest, "r", encoding="utf-8") as f:
                rot_deg = int(json.load(f).get("camera_sensor_to_display_rotation_deg", 0) or 0) % 360
        except Exception:
            rot_deg = 0

    if rot_deg not in (90, 180, 270):
        return load_and_preprocess_images_fn(img_paths), False

    import cv2

    tensors = []
    target_size = 518
    patch_size = 14
    out_w, out_h = None, None
    for p in img_paths:
        bgr = cv2.imread(p, cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError(f"Could not read image: {p}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        if rot_deg == 90:
            rgb = cv2.rotate(rgb, cv2.ROTATE_90_CLOCKWISE)
        elif rot_deg == 180:
            rgb = cv2.rotate(rgb, cv2.ROTATE_180)
        elif rot_deg == 270:
            rgb = cv2.rotate(rgb, cv2.ROTATE_90_COUNTERCLOCKWISE)
        h, w = rgb.shape[:2]
        if out_w is None or out_h is None:
            if h >= w:
                out_h = target_size
                out_w = max(patch_size, int(round((w * (target_size / float(h))) / patch_size) * patch_size))
            else:
                out_w = target_size
                out_h = max(patch_size, int(round((h * (target_size / float(w))) / patch_size) * patch_size))
        resized = cv2.resize(rgb, (out_w, out_h), interpolation=cv2.INTER_AREA)
        t_img = torch.from_numpy(resized).permute(2, 0, 1).float() / 255.0
        tensors.append(t_img)
    return torch.stack(tensors, dim=0), True


def compute_2d_cosine_border_weights(
    h_sub: int,
    w_sub: int,
    h_full: int | None = None,
    w_full: int | None = None,
    patch_size: int = 14,
) -> np.ndarray:
    """Compute 2D cosine border feathering weights w_border(u, v) in [0, 1] over the subsampled grid.

    Derived from ViT patch size P=14 (3 patches = 42 px border band in full resolution max(H, W)).
    """
    if h_sub <= 2 or w_sub <= 2:
        return np.ones((max(1, h_sub), max(1, w_sub)), dtype=np.float32)
    max_dim = float(max(h_full or (h_sub * 2 + 8), w_full or (w_sub * 2 + 8), 3 * patch_size + 1))
    alpha_border = float(np.clip((3.0 * float(patch_size)) / max_dim, 0.03, 0.20))

    v_coords = np.arange(h_sub, dtype=np.float32) / float(max(1, h_sub - 1))
    u_coords = np.arange(w_sub, dtype=np.float32) / float(max(1, w_sub - 1))
    dv = np.minimum(v_coords, 1.0 - v_coords)[:, None]
    du = np.minimum(u_coords, 1.0 - u_coords)[None, :]
    d_edge = np.minimum(dv, du)

    w_border = np.ones((h_sub, w_sub), dtype=np.float32)
    in_band = d_edge < alpha_border
    if np.any(in_band):
        w_border[in_band] = 0.5 * (
            1.0 - np.cos(np.pi * (d_edge[in_band] / alpha_border))
        ).astype(np.float32)
    return w_border


def recover_keyframe_local_camera_grid(
    pts_vggt_grid: np.ndarray,
    conf_grid: np.ndarray,
    ext_3x4: np.ndarray | None = None,
    c_frac: float = 0.50,
    step: int = 4,
) -> np.ndarray:
    """Transform a keyframe's 2D point grid (H_sub, W_sub, 3) from VGGT window coordinates
    into the keyframe's own local camera coordinate system (+X right, +Y down, +Z forward).

    Uses VGGT camera_head extrinsic (R_w2c, t_w2c) directly when non-degenerate and collinear
    with pinhole scanlines, with a ray-normalized SO(3) local-basis fallback on unit rays.
    """
    P_grid = np.asarray(pts_vggt_grid, dtype=np.float64)
    C_grid = np.asarray(conf_grid, dtype=np.float64)
    if P_grid.ndim != 3 or P_grid.shape[2] != 3:
        return P_grid.astype(np.float32)

    H_s, W_s, _ = P_grid.shape
    r0 = int(round((0.5 - 0.5 * c_frac) * H_s))
    r1 = max(r0 + step + 2, int(round((0.5 + 0.5 * c_frac) * H_s)))
    c0 = int(round((0.5 - 0.5 * c_frac) * W_s))
    c1 = max(c0 + step + 2, int(round((0.5 + 0.5 * c_frac) * W_s)))
    r0 = max(0, min(r0, H_s - 2))
    r1 = max(r0 + 2, min(r1, H_s))
    c0 = max(0, min(c0, W_s - 2))
    c1 = max(c0 + 2, min(c1, W_s))
    dk = min(step, max(1, min(r1 - r0 - 1, c1 - c0 - 1)))

    if ext_3x4 is not None:
        ext_arr = np.asarray(ext_3x4, dtype=np.float64)
        if ext_arr.shape == (3, 4) and np.all(np.isfinite(ext_arr)):
            R_w2c = ext_arr[:, :3]
            t_w2c = ext_arr[:, 3]
            if float(np.linalg.det(R_w2c)) >= 0.10:
                P_cam = P_grid @ R_w2c.T + t_w2c
                valid = np.isfinite(P_cam).all(axis=-1) & np.isfinite(C_grid) & (C_grid > 1.05)
                if not np.any(valid):
                    valid = np.isfinite(P_cam).all(axis=-1)
                if np.any(valid) and float(np.median(P_cam[valid, 2])) > 0.05:
                    return P_cam.astype(np.float32)

    # Fallback: ray-normalized SO(3) local camera frame recovery on unit rays
    norms = np.maximum(np.linalg.norm(P_grid, axis=-1, keepdims=True), 1e-6)
    P_unit = P_grid / norms
    crop_u = P_unit[r0:r1, c0:c1]
    crop_c = C_grid[r0:r1, c0:c1]
    valid_c = np.isfinite(crop_u).all(axis=-1) & (crop_c > 1.05)

    vz = np.nanmedian(crop_u[valid_c], axis=0) if np.any(valid_c) else np.nanmedian(crop_u.reshape(-1, 3), axis=0)
    dx_arr = crop_u[:, dk:, :] - crop_u[:, :-dk, :]
    dy_arr = crop_u[dk:, :, :] - crop_u[:-dk, :, :]
    vx = np.nanmedian(dx_arr.reshape(-1, 3), axis=0)
    vy = np.nanmedian(dy_arr.reshape(-1, 3), axis=0)

    nz = float(np.linalg.norm(vz))
    if nz < 1e-6 or not np.all(np.isfinite(vz)):
        return P_grid.astype(np.float32)
    ez = vz / nz
    vx_proj = vx - float(np.dot(vx, ez)) * ez
    nx = float(np.linalg.norm(vx_proj))
    if nx < 1e-6 or not np.all(np.isfinite(vx_proj)):
        return P_grid.astype(np.float32)
    ex = vx_proj / nx
    ey = np.cross(ez, ex)
    if float(np.dot(ey, vy)) < 0.0:
        ex = -ex
        ey = -ey
    R_rect = np.column_stack([ex, ey, ez])
    U_s, _, Vt_s = np.linalg.svd(R_rect)
    R_rect = U_s @ np.diag([1.0, 1.0, float(np.linalg.det(U_s @ Vt_s))]) @ Vt_s
    return (P_grid @ R_rect).astype(np.float32)


def refine_optical_axis_1dof_icp(
    pts_by_frame: dict[int, np.ndarray],
    conf_by_frame: dict[int, np.ndarray],
    rots_u: np.ndarray,
    t_vios: np.ndarray,
    max_shift_m: float = 0.025,
    lambda_reg: float = 0.25,
) -> dict[int, float]:
    """Solve bounded 1-DoF optical-axis translations delta_zeta_g in [-max_shift_m, +max_shift_m]
    across covisible keyframes using high-confidence point-to-plane residuals.
    """
    from scipy.spatial import cKDTree

    f_keys = sorted(pts_by_frame.keys())
    if len(f_keys) < 3:
        return {k: 0.0 for k in f_keys}

    rng = np.random.default_rng(42)
    sub_pts_list = []
    sub_nrm_list = []
    sub_fid_list = []

    for g_idx in f_keys:
        pts_g = pts_by_frame[g_idx]
        cf_g = conf_by_frame[g_idx]
        hi = cf_g > 1.35
        if int(np.count_nonzero(hi)) < 40:
            hi = cf_g > 1.15
        p_hi = pts_g[hi]
        if len(p_hi) < 25:
            continue
        if len(p_hi) > 500:
            sel = np.sort(rng.choice(len(p_hi), size=500, replace=False))
            p_hi = p_hi[sel]
        # Estimate local normals via PCA in 8cm radius
        tree_g = cKDTree(p_hi)
        _, nn_idx = tree_g.query(p_hi, k=min(12, len(p_hi)))
        neigh = p_hi[nn_idx]
        centered = neigh - np.mean(neigh, axis=1, keepdims=True)
        cov = np.einsum("nki,nkj->nij", centered, centered)
        _, _, vh = np.linalg.svd(cov)
        nrms = vh[:, -1, :]
        # Orient normals toward keyframe camera center t_vios[g_idx]
        view_vec = t_vios[g_idx][None, :] - p_hi
        flip = np.sum(nrms * view_vec, axis=1) < 0.0
        nrms[flip] = -nrms[flip]
        sub_pts_list.append(p_hi)
        sub_nrm_list.append(nrms)
        sub_fid_list.append(np.full(len(p_hi), g_idx, dtype=np.int32))

    if len(sub_pts_list) < 3:
        return {k: 0.0 for k in f_keys}

    all_p = np.concatenate(sub_pts_list, axis=0)
    all_n = np.concatenate(sub_nrm_list, axis=0)
    all_f = np.concatenate(sub_fid_list, axis=0)

    tree_all = cKDTree(all_p)
    dists, idxs = tree_all.query(all_p, k=6, distance_upper_bound=0.035)

    f_to_pos = {g_idx: i for i, g_idx in enumerate(f_keys)}
    n_f = len(f_keys)
    pair_res: dict[tuple[int, int], list[tuple[np.ndarray, float]]] = {}

    for k_nbr in range(1, 6):
        valid = np.isfinite(dists[:, k_nbr]) & (idxs[:, k_nbr] < len(all_p))
        src_i = np.flatnonzero(valid)
        dst_i = idxs[src_i, k_nbr]
        diff_f = all_f[src_i] < all_f[dst_i]
        src_i = src_i[diff_f]
        dst_i = dst_i[diff_f]
        if len(src_i) == 0:
            continue
        n_dot = np.sum(all_n[src_i] * all_n[dst_i], axis=1)
        ok_n = n_dot >= 0.92  # ~23 deg normal agreement
        src_i = src_i[ok_n]
        dst_i = dst_i[ok_n]
        if len(src_i) == 0:
            continue
        n_avg = all_n[src_i] + all_n[dst_i]
        n_avg /= np.maximum(np.linalg.norm(n_avg, axis=1, keepdims=True), 1e-6)
        r_val = np.sum(n_avg * (all_p[dst_i] - all_p[src_i]), axis=1)
        ok_r = np.abs(r_val) <= 0.030
        for s_idx, d_idx, nv, rv in zip(src_i[ok_r], dst_i[ok_r], n_avg[ok_r], r_val[ok_r]):
            pair_key = (int(all_f[s_idx]), int(all_f[d_idx]))
            pair_res.setdefault(pair_key, []).append((nv, float(rv)))

    A_rows = []
    b_rows = []
    reg_w = float(np.sqrt(max(lambda_reg, 1e-4)))
    for i in range(n_f):
        row = np.zeros(n_f, dtype=np.float64)
        row[i] = reg_w
        A_rows.append(row)
        b_rows.append(0.0)

    n_pairs_used = 0
    for (g_i, g_j), items in pair_res.items():
        if len(items) < 12:
            continue
        n_arr = np.asarray([it[0] for it in items], dtype=np.float64)
        r_arr = np.asarray([it[1] for it in items], dtype=np.float64)
        n_bar = np.median(n_arr, axis=0)
        n_norm = float(np.linalg.norm(n_bar))
        if n_norm < 1e-6:
            continue
        n_bar /= n_norm
        r_med = float(np.median(r_arr))
        zg = rots_u[g_i][:, 2]
        zh = rots_u[g_j][:, 2]
        cg = float(np.dot(n_bar, zg))
        ch = float(np.dot(n_bar, zh))
        if abs(cg) < 0.15 and abs(ch) < 0.15:
            continue
        w_pair = min(2.0, float(np.sqrt(len(items) / 20.0)))
        row = np.zeros(n_f, dtype=np.float64)
        row[f_to_pos[g_i]] = w_pair * cg
        row[f_to_pos[g_j]] = -w_pair * ch
        A_rows.append(row)
        b_rows.append(w_pair * r_med)
        n_pairs_used += 1

    if n_pairs_used == 0:
        return {k: 0.0 for k in f_keys}

    dz, _, _, _ = np.linalg.lstsq(np.asarray(A_rows), np.asarray(b_rows), rcond=None)
    dz = dz - float(np.mean(dz))
    dz = np.clip(dz, -float(max_shift_m), float(max_shift_m))
    return {f_keys[i]: float(dz[i]) for i in range(n_f)}


def harmonize_multi_window_rgb_exposure(
    pts_vio: np.ndarray,
    cols_all: np.ndarray,
    fidx_all: np.ndarray,
    voxel_size: float = 0.020,
    lambda_color: float = 0.10,
    gain_bounds: tuple[float, float] = (0.82, 1.22),
) -> np.ndarray:
    """Global per-keyframe RGB exposure gain harmonization across overlapping 2cm voxels."""
    if len(pts_vio) == 0 or len(cols_all) != len(pts_vio) or len(fidx_all) != len(pts_vio):
        return cols_all
    unique_f = np.unique(fidx_all)
    if len(unique_f) < 2:
        return cols_all

    rgb_f = cols_all.astype(np.float64) / 255.0
    valid_c = np.all((rgb_f >= 0.05) & (rgb_f <= 0.95), axis=1)
    if int(np.count_nonzero(valid_c)) < 500:
        return cols_all

    p_val = pts_vio[valid_c]
    c_val = rgb_f[valid_c]
    f_val = fidx_all[valid_c]

    p_min = np.min(p_val, axis=0)
    q_ijk = np.floor((p_val - p_min) / float(voxel_size)).astype(np.int64)
    vox_hash = (
        q_ijk[:, 0] * 73856093
        ^ q_ijk[:, 1] * 19349663
        ^ q_ijk[:, 2] * 83492791
    )

    # Sort by (vox_hash, f_val) to aggregate mean RGB per (voxel, frame)
    order = np.lexsort((f_val, vox_hash))
    vh_s = vox_hash[order]
    fv_s = f_val[order]
    cv_s = c_val[order]

    change = np.empty(len(vh_s), dtype=bool)
    change[0] = True
    change[1:] = (vh_s[1:] != vh_s[:-1]) | (fv_s[1:] != fv_s[:-1])
    grp_starts = np.flatnonzero(change)
    grp_counts = np.diff(np.append(grp_starts, len(vh_s)))

    grp_vh = vh_s[grp_starts]
    grp_fv = fv_s[grp_starts]
    grp_rgb = np.add.reduceat(cv_s, grp_starts, axis=0) / grp_counts[:, None]

    # Keep only voxels observed by >= 2 distinct keyframes
    v_change = np.empty(len(grp_vh), dtype=bool)
    v_change[0] = True
    v_change[1:] = grp_vh[1:] != grp_vh[:-1]
    v_starts = np.flatnonzero(v_change)
    v_counts = np.diff(np.append(v_starts, len(grp_vh)))
    multi_v = np.flatnonzero(v_counts >= 2)
    if len(multi_v) < 15:
        return cols_all

    f_to_pos = {int(fid): i for i, fid in enumerate(unique_f)}
    n_f = len(unique_f)
    pair_ratios: dict[tuple[int, int], list[np.ndarray]] = {}

    for mv_i in multi_v:
        st = int(v_starts[mv_i])
        cnt = int(v_counts[mv_i])
        # Connect consecutive keyframes in the same voxel
        for a in range(cnt - 1):
            fa = int(grp_fv[st + a])
            fb = int(grp_fv[st + a + 1])
            if fa == fb:
                continue
            ia = grp_rgb[st + a]
            ib = grp_rgb[st + a + 1]
            if fa < fb:
                pair_ratios.setdefault((fa, fb), []).append(np.log(ib) - np.log(ia))
            else:
                pair_ratios.setdefault((fb, fa), []).append(np.log(ia) - np.log(ib))

    A_rows = []
    B_mat = []
    reg_w = float(np.sqrt(max(lambda_color, 1e-4)))
    for i in range(n_f):
        row = np.zeros(n_f, dtype=np.float64)
        row[i] = reg_w
        A_rows.append(row)
        B_mat.append([0.0, 0.0, 0.0])

    n_edges = 0
    for (fa, fb), r_list in pair_ratios.items():
        if len(r_list) < 10:
            continue
        med_dl = np.median(np.asarray(r_list, dtype=np.float64), axis=0)
        w_e = min(2.0, float(np.sqrt(len(r_list) / 15.0)))
        row = np.zeros(n_f, dtype=np.float64)
        row[f_to_pos[fa]] = w_e
        row[f_to_pos[fb]] = -w_e
        A_rows.append(row)
        B_mat.append((w_e * med_dl).tolist())
        n_edges += 1

    if n_edges == 0:
        return cols_all

    A_arr = np.asarray(A_rows, dtype=np.float64)
    B_arr = np.asarray(B_mat, dtype=np.float64)
    gamma, _, _, _ = np.linalg.lstsq(A_arr, B_arr, rcond=None)
    gamma = gamma - np.mean(gamma, axis=0, keepdims=True)
    gains = np.clip(np.exp(gamma), float(gain_bounds[0]), float(gain_bounds[1]))

    gain_lut = np.ones((int( np.max(unique_f)) + 1, 3), dtype=np.float32)
    for i, fid in enumerate(unique_f):
        gain_lut[int(fid)] = gains[i].astype(np.float32)

    harmonized = np.clip(np.round(cols_all.astype(np.float32) * gain_lut[fidx_all]), 0.0, 255.0).astype(np.uint8)
    return harmonized


def _run_covisible_multi_window_vggt(
    model,
    images: torch.Tensor,
    is_upright: bool,
    frames: list[dict],
    session_dir: str,
    max_points: int,
    device: str,
    dtype: torch.dtype,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float] | None:
    """Run VGGT over overlapping covisible viewing-direction windows for wide-yaw
    360-degree scans so keyframes >70 deg from Frame 0 do not collapse to conf == 1.0."""
    from scipy.spatial.transform import Rotation as _Rot
    try:
        from vggt.utils.pose_enc import pose_encoding_to_extri_intri
    except ImportError:
        pose_encoding_to_extri_intri = None

    n_seq = int(images.shape[1])
    if n_seq < 12 or len(frames) < n_seq:
        return None
    if not all(all(k in frames[i] for k in ("qx", "qy", "qz", "qw")) for i in range(n_seq)):
        return None

    rot_deg = 0
    sess_manifest = os.path.join(session_dir, "manifest.json")
    if os.path.isfile(sess_manifest):
        try:
            with open(sess_manifest, "r", encoding="utf-8") as f:
                rot_deg = int(json.load(f).get("camera_sensor_to_display_rotation_deg", 0) or 0) % 360
        except Exception:
            rot_deg = 0

    R_su = (
        np.array([[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
        if is_upright
        else np.eye(3, dtype=np.float64)
    )
    rots_u = []
    t_vios = []
    for i in range(n_seq):
        fr = frames[i]
        r_wc = _Rot.from_quat([
            float(fr["qx"]),
            float(fr["qy"]),
            float(fr["qz"]),
            float(fr["qw"]),
        ]).as_matrix()
        r_u = r_wc @ R_su
        rots_u.append(r_u)
        t_vios.append([float(fr.get("x", 0.0)), float(fr.get("y", 0.0)), float(fr.get("z", 0.0))])

    rots_u = np.asarray(rots_u, dtype=np.float64)
    t_vios = np.asarray(t_vios, dtype=np.float64)
    fwd_all = rots_u[:, :, 2]
    if abs(float(np.median(rots_u[:, 1, 1]))) > abs(float(np.median(rots_u[:, 2, 1]))):
        # Y-up world (ARCore odometry): horizontal plane is (X, -Z), vertical is Y
        yaws_deg = np.degrees(np.arctan2(fwd_all[:, 0], -fwd_all[:, 2])).astype(np.float64)
        pitches_deg = np.degrees(np.arcsin(np.clip(fwd_all[:, 1], -1.0, 1.0))).astype(np.float64)
    else:
        # Z-up world (Server VIO): horizontal plane is (X, Y), vertical is Z
        yaws_deg = np.degrees(np.arctan2(fwd_all[:, 1], fwd_all[:, 0])).astype(np.float64)
        pitches_deg = np.degrees(np.arcsin(np.clip(fwd_all[:, 2], -1.0, 1.0))).astype(np.float64)

    # Check circular yaw coverage: only use multi-window mode when yaw span > 110 deg
    sorted_yaws = np.sort(yaws_deg)
    gaps = np.diff(np.concatenate([sorted_yaws, [sorted_yaws[0] + 360.0]]))
    yaw_span_deg = 360.0 - float(np.max(gaps))
    if yaw_span_deg <= 110.0:
        return None

    img_gray = images[0].float().mean(dim=1)
    _, H_g, W_g = img_gray.shape
    dx = (img_gray[:, :, 1:] - img_gray[:, :, :-1]).abs()
    dy = (img_gray[:, 1:, :] - img_gray[:, :-1, :]).abs()
    gh, gw = max(1, H_g // 4), max(1, W_g // 4)
    cell_stds = []
    for r_i in range(4):
        for c_i in range(4):
            pdx = dx[:, r_i * gh : (r_i + 1) * gh, c_i * gw : min((c_i + 1) * gw, W_g - 1)]
            pdy = dy[:, r_i * gh : min((r_i + 1) * gh, H_g - 1), c_i * gw : (c_i + 1) * gw]
            cell_stds.append((pdx.mean(dim=(1, 2)) + pdy.mean(dim=(1, 2))).cpu().numpy())
    cell_stds_arr = np.stack(cell_stds, axis=1)
    sharp_scores = np.percentile(cell_stds_arr, 35, axis=1) * np.percentile(cell_stds_arr, 75, axis=1)

    # Load 8x8 dToF depths and ray-projected status==5 zones per frame if processed_lidar.csv exists
    lidar_depths_by_idx: dict[int, float] = {}
    lidar_s5_zones_by_idx: dict[int, list[tuple[float, float, float]]] = {}
    lidar_csv = os.path.join(session_dir, "processed_lidar.csv")
    vio_csv = os.path.join(session_dir, "processed_vio.csv")
    cam_mat_csv = os.path.join(session_dir, "camera_matrix.csv")
    if os.path.isfile(lidar_csv) and os.path.getsize(lidar_csv) > 0:
        try:
            import pandas as pd
            from app.services.calibration import (
                load_camera_intrinsics,
                load_session_extrinsics,
                probe_session_image_size,
                project_lidar_frame_to_sparse_tof_with_status,
            )

            K_sens = load_camera_intrinsics(cam_mat_csv) if os.path.isfile(cam_mat_csv) else None
            T_lidar_cam = None
            lidar_rays = None
            flip_h_meta, flip_v_meta = False, False
            if K_sens is not None:
                try:
                    _, T_lidar_cam, ext_meta = load_session_extrinsics(session_dir)
                    lidar_rays = ext_meta.get("rays")
                    flip_h_meta = bool(ext_meta.get("flip_h", False))
                    flip_v_meta = bool(ext_meta.get("flip_v", False))
                except Exception:
                    T_lidar_cam = None

            ts_by_frame: dict[int, float] = {}
            if os.path.isfile(vio_csv) and os.path.getsize(vio_csv) > 0:
                vdf = pd.read_csv(vio_csv)
                if "frame" in vdf.columns and "device_timestamp_ns" in vdf.columns:
                    ts_by_frame = {
                        int(r["frame"]): float(r["device_timestamp_ns"]) for _, r in vdf.iterrows()
                    }
            ldf = pd.read_csv(lidar_csv)
            ts_col = (
                "device_timestamp_ns"
                if "device_timestamp_ns" in ldf.columns
                else ("timestamp" if "timestamp" in ldf.columns else ldf.columns[0])
            )
            l_ts = ldf[ts_col].to_numpy(dtype=np.float64)
            dist_cols = [f"distance_{z}" for z in range(64) if f"distance_{z}" in ldf.columns]
            stat_cols = [f"status_{z}" for z in range(64) if f"status_{z}" in ldf.columns]
            if len(dist_cols) == 64:
                d_mat = ldf[dist_cols].to_numpy(dtype=np.float64)
                if np.nanmedian(d_mat) > 20.0:
                    d_mat = d_mat / 1000.0
                s_mat = ldf[stat_cols].to_numpy(dtype=np.float64) if len(stat_cols) == 64 else np.full_like(d_mat, 5.0)
                w_sens, h_sens = probe_session_image_size(
                    session_dir=session_dir, K=K_sens, default_size=(640, 480)
                )
                for idx_f in range(n_seq):
                    fr_f = frames[idx_f]
                    fts = float(fr_f.get("device_timestamp_ns", fr_f.get("timestamp", 0.0)) or 0.0)
                    if fts <= 0.0 and "source_frame_index" in fr_f:
                        fts = ts_by_frame.get(int(fr_f["source_frame_index"]), 0.0)
                    if fts <= 0.0:
                        continue
                    row_i = int(np.argmin(np.abs(l_ts - fts)))
                    d_row = d_mat[row_i]
                    s_row = s_mat[row_i]
                    # Strictly prefer status == 5 hardware-validated SPAD zones
                    valid_s5 = d_row[np.isfinite(d_row) & (d_row > 0.18) & (d_row < 3.8) & (s_row == 5)]
                    if len(valid_s5) >= 4:
                        lidar_depths_by_idx[idx_f] = float(np.median(valid_s5))
                    else:
                        valid_d = d_row[np.isfinite(d_row) & (d_row > 0.18) & (d_row < 3.8) & np.isin(s_row, [5, 9])]
                        if len(valid_d) >= 6:
                            lidar_depths_by_idx[idx_f] = float(np.median(valid_d))
                    if K_sens is not None and T_lidar_cam is not None and lidar_rays is not None:
                        tof_map, st_map = project_lidar_frame_to_sparse_tof_with_status(
                            d_row,
                            s_row,
                            lidar_rays,
                            T_lidar_cam,
                            K_sens,
                            h_sens,
                            w_sens,
                            flip_h=flip_h_meta,
                            flip_v=flip_v_meta,
                            rot_deg=rot_deg,
                        )
                        if is_upright and rot_deg in (90, 180, 270):
                            k_rot = 3 if rot_deg == 90 else (2 if rot_deg == 180 else 1)
                            tof_map = np.rot90(tof_map, k=k_rot)
                            st_map = np.rot90(st_map, k=k_rot)
                        rs_5, cs_5 = np.where((st_map == 5) & (tof_map > 0.18) & (tof_map < 3.80))
                        if len(rs_5) >= 4:
                            h_up_m, w_up_m = tof_map.shape
                            zones_f = [
                                (float(r_p) / float(h_up_m), float(c_p) / float(w_up_m), float(tof_map[r_p, c_p]))
                                for r_p, c_p in zip(rs_5, cs_5)
                            ]
                            lidar_s5_zones_by_idx[idx_f] = zones_f
        except Exception:
            lidar_depths_by_idx = {}
            lidar_s5_zones_by_idx = {}

    def _estimate_ray_aligned_tof_scale(
        g_idx: int,
        P_cam_grid: np.ndarray,
        C_grid: np.ndarray,
        H_full: int,
        W_full: int,
    ) -> float | None:
        """Compute ray-aligned dToF scale s_g = median_{z in Z_5}(Z_tof / Z_cam) on the 2D camera grid."""
        H_sub, W_sub, _ = P_cam_grid.shape
        Z_grid = P_cam_grid[:, :, 2]
        if g_idx in lidar_s5_zones_by_idx:
            ratios_z = []
            for r_norm, c_norm, z_tof in lidar_s5_zones_by_idx[g_idx]:
                if H_full > 16 and W_full > 16:
                    r_sub = int(round((r_norm * float(H_full) - 4.0) / 2.0))
                    c_sub = int(round((c_norm * float(W_full) - 4.0) / 2.0))
                else:
                    r_sub = int(round((r_norm * float(H_full)) / 2.0))
                    c_sub = int(round((c_norm * float(W_full)) / 2.0))
                r0 = max(0, r_sub - 1)
                r1 = min(H_sub, r_sub + 2)
                c0 = max(0, c_sub - 1)
                c1 = min(W_sub, c_sub + 2)
                if r1 <= r0 or c1 <= c0:
                    continue
                z_patch = Z_grid[r0:r1, c0:c1]
                c_patch = C_grid[r0:r1, c0:c1]
                ok_hi = np.isfinite(z_patch) & (z_patch > 0.15) & (c_patch > 1.35)
                if not np.any(ok_hi):
                    ok_hi = np.isfinite(z_patch) & (z_patch > 0.15) & (c_patch > 1.10)
                if np.any(ok_hi):
                    z_cam_med = float(np.median(z_patch[ok_hi]))
                    if z_cam_med > 0.15:
                        ratios_z.append(float(z_tof / z_cam_med))
            if len(ratios_z) >= 4:
                return float(np.clip(np.median(ratios_z), 0.70, 1.40))

        if g_idx in lidar_depths_by_idx:
            fin_c = np.isfinite(P_cam_grid).all(axis=-1) & (C_grid > 1.15)
            if int(np.count_nonzero(fin_c)) >= 60:
                p_sub = P_cam_grid[fin_c]
                z_all = p_sub[:, 2]
                cone_m = (
                    (z_all > 0.15)
                    & (np.abs(p_sub[:, 0]) < 0.35 * z_all)
                    & (np.abs(p_sub[:, 1]) < 0.35 * z_all)
                )
                z_pos = z_all[cone_m] if int(np.count_nonzero(cone_m)) >= 60 else z_all[z_all > 0.15]
                if len(z_pos) >= 60:
                    return float(np.clip(lidar_depths_by_idx[g_idx] / float(np.median(z_pos)), 0.70, 1.40))
        return None

    def _run_window(w_indices: list[int]) -> dict:
        anc = int(w_indices[0])
        w_imgs = images[:, torch.tensor(w_indices, dtype=torch.long)].to(device, dtype=dtype)
        f_dict: dict[int, dict] = {}
        scale_samples: list[float] = []
        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=(device == "cuda"), dtype=dtype):
                agg_list, p_start = model.aggregator(w_imgs)
                _, S_w, _, H, W = w_imgs.shape
                chunk_sz = 4
                w_border_grid = None
                for s_i in range(0, S_w, chunk_sz):
                    e_i = min(s_i + chunk_sz, S_w)
                    c_toks = [
                        agg_list[layer_i][:, s_i:e_i].to(device)
                        if layer_i in model.point_head.intermediate_layer_idx
                        else None
                        for layer_i in range(len(agg_list))
                    ]
                    img_c = w_imgs[:, s_i:e_i]
                    pts_c, conf_c = model.point_head._forward_impl(c_toks, img_c, p_start, 0, e_i - s_i)
                    pts_c = pts_c.float()
                    conf_c = conf_c.float()
                    if H > 16 and W > 16:
                        pts_sub = pts_c[:, :, 4:-4:2, 4:-4:2, :]
                        conf_sub = conf_c[:, :, 4:-4:2, 4:-4:2]
                        rgb_sub = img_c.permute(0, 1, 3, 4, 2)[:, :, 4:-4:2, 4:-4:2, :]
                    else:
                        pts_sub = pts_c[:, :, ::2, ::2, :]
                        conf_sub = conf_c[:, :, ::2, ::2]
                        rgb_sub = img_c.permute(0, 1, 3, 4, 2)[:, :, ::2, ::2, :]
                    if w_border_grid is None:
                        w_border_grid = compute_2d_cosine_border_weights(
                            int(pts_sub.shape[2]), int(pts_sub.shape[3]), h_full=int(H), w_full=int(W)
                        )
                    w_border_flat = w_border_grid.reshape(-1)
                    for loc_f in range(e_i - s_i):
                        w_pos = s_i + loc_f
                        g_idx = int(w_indices[w_pos])
                        P_vggt_grid = pts_sub[0, loc_f].cpu().numpy()
                        C_raw_grid = conf_sub[0, loc_f].cpu().numpy()

                        cf_raw = C_raw_grid.reshape(-1)
                        # Module 3.1: 2D cosine border feathering on excess confidence above 1.0
                        cf = np.where(
                            cf_raw > 1.0,
                            1.0 + (cf_raw - 1.0) * w_border_flat,
                            cf_raw,
                        ).astype(np.float32)
                        # Drop extreme border curl where w_border < 0.15 and raw confidence < 1.35
                        cf[(w_border_flat < 0.15) & (cf_raw < 1.35)] = 0.0

                        pf = P_vggt_grid.reshape(-1, 3)
                        rf = rgb_sub[0, loc_f].float().clamp(0.0, 1.0).reshape(-1, 3).cpu().numpy()
                        col = np.round(rf * 255.0).astype(np.uint8)
                        fin = np.isfinite(pf).all(axis=-1) & np.isfinite(cf_raw)
                        ratio = float(np.mean(fin & (cf_raw > 1.05)))
                        f_dict[g_idx] = {
                            "pf": pf,
                            "P_grid": P_vggt_grid,
                            "C_grid": C_raw_grid,
                            "H_full": int(H),
                            "W_full": int(W),
                            "cf": cf,
                            "cf_raw": cf_raw,
                            "col": col,
                            "ratio": ratio,
                        }
                        if g_idx in lidar_depths_by_idx and ratio >= 0.35:
                            R_anc_to_g = rots_u[anc].T @ rots_u[g_idx]
                            P_rel_grid = P_vggt_grid @ R_anc_to_g
                            sc_val = _estimate_ray_aligned_tof_scale(g_idx, P_rel_grid, C_raw_grid, int(H), int(W))
                            if sc_val is not None:
                                scale_samples.append(sc_val)
                                if g_idx == anc:
                                    scale_samples.append(sc_val)
        del w_imgs, agg_list
        if device == "cuda":
            torch.cuda.empty_cache()
        s_w = float(np.median(scale_samples)) if scale_samples else 1.05
        return {"anc": anc, "s_w": float(np.clip(s_w, 0.55, 1.85)), "frames": f_dict}

    sharp_p25 = float(np.percentile(sharp_scores, 25.0)) if len(sharp_scores) > 0 else 0.0
    window_centers = [-150.0, -90.0, -30.0, 30.0, 90.0, 150.0]
    win_data: dict[int, dict] = {}
    for w_idx, wc in enumerate(window_centers):
        dyaw = np.abs((yaws_deg - wc + 180.0) % 360.0 - 180.0)
        cand = np.flatnonzero(dyaw <= 62.0).tolist()
        if len(cand) < 4:
            continue
        anc_cands = [
            i for i in cand
            if dyaw[i] <= 28.0 and abs(pitches_deg[i]) <= 38.0 and sharp_scores[i] >= sharp_p25
        ]
        if not anc_cands:
            anc_cands = [i for i in cand if dyaw[i] <= 28.0 and abs(pitches_deg[i]) <= 38.0]
        if not anc_cands:
            anc_cands = cand
        anc_scores = np.array([
            sharp_scores[i] * float(np.cos(np.radians(dyaw[i] * 0.8)))
            for i in anc_cands
        ])
        anc = int(anc_cands[int(np.argmax(anc_scores))])
        rem = [i for i in cand if i != anc]
        rem.sort(key=lambda i: -float(np.trace(rots_u[anc].T @ rots_u[i])))
        w_indices = [anc] + rem[:17]
        win_data[w_idx] = _run_window(w_indices)

    # Rescue local covisible windows for any keyframes that did not achieve confident cross-attention
    tried_rescue_anchors: set[int] = set()
    next_w_idx = len(window_centers)
    for _ in range(16):
        best_ratio_by_f = {
            g_i: max((wd["frames"].get(g_i, {}).get("ratio", 0.0) for wd in win_data.values()), default=0.0)
            for g_i in range(n_seq)
        }
        in_any_win = {
            g_i: any(g_i in wd["frames"] for wd in win_data.values())
            for g_i in range(n_seq)
        }
        uncovered = [
            g_i for g_i in range(n_seq)
            if (best_ratio_by_f[g_i] < 0.45 or not in_any_win[g_i]) and g_i not in tried_rescue_anchors
        ]
        if not uncovered:
            break
        never_seen = [g_i for g_i in uncovered if not in_any_win[g_i]]
        unrep = [g_i for g_i in uncovered if best_ratio_by_f[g_i] < 0.15]
        base_pool = never_seen if never_seen else (unrep if unrep else uncovered)
        sharp_unc = [g_i for g_i in base_pool if sharp_scores[g_i] >= sharp_p25]
        pool_unc = sharp_unc if sharp_unc else base_pool
        u_anc = max(
            pool_unc,
            key=lambda i: float(sharp_scores[i]) * (0.5 if abs(pitches_deg[i]) > 38.0 else 1.0),
        )
        tried_rescue_anchors.add(int(u_anc))
        rem_u = [i for i in range(n_seq) if i != u_anc]
        rem_u.sort(
            key=lambda i: -(
                (1.50 if (not in_any_win[i] and float(rots_u[u_anc][:, 2] @ rots_u[i][:, 2]) > 0.15) else 0.0)
                + (1.10 if (best_ratio_by_f[i] < 0.15 and float(rots_u[u_anc][:, 2] @ rots_u[i][:, 2]) > 0.25) else 0.0)
                + (0.35 if (best_ratio_by_f[i] < 0.35 and float(rots_u[u_anc][:, 2] @ rots_u[i][:, 2]) > 0.25) else 0.0)
                + float(rots_u[u_anc][:, 2] @ rots_u[i][:, 2])
                + 0.25 * float(np.trace(rots_u[u_anc].T @ rots_u[i]))
            )
        )
        w_rescue = [int(u_anc)] + rem_u[:15]
        win_data[next_w_idx] = _run_window(w_rescue)
        next_w_idx += 1

    if len(win_data) < 2:
        return None

    win_keys = sorted(win_data.keys())
    n_w = len(win_keys)
    # 1. Reconcile window scales using dToF prior + centered 3D spread on shared confident pixels
    A_rows, b_rows, w_rows = [], [], []
    for ii, k_i in enumerate(win_keys):
        row = np.zeros(n_w, dtype=np.float64)
        row[ii] = 1.0
        A_rows.append(row)
        b_rows.append(np.log(win_data[k_i]["s_w"]))
        w_rows.append(1.5)
    for ii in range(n_w):
        for jj in range(ii + 1, n_w):
            fi = win_data[win_keys[ii]]["frames"]
            fj = win_data[win_keys[jj]]["frames"]
            shared = sorted(set(fi.keys()).intersection(fj.keys()))
            ratios = []
            for g_idx in shared:
                mask = (fi[g_idx]["cf"] > 1.25) & (fj[g_idx]["cf"] > 1.25)
                if int(mask.sum()) > 500:
                    pi = fi[g_idx]["pf"][mask]
                    pj = fj[g_idx]["pf"][mask]
                    di = np.linalg.norm(pi - np.median(pi, axis=0), axis=1)
                    dj = np.linalg.norm(pj - np.median(pj, axis=0), axis=1)
                    valid = (di > 0.05) & (dj > 0.05)
                    if int(valid.sum()) > 200:
                        ratios.append(float(np.median(dj[valid] / di[valid])))
            if ratios:
                row = np.zeros(n_w, dtype=np.float64)
                row[ii] = 1.0
                row[jj] = -1.0
                A_rows.append(row)
                b_rows.append(np.log(float(np.median(ratios))))
                w_rows.append(2.0)
    W_mat = np.diag(w_rows)
    log_s = np.linalg.lstsq(W_mat @ np.asarray(A_rows), W_mat @ np.asarray(b_rows), rcond=None)[0]
    s_arr = np.exp(log_s)
    s_ref = float(np.median([win_data[k]["s_w"] for k in win_keys]))
    s_opt = {win_keys[ii]: float(s_arr[ii]) for ii in range(n_w)}

    # 2. Reconcile window anchor translations around VIO camera centers
    t_prior = np.array([t_vios[win_data[k]["anc"]] for k in win_keys], dtype=np.float64)
    Tx, bx, by, bz, tw = [], [], [], [], []
    for ii in range(n_w):
        row = np.zeros(n_w, dtype=np.float64)
        row[ii] = 1.0
        Tx.append(row)
        bx.append(t_prior[ii, 0])
        by.append(t_prior[ii, 1])
        bz.append(t_prior[ii, 2])
        tw.append(1.0)
    for ii in range(n_w):
        for jj in range(ii + 1, n_w):
            ki, kj = win_keys[ii], win_keys[jj]
            ai, aj = win_data[ki]["anc"], win_data[kj]["anc"]
            Ri, Rj = rots_u[ai], rots_u[aj]
            si, sj = s_opt[ki], s_opt[kj]
            fi, fj = win_data[ki]["frames"], win_data[kj]["frames"]
            for g_idx in sorted(set(fi.keys()).intersection(fj.keys())):
                mask = (fi[g_idx]["cf"] > 1.25) & (fj[g_idx]["cf"] > 1.25)
                if int(mask.sum()) > 500:
                    diff = np.median(
                        sj * (fj[g_idx]["pf"][mask] @ Rj.T) - si * (fi[g_idx]["pf"][mask] @ Ri.T),
                        axis=0,
                    )
                    row = np.zeros(n_w, dtype=np.float64)
                    row[ii] = 1.0
                    row[jj] = -1.0
                    Tx.append(row)
                    bx.append(float(diff[0]))
                    by.append(float(diff[1]))
                    bz.append(float(diff[2]))
                    tw.append(2.0)
    Wt = np.diag(tw)
    At = np.asarray(Tx, dtype=np.float64)
    t_arr = np.column_stack([
        np.linalg.lstsq(Wt @ At, Wt @ np.asarray(bx), rcond=None)[0],
        np.linalg.lstsq(Wt @ At, Wt @ np.asarray(by), rcond=None)[0],
        np.linalg.lstsq(Wt @ At, Wt @ np.asarray(bz), rcond=None)[0],
    ])
    t_arr -= np.mean(t_arr - t_prior, axis=0)
    t_opt = {win_keys[ii]: t_arr[ii].copy() for ii in range(n_w)}

    # 3. Assign each frame to the window with highest confidence-angle score
    chosen: dict[int, tuple[float, int]] = {}
    for k in win_keys:
        anc = win_data[k]["anc"]
        for g_idx, fd in win_data[k]["frames"].items():
            if fd["ratio"] < 0.15:
                continue
            ang = float(np.degrees(np.arccos(np.clip((np.trace(rots_u[anc].T @ rots_u[g_idx]) - 1.0) / 2.0, -1.0, 1.0))))
            sc = fd["ratio"] - 0.008 * ang
            if g_idx not in chosen or sc > chosen[g_idx][0]:
                chosen[g_idx] = (sc, k)
    # Fallback assignment for any keyframe with lower confident-pixel ratio so 100% of keyframes are represented
    for g_idx in range(n_seq):
        if g_idx in chosen:
            continue
        for k in win_keys:
            if g_idx not in win_data[k]["frames"]:
                continue
            fd = win_data[k]["frames"][g_idx]
            anc = win_data[k]["anc"]
            ang = float(np.degrees(np.arccos(np.clip((np.trace(rots_u[anc].T @ rots_u[g_idx]) - 1.0) / 2.0, -1.0, 1.0))))
            sc = fd["ratio"] + 0.01 * float(np.nanmean(fd["cf_raw"])) - 0.005 * ang
            if g_idx not in chosen or sc > chosen[g_idx][0]:
                chosen[g_idx] = (sc, k)

    if len(chosen) < 3:
        return None

    target_pts = min(max(int(max_points), 350_000), 600_000)
    per_frame_quota = max(1, target_pts // max(1, len(chosen)))
    rng_sub = np.random.default_rng(42)
    pts_by_frame: dict[int, np.ndarray] = {}
    cf_by_frame: dict[int, np.ndarray] = {}
    col_by_frame: dict[int, np.ndarray] = {}

    for g_idx, (_, k) in sorted(chosen.items()):
        fd = win_data[k]["frames"][g_idx]
        anc = win_data[k]["anc"]
        fin_pf = np.isfinite(fd["pf"]).all(axis=-1) & np.isfinite(fd["cf_raw"])
        mask = fin_pf & np.isfinite(fd["cf"]) & (fd["cf"] > 1.05)
        if not np.any(mask):
            mask = fin_pf & (fd["cf_raw"] > 1.002)
        if not np.any(mask) and np.any(fin_pf):
            thr_fb = float(np.percentile(fd["cf_raw"][fin_pf], 50.0))
            mask = fin_pf & (fd["cf_raw"] >= thr_fb)
        if not np.any(mask):
            continue

        # Module 2.1: Recover keyframe's local camera coordinate grid using VIO-reconciled metric extrinsics
        R_w2c_metric = s_opt[k] * (rots_u[g_idx].T @ rots_u[anc])
        t_w2c_metric = rots_u[g_idx].T @ (t_opt[k] - t_vios[g_idx])
        ext_vio_3x4 = np.column_stack([R_w2c_metric, t_w2c_metric])
        P_cam_metric_grid = recover_keyframe_local_camera_grid(
            fd["P_grid"], fd["C_grid"], ext_3x4=ext_vio_3x4
        )
        p_cam_all = P_cam_metric_grid.reshape(-1, 3)
        p = (p_cam_all[mask] @ rots_u[g_idx].T) + t_vios[g_idx]

        # Module 2.2: Ray-aligned per-keyframe scale calibration anchored to status==5 dToF
        if g_idx in lidar_depths_by_idx or g_idx in lidar_s5_zones_by_idx:
            p_cam = p_cam_all[mask]
            z_c = p_cam[:, 2]
            cone = (
                (z_c > 0.15)
                & (np.abs(p_cam[:, 0]) < 0.35 * z_c)
                & (np.abs(p_cam[:, 1]) < 0.35 * z_c)
            )
            if int(np.count_nonzero(cone)) >= 50:
                z_cone = z_c[cone]
                z_med = float(np.median(z_cone))
                z_iqr = float(np.percentile(z_cone, 75.0) - np.percentile(z_cone, 25.0))
                if z_med > 0.18 and (z_iqr / z_med) < 0.28:
                    alpha_tof = _estimate_ray_aligned_tof_scale(
                        g_idx, P_cam_metric_grid, fd["C_grid"], fd["H_full"], fd["W_full"]
                    )
                    if alpha_tof is not None and np.isfinite(alpha_tof):
                        alpha_raw = float(np.clip(alpha_tof, 0.84, 1.24))
                    elif g_idx in lidar_depths_by_idx:
                        alpha_raw = float(np.clip(lidar_depths_by_idx[g_idx] / z_med, 0.84, 1.24))
                    else:
                        alpha_raw = 1.0
                    w_hom = float(np.clip(1.0 - (z_iqr / z_med) / 0.28, 0.35, 0.85))
                    alpha_g = 1.0 + w_hom * (alpha_raw - 1.0)
                    p = t_vios[g_idx] + alpha_g * (p - t_vios[g_idx])

        cf = np.maximum(fd["cf"][mask], 1.02)
        col = fd["col"][mask]
        if len(cf) > per_frame_quota:
            idx = np.sort(rng_sub.choice(len(cf), size=per_frame_quota, replace=False))
            p, cf, col = p[idx], cf[idx], col[idx]
        pts_by_frame[g_idx] = p
        cf_by_frame[g_idx] = cf
        col_by_frame[g_idx] = col

    if not pts_by_frame:
        return None

    # Module 2.3: Bounded 1-DoF optical-axis ICP refinement (delta_zeta_g in [-0.025m, +0.025m])
    dz_map = refine_optical_axis_1dof_icp(pts_by_frame, cf_by_frame, rots_u, t_vios)
    t_vios_refined = t_vios.copy()
    all_pts, all_cf, all_cols, all_fidx = [], [], [], []
    for g_idx in sorted(pts_by_frame.keys()):
        dz_g = dz_map.get(g_idx, 0.0)
        shift_vec = dz_g * rots_u[g_idx][:, 2]
        t_vios_refined[g_idx] = t_vios[g_idx] + shift_vec
        p_g = pts_by_frame[g_idx] + shift_vec[None, :]
        all_pts.append(p_g)
        all_cf.append(cf_by_frame[g_idx])
        all_cols.append(col_by_frame[g_idx])
        all_fidx.append(np.full(len(p_g), g_idx, dtype=np.int32))

    pts_vio = np.concatenate(all_pts, axis=0)
    cf_all = np.concatenate(all_cf, axis=0).astype(np.float32)
    cols_all = np.concatenate(all_cols, axis=0).astype(np.uint8)
    fidx_all = np.concatenate(all_fidx, axis=0).astype(np.int32)

    # Module 3.3: Global RGB exposure gain harmonization across overlapping 2cm voxels
    cols_all = harmonize_multi_window_rgb_exposure(pts_vio, cols_all, fidx_all)

    R0 = rots_u[0]
    t0 = t_vios[0]
    s_export = float(s_ref) if np.isfinite(s_ref) and s_ref > 1e-3 else 1.0
    pts_vggt = (((pts_vio - t0) @ R0) / s_export).astype(np.float32)
    ext_list = []
    for i in range(n_seq):
        R_w2c = rots_u[i].T @ R0
        center = ((t_vios[i] - t0) @ R0) / s_export
        t_w2c = -R_w2c @ center
        ext_list.append(np.column_stack([R_w2c, t_w2c]))
    ext_np = np.stack(ext_list, axis=0).astype(np.float32)
    return pts_vggt, cf_all, cols_all, fidx_all, ext_np, s_export


def run_vggt_inference(
    session_dir: str,
    force_recompute: bool = False,
    max_points: int = 750_000,
    model_id: str = "facebook/VGGT-1B",
) -> bool:
    """Run VGGT on session keyframes to produce cameras.json and project_point_cloud.npz."""
    vggt_dir = os.path.join(session_dir, "vggt")
    cam_path = os.path.join(vggt_dir, "cameras.json")
    npz_path = os.path.join(vggt_dir, "project_point_cloud.npz")
    man_path = os.path.join(vggt_dir, "keyframe_manifest.json")
    clean_ply_path = os.path.join(vggt_dir, "project_point_cloud_clean.ply")

    # Break symlink if vggt_dir was symlinked
    if os.path.islink(vggt_dir):
        logger.info("Unlinking symlinked vggt dir: %s", vggt_dir)
        os.unlink(vggt_dir)
        os.makedirs(vggt_dir, exist_ok=True)

    if not force_recompute and os.path.isfile(cam_path) and os.path.isfile(npz_path) and os.path.isfile(man_path):
        try:
            with open(man_path, "r", encoding="utf-8") as f:
                manifest_data = json.load(f)
            valid_npz = False
            try:
                with np.load(npz_path) as check_npz:
                    has_colors = "colors" in check_npz
                    has_conf = "confidence" in check_npz
                    pts_len = len(check_npz["points"]) if "points" in check_npz else 0
                    conf_arr = np.asarray(check_npz["confidence"], dtype=float) if has_conf else np.empty((0,), dtype=float)
                    max_conf = float(conf_arr.max()) if len(conf_arr) > 0 else 0.0
                    conf_ratio = float(np.mean(conf_arr > 1.05)) if len(conf_arr) > 0 else 0.0
                    valid_npz = has_colors and pts_len >= 50_000 and max_conf > 1.05 and conf_ratio >= 0.50
            except Exception:
                valid_npz = False

            if valid_npz and len(manifest_data.get("frames", [])) >= 3:
                vio_csv = os.path.join(session_dir, "processed_vio.csv")
                if os.path.isfile(vio_csv) and os.path.getsize(vio_csv) > 0:
                    try:
                        import pandas as pd
                        vdf = pd.read_csv(vio_csv)
                        if {"frame", "x", "y", "z", "qx", "qy", "qz", "qw"}.issubset(vdf.columns):
                            vmap = {int(r["frame"]): r for _, r in vdf.iterrows()}
                            updated_frames = False
                            for fr in manifest_data.get("frames", []):
                                sidx = fr.get("source_frame_index")
                                if sidx is not None and int(sidx) in vmap:
                                    vr = vmap[int(sidx)]
                                    for col in ("x", "y", "z", "qx", "qy", "qz", "qw"):
                                        fr[col] = float(vr[col])
                                    updated_frames = True
                            if updated_frames and manifest_data.get("pose_source") != "server_vio":
                                manifest_data["pose_source"] = "server_vio"
                                with open(man_path, "w", encoding="utf-8") as mf_out:
                                    json.dump(manifest_data, mf_out, indent=2)
                    except Exception as e:
                        logger.debug("Could not refresh keyframe_manifest poses from processed_vio.csv: %s", e)
                need_reclean = (
                    not os.path.isfile(clean_ply_path)
                    or os.path.getmtime(npz_path) > os.path.getmtime(clean_ply_path)
                    or (os.path.isfile(vio_csv) and os.path.getmtime(vio_csv) > os.path.getmtime(clean_ply_path))
                )
                if not need_reclean:
                    try:
                        import open3d as o3d
                        p_check = o3d.io.read_point_cloud(clean_ply_path)
                        if len(p_check.points) < 50_000:
                            need_reclean = True
                    except Exception:
                        need_reclean = True
                if need_reclean:
                    try:
                        import open3d as o3d
                        from app.services.manhattan_alignment import clean_and_align_point_cloud

                        with np.load(npz_path) as npz:
                            c_pts = np.asarray(npz["points"], dtype=float)
                            c_cols = np.asarray(npz["colors"], dtype=float) / 255.0 if "colors" in npz else None
                        pcd_cached = o3d.geometry.PointCloud()
                        pcd_cached.points = o3d.utility.Vector3dVector(c_pts)
                        if c_cols is not None:
                            pcd_cached.colors = o3d.utility.Vector3dVector(c_cols)
                        clean_cached, _ = clean_and_align_point_cloud(
                            pcd_cached, voxel=0.003, stat_neighbors=16, stat_std=2.8, R_init=None, session_dir=session_dir
                        )
                        o3d.io.write_point_cloud(clean_ply_path, clean_cached, write_ascii=False)
                        logger.info("Generated clean Manhattan point cloud from cached npz: %s (%d points)",
                                    clean_ply_path, len(clean_cached.points))
                    except Exception as e:
                        logger.warning("Could not create clean ply from existing npz: %s", e)
                logger.info("VGGT artifacts already exist in %s with pose_source '%s' and high fidelity (%d points); skipping inference.", vggt_dir, manifest_data.get("pose_source"), pts_len)
                return True
            else:
                logger.warning("VGGT artifacts need recomputing (frames: %d, valid_npz: %s); forcing recomputation.",
                               len(manifest_data.get("frames", [])), valid_npz)
                force_recompute = True
        except Exception:
            force_recompute = True

    # 1. Export keyframes if manifest does not exist or force_recompute
    if force_recompute or not os.path.isfile(man_path):
        logger.info("Exporting VGGT keyframes for session: %s", session_dir)
        export_res = export_vggt_keyframes(
            session_dir,
            subsample_method="spherical",
            prefer_server_vio=True,
            max_n=72,
        )
        if export_res.n_sent_to_vggt < 3:
            export_res = export_vggt_keyframes(session_dir, prefer_server_vio=True)
        if export_res.n_sent_to_vggt < 3:
            logger.warning("Too few keyframes selected (%d) for VGGT prior.", export_res.n_sent_to_vggt)
            return False

    with open(man_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    frames = manifest.get("frames", [])
    if len(frames) < 3:
        logger.warning("Manifest has fewer than 3 frames (%d).", len(frames))
        return False

    # 2. Check image paths
    img_dir = os.path.join(session_dir, "keyframes_selected", "images")
    img_paths = [os.path.join(img_dir, fr["vggt_filename"]) for fr in frames]
    valid_paths = [p for p in img_paths if os.path.isfile(p)]
    if len(valid_paths) < 3:
        # Fallback to rgb directory
        rgb_dir = os.path.join(session_dir, "rgb")
        img_paths = []
        for fr in frames:
            fn = fr.get("source_filename") or fr.get("vggt_filename")
            p = os.path.join(rgb_dir, fn)
            if not os.path.isfile(p) and "source_frame_index" in fr:
                p = os.path.join(rgb_dir, f"{int(fr['source_frame_index']):06d}.jpg")
            img_paths.append(p)
        valid_paths = [p for p in img_paths if os.path.isfile(p)]

    if len(valid_paths) < 3:
        logger.warning("Could not find at least 3 valid keyframe image files.")
        return False

    # 3. Load VGGT model onto GPU
    try:
        from vggt.models.vggt import VGGT
        from vggt.utils.load_fn import load_and_preprocess_images
        from vggt.utils.pose_enc import pose_encoding_to_extri_intri
    except ImportError:
        logger.exception("VGGT package not found in sys.path (%s).", sys.path)
        return False

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    logger.info("Initializing VGGT (%s) on %s (%s)...", model_id, device, dtype)
    t_start = time.time()
    if device == "cuda":
        torch.cuda.empty_cache()

    preloaded_model = None
    if "backend_api_extended_manhattan" in sys.modules:
        preloaded_model = getattr(sys.modules["backend_api_extended_manhattan"], "model", None)

    if preloaded_model is not None:
        logger.info("Reusing preloaded VGGT model from backend_api_extended_manhattan.")
        model = preloaded_model
        _patch_vggt_pos_embed(model)
        should_del_model = False
    else:
        model = VGGT.from_pretrained(model_id)
        # Disable depth head before GPU transfer to save VRAM and host RAM
        model.depth_head = None
        model = model.to(device, dtype=dtype)
        model.eval()
        _patch_vggt_pos_embed(model)
        should_del_model = True
        gc.collect()
        try:
            import ctypes
            ctypes.CDLL("libc.so.6").malloc_trim(0)
        except Exception:
            pass

    try:
        multi_win_res = None
        with _enable_feature_diet(model):
            logger.info("Preprocessing %d keyframes for VGGT...", len(img_paths))
            images, is_upright = _load_and_preprocess_vggt_images(
                img_paths, session_dir, load_and_preprocess_images
            )
            if images.dim() == 4:
                images = images.unsqueeze(0)

            t_infer = time.time()
            try:
                multi_win_res = _run_covisible_multi_window_vggt(
                    model, images, is_upright, frames, session_dir, max_points, device, dtype
                )
            except Exception as mw_err:
                logger.warning("Multi-window VGGT fallback to single-pass: %s", mw_err)
                multi_win_res = None

            if multi_win_res is not None:
                pts_valid, conf_valid, colors_valid, fidx_valid, ext_np, s_ref_mw = multi_win_res
                pose_enc = None
                inv_perm = list(range(len(frames)))
            else:
                # Select the highest-texture moderate-pitch keyframe as index 0 (VGGT world reference frame)
                # so a blurry or featureless close-up wall at frame 0 does not collapse cross-frame attention.
                n_seq = int(images.shape[1])
                perm = list(range(n_seq))
                inv_perm = list(range(n_seq))
                if n_seq >= 3:
                    try:
                        img_gray = images[0].float().mean(dim=1)  # (S, H, W)
                        dx = img_gray[:, :, 1:] - img_gray[:, :, :-1]
                        dy = img_gray[:, 1:, :] - img_gray[:, :-1, :]
                        sharp_scores = (dx.var(dim=(1, 2)) + dy.var(dim=(1, 2))).cpu().numpy()
                        anchor_scores = sharp_scores.copy()
                        rots_vio = []
                        from scipy.spatial.transform import Rotation as _Rot
                        for idx_f in range(n_seq):
                            fr_meta = frames[idx_f] if idx_f < len(frames) else {}
                            if all(k in fr_meta for k in ("qx", "qy", "qz", "qw")):
                                r_m = _Rot.from_quat([
                                    float(fr_meta["qx"]),
                                    float(fr_meta["qy"]),
                                    float(fr_meta["qz"]),
                                    float(fr_meta["qw"]),
                                ]).as_matrix()
                                rots_vio.append(r_m)
                                pitch_deg = float(np.degrees(np.arcsin(np.clip(-r_m[2, 2], -1.0, 1.0))))
                                if abs(pitch_deg) > 35.0:
                                    anchor_scores[idx_f] *= 0.25
                            else:
                                rots_vio.append(None)
                        best_anchor = int(np.argmax(anchor_scores))
                        if best_anchor != 0:
                            if len(rots_vio) == n_seq and rots_vio[best_anchor] is not None and all(r is not None for r in rots_vio):
                                r_anc = rots_vio[best_anchor]
                                rem = [i for i in range(n_seq) if i != best_anchor]
                                rem.sort(key=lambda i: -float(np.trace(r_anc.T @ rots_vio[i])))
                                perm = [best_anchor] + rem
                            else:
                                perm = [best_anchor] + [i for i in range(n_seq) if i != best_anchor]
                            inv_perm = [0] * n_seq
                            for new_i, orig_i in enumerate(perm):
                                inv_perm[orig_i] = new_i
                            perm_t = torch.tensor(perm, dtype=torch.long)
                            images = images[:, perm_t]
                    except Exception:
                        perm = list(range(n_seq))
                        inv_perm = list(range(n_seq))

                images = images.to(device, dtype=dtype)

                logger.info("Executing VGGT forward pass on GPU (Tensor shape: %s)...", tuple(images.shape))
                with torch.no_grad():
                    with torch.cuda.amp.autocast(enabled=(device == "cuda"), dtype=dtype):
                        aggregated_tokens_list, patch_start_idx = model.aggregator(images)

                        # Camera head
                        cam_tokens = [None] * (len(aggregated_tokens_list) - 1) + [aggregated_tokens_list[-1].to(device)]
                        pose_enc_list = model.camera_head(cam_tokens)
                        pose_enc = pose_enc_list[-1]

                        # Point head chunked processing (border-cropped [4:-4:2, 4:-4:2] striding)
                        B, S, _, H, W = images.shape
                        chunk_size = 4
                        per_frame_quota = max(1, int(max_points // max(1, S)))
                        all_pts = []
                        all_conf = []
                        all_colors = []
                        all_fidx = []
                        fallback_pts = []
                        fallback_conf = []
                        fallback_colors = []
                        fallback_fidx = []

                        for s_idx in range(0, S, chunk_size):
                            e_idx = min(s_idx + chunk_size, S)
                            chunk_tokens = [
                                aggregated_tokens_list[i][:, s_idx:e_idx].to(device)
                                if i in model.point_head.intermediate_layer_idx else None
                                for i in range(len(aggregated_tokens_list))
                            ]
                            img_chunk = images[:, s_idx:e_idx]
                            pts_c, conf_c = model.point_head._forward_impl(
                                chunk_tokens, img_chunk, patch_start_idx, 0, e_idx - s_idx
                            )
                            pts_c = pts_c.float()
                            conf_c = conf_c.float()

                            if H > 16 and W > 16:
                                pts_sub_4d = pts_c[:, :, 4:-4:2, 4:-4:2, :]
                                conf_sub_3d = conf_c[:, :, 4:-4:2, 4:-4:2]
                                img_chunk_perm = img_chunk.permute(0, 1, 3, 4, 2)[:, :, 4:-4:2, 4:-4:2, :]
                            else:
                                pts_sub_4d = pts_c[:, :, ::2, ::2, :]
                                conf_sub_3d = conf_c[:, :, ::2, ::2]
                                img_chunk_perm = img_chunk.permute(0, 1, 3, 4, 2)[:, :, ::2, ::2, :]

                            n_in_chunk = e_idx - s_idx
                            for local_f in range(n_in_chunk):
                                fid = int(perm[s_idx + local_f])
                                pts_f = pts_sub_4d[0, local_f].reshape(-1, 3).cpu()
                                conf_f = conf_sub_3d[0, local_f].reshape(-1).cpu()
                                rgb_f = img_chunk_perm[0, local_f].float().clamp(0.0, 1.0).reshape(-1, 3).cpu()
                                col_f = (rgb_f * 255.0).round().to(torch.uint8)
                                finite_f = torch.isfinite(pts_f).all(dim=-1) & torch.isfinite(conf_f)
                                mask_f = finite_f & (conf_f > 1.05)
                                if mask_f.any():
                                    pts_k = pts_f[mask_f]
                                    conf_k = conf_f[mask_f]
                                    col_k = col_f[mask_f]
                                    if len(conf_k) > per_frame_quota:
                                        top_k_idx = torch.linspace(
                                            0, len(conf_k) - 1, steps=per_frame_quota
                                        ).round().long()
                                        pts_k = pts_k[top_k_idx]
                                        conf_k = conf_k[top_k_idx]
                                        col_k = col_k[top_k_idx]
                                    all_pts.append(pts_k)
                                    all_conf.append(conf_k)
                                    all_colors.append(col_k)
                                    all_fidx.append(torch.full((len(pts_k),), fid, dtype=torch.int32))
                                elif (finite_f & (conf_f >= 1.0)).any():
                                    mask_fb = finite_f & (conf_f >= 1.0)
                                    pts_fb = pts_f[mask_fb][: min(256, int(mask_fb.sum()))]
                                    conf_fb = conf_f[mask_fb][: len(pts_fb)]
                                    col_fb = col_f[mask_fb][: len(pts_fb)]
                                    fallback_pts.append(pts_fb)
                                    fallback_conf.append(conf_fb)
                                    fallback_colors.append(col_fb)
                                    fallback_fidx.append(torch.full((len(pts_fb),), fid, dtype=torch.int32))

                        if device == "cuda":
                            torch.cuda.empty_cache()

                        total_conf_pts = sum(len(x) for x in all_pts)
                        if total_conf_pts == 0 and fallback_pts:
                            all_pts = fallback_pts
                            all_conf = fallback_conf
                            all_colors = fallback_colors
                            all_fidx = fallback_fidx

                        if all_pts:
                            pts_valid = torch.cat(all_pts, dim=0).float().numpy()
                            conf_valid = torch.cat(all_conf, dim=0).float().numpy()
                            colors_valid = torch.cat(all_colors, dim=0).numpy()
                            fidx_valid = torch.cat(all_fidx, dim=0).numpy()
                        else:
                            pts_valid = np.empty((0, 3), dtype=np.float32)
                            conf_valid = np.empty((0,), dtype=np.float32)
                            colors_valid = np.empty((0, 3), dtype=np.uint8)
                            fidx_valid = np.empty((0,), dtype=np.int32)

        dur_infer = time.time() - t_infer
        logger.info("VGGT forward pass completed in %.2fs. Peak VRAM: %.1f MB",
                    dur_infer, (torch.cuda.max_memory_allocated(0) / 1024**2) if device == "cuda" else 0.0)

        # 4. Decode camera extrinsics & write cameras.json
        if multi_win_res is None:
            extrinsic, intrinsic = pose_encoding_to_extri_intri(pose_enc, images.shape[-2:])
            if extrinsic.dim() == 4:
                extrinsic = extrinsic.squeeze(0)
            ext_np = extrinsic.detach().cpu().float().numpy()
            if len(ext_np) == len(inv_perm):
                ext_np = ext_np[inv_perm]

        # Release GPU VRAM immediately before CPU-bound Open3D & NPZ steps
        if "model" in locals() and model is not None:
            if should_del_model:
                model = None
        images = None
        aggregated_tokens_list = None
        cam_tokens = None
        pose_enc_list = None
        pose_enc = None
        extrinsic = None
        intrinsic = None
        chunk_tokens = None
        img_chunk = None
        pts_c = None
        conf_c = None
        img_chunk_perm = None
        all_pts = None
        all_conf = None
        all_colors = None
        all_fidx = None
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()

        cams = []
        for i, fr in enumerate(frames):
            if i >= len(ext_np):
                break
            ext = ext_np[i]  # 3x4
            R = ext[:, :3]
            t = ext[:, 3]
            center = -R.T @ t
            cams.append({
                "vggt_filename": fr["vggt_filename"],
                "extrinsic_3x4": ext.tolist(),
                "center": center.tolist(),
            })

        cameras_doc = {
            "model_id": model_id,
            "convention": "opencv_w2c",
            "image_orientation": "upright" if is_upright else "sensor",
            "cameras": cams,
        }
        os.makedirs(vggt_dir, exist_ok=True)
        with open(cam_path, "w", encoding="utf-8") as f:
            json.dump(cameras_doc, f, indent=2)
        logger.info("Saved %d camera poses to %s", len(cams), cam_path)

        # 5. Extract points & write project_point_cloud.npz, project_point_cloud.ply, project_point_cloud_clean.ply
        mask = np.isfinite(pts_valid).all(axis=-1) & np.isfinite(conf_valid) & (conf_valid >= 1.0)
        pts_valid = pts_valid[mask]
        conf_valid = conf_valid[mask]
        colors_valid = colors_valid[mask]
        fidx_valid = fidx_valid[mask]

        if len(pts_valid) > max_points:
            top_idx = np.linspace(0, len(pts_valid) - 1, max_points, dtype=np.int64)
            pts_valid = pts_valid[top_idx]
            conf_valid = conf_valid[top_idx]
            colors_valid = colors_valid[top_idx]
            fidx_valid = fidx_valid[top_idx]

        # Save NPZ before clean_and_align_point_cloud so estimate_vggt_gravity_up can refine against 3D planes
        np.savez_compressed(
            npz_path,
            points=pts_valid,
            confidence=conf_valid,
            colors=colors_valid,
            frame_indices=fidx_valid,
            image_names=np.array([fr["vggt_filename"] for fr in frames[:len(ext_np)]]),
        )
        logger.info("Saved %d 3D points with RGB colors to %s", len(pts_valid), npz_path)

        import open3d as o3d
        from app.services.manhattan_alignment import clean_and_align_point_cloud

        raw_pcd = o3d.geometry.PointCloud()
        if len(pts_valid) > 0:
            raw_pcd.points = o3d.utility.Vector3dVector(pts_valid)
            raw_pcd.colors = o3d.utility.Vector3dVector(colors_valid.astype(float) / 255.0)

        raw_ply_path = os.path.join(vggt_dir, "project_point_cloud.ply")
        o3d.io.write_point_cloud(raw_ply_path, raw_pcd, write_ascii=False)

        clean_pcd, clean_info = clean_and_align_point_cloud(
            raw_pcd, voxel=0.003, stat_neighbors=16, stat_std=2.8, R_init=None, session_dir=session_dir
        )
        o3d.io.write_point_cloud(clean_ply_path, clean_pcd, write_ascii=False)
        logger.info(
            "Cleaned & Manhattan-aligned VGGT point cloud: %d -> %d points (saved to %s)",
            len(pts_valid), len(clean_pcd.points), clean_ply_path,
        )

        total_dur = time.time() - t_start
        logger.info("VGGT processing complete for session in %.2fs!", total_dur)
        return True

    except Exception:
        logger.exception("VGGT inference failed for session %s", session_dir)
        return False
    finally:
        # Guarantee full cleanup of model from GPU VRAM
        if "model" in locals() and model is not None:
            if should_del_model:
                model = None
        images = None
        aggregated_tokens_list = None
        all_pts = None
        all_conf = None
        all_colors = None
        gc.collect()
        if "device" in locals() and device == "cuda":
            torch.cuda.empty_cache()
