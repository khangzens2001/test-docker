"""VGGT Local GPU Inference Service.

Extracts keyframes from mobile session and runs facebook/VGGT-1B locally on GPU
in FP16 (optimized for NVIDIA RTX 2060 12GB VRAM). Produces vggt/cameras.json
and vggt/project_point_cloud.npz for downstream Manhattan topology prior.
"""
from __future__ import annotations

import gc
import json
import logging
import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import sys
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
                    max_conf = float(check_npz["confidence"].max()) if has_conf and len(check_npz["confidence"]) > 0 else 0.0
                    valid_npz = has_colors and pts_len >= 100_000 and max_conf >= 1.0
            except Exception:
                valid_npz = False

            if valid_npz and len(manifest_data.get("frames", [])) >= 3:
                vio_csv = os.path.join(session_dir, "processed_vio.csv")
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

    # Patch aggregator to offload intermediate tokens to CPU (saves ~4.3GB VRAM)
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
        if model.aggregator.rope is not None:
            p = model.aggregator.position_getter(
                B * S, H // model.aggregator.patch_size, W // model.aggregator.patch_size, device=imgs.device
            )

        if model.aggregator.patch_start_idx > 0:
            p = p + 1
            p_special = torch.zeros(B * S, model.aggregator.patch_start_idx, 2, device=imgs.device, dtype=p.dtype)
            p = torch.cat([p_special, p], dim=1)

        _, P, C = tkns.shape
        f_idx = 0
        g_idx = 0
        out_list = []
        needed_layers = set(getattr(model.point_head, "intermediate_layer_idx", [4, 11, 17, 23])) | {
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

        return out_list, model.aggregator.patch_start_idx

    orig_aggregator_forward = getattr(model.aggregator, "forward", None)
    model.aggregator.forward = _optimized_aggregator_forward

    try:
        logger.info("Preprocessing %d keyframes for VGGT...", len(img_paths))
        images, is_upright = _load_and_preprocess_vggt_images(
            img_paths, session_dir, load_and_preprocess_images
        )
        images = images.to(device, dtype=dtype)
        if images.dim() == 4:
            images = images.unsqueeze(0)

        logger.info("Executing VGGT forward pass on GPU (Tensor shape: %s)...", tuple(images.shape))
        t_infer = time.time()
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
                        fid = s_idx + local_f
                        pts_f = pts_sub_4d[0, local_f].reshape(-1, 3).cpu()
                        conf_f = conf_sub_3d[0, local_f].reshape(-1).cpu()
                        rgb_f = img_chunk_perm[0, local_f].float().clamp(0.0, 1.0).reshape(-1, 3).cpu()
                        col_f = (rgb_f * 255.0).round().to(torch.uint8)
                        mask_f = (
                            torch.isfinite(pts_f).all(dim=-1)
                            & torch.isfinite(conf_f)
                            & (conf_f >= 1.0)
                        )
                        if mask_f.any():
                            pts_f = pts_f[mask_f]
                            conf_f = conf_f[mask_f]
                            rgb_f = rgb_f[mask_f]
                            col_f = col_f[mask_f]
                            if len(conf_f) > per_frame_quota:
                                sat_f = rgb_f.max(dim=-1).values - rgb_f.min(dim=-1).values
                                lum_f = 0.299 * rgb_f[:, 0] + 0.587 * rgb_f[:, 1] + 0.114 * rgb_f[:, 2]
                                rg_f = rgb_f[:, 0] - rgb_f[:, 1]
                                gr_f = rgb_f[:, 1] - rgb_f[:, 0]
                                # Boost salient colored/fixture pixels so smooth objects (basins, doors, mirrors, shelves)
                                # are not discarded in favor of high-frequency tile grout lines
                                salience = (
                                    (sat_f >= 0.08).float() * 2.0
                                    + (rg_f >= 0.06).float() * 2.5
                                    + (gr_f >= 0.04).float() * 2.5
                                    + ((lum_f <= 0.32) | (lum_f >= 0.88)).float() * 1.0
                                )
                                # Also give a bonus to every 2nd pixel so background wall/floor remains spatially uniform
                                uniform_grid = torch.zeros_like(conf_f)
                                uniform_grid[::2] = 1.5
                                score_f = conf_f + salience + uniform_grid
                                top_k_idx = torch.topk(
                                    score_f, k=per_frame_quota, largest=True, sorted=False
                                ).indices
                                top_k_idx, _ = torch.sort(top_k_idx)
                                pts_f = pts_f[top_k_idx]
                                conf_f = conf_f[top_k_idx]
                                col_f = col_f[top_k_idx]
                            all_pts.append(pts_f)
                            all_conf.append(conf_f)
                            all_colors.append(col_f)
                            all_fidx.append(torch.full((len(pts_f),), fid, dtype=torch.int32))

                if device == "cuda":
                    torch.cuda.empty_cache()

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
        extrinsic, intrinsic = pose_encoding_to_extri_intri(pose_enc, images.shape[-2:])
        if extrinsic.dim() == 4:
            extrinsic = extrinsic.squeeze(0)
        ext_np = extrinsic.detach().cpu().float().numpy()

        # Release GPU VRAM immediately before CPU-bound Open3D & NPZ steps
        if "model" in locals() and model is not None:
            if "orig_aggregator_forward" in locals() and orig_aggregator_forward is not None:
                model.aggregator.forward = orig_aggregator_forward
            if should_del_model:
                model = None
        orig_aggregator_forward = None
        _optimized_aggregator_forward = None
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
            top_idx = np.sort(np.argsort(conf_valid)[-max_points:])
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
            if "orig_aggregator_forward" in locals() and orig_aggregator_forward is not None:
                model.aggregator.forward = orig_aggregator_forward
            if should_del_model:
                model = None
        orig_aggregator_forward = None
        _optimized_aggregator_forward = None
        images = None
        aggregated_tokens_list = None
        all_pts = None
        all_conf = None
        all_colors = None
        gc.collect()
        if "device" in locals() and device == "cuda":
            torch.cuda.empty_cache()
