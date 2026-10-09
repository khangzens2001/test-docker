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


def run_vggt_inference(
    session_dir: str,
    force_recompute: bool = False,
    max_points: int = 500_000,
    model_id: str = "facebook/VGGT-1B",
) -> bool:
    """Run VGGT on session keyframes to produce cameras.json and project_point_cloud.npz."""
    vggt_dir = os.path.join(session_dir, "vggt")
    cam_path = os.path.join(vggt_dir, "cameras.json")
    npz_path = os.path.join(vggt_dir, "project_point_cloud.npz")
    man_path = os.path.join(vggt_dir, "keyframe_manifest.json")

    # Break symlink if vggt_dir was symlinked
    if os.path.islink(vggt_dir):
        logger.info("Unlinking symlinked vggt dir: %s", vggt_dir)
        os.unlink(vggt_dir)
        os.makedirs(vggt_dir, exist_ok=True)

    from app.pipeline.runner import load_pose_table_for_tsdf
    current_pose_df = load_pose_table_for_tsdf(session_dir)
    current_pose_source = current_pose_df.attrs.get("pose_source") if hasattr(current_pose_df, "attrs") else None

    if not force_recompute and os.path.isfile(cam_path) and os.path.isfile(npz_path) and os.path.isfile(man_path):
        try:
            with open(man_path, "r", encoding="utf-8") as f:
                manifest_data = json.load(f)
            has_colors = False
            try:
                with np.load(npz_path) as check_npz:
                    has_colors = "colors" in check_npz
            except Exception:
                has_colors = False
            if manifest_data.get("pose_source") == current_pose_source and has_colors:
                logger.info("VGGT artifacts already exist in %s with matching pose_source '%s' and colors; skipping inference.", vggt_dir, current_pose_source)
                return True
            else:
                logger.warning("VGGT artifacts need recomputing (pose_source match: %s, has_colors: %s); forcing recomputation.",
                               manifest_data.get("pose_source") == current_pose_source, has_colors)
                force_recompute = True
        except Exception:
            force_recompute = True

    # 1. Export keyframes if manifest does not exist or force_recompute
    if force_recompute or not os.path.isfile(man_path):
        logger.info("Exporting VGGT keyframes for session: %s", session_dir)
        export_res = export_vggt_keyframes(session_dir)
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
        model = VGGT.from_pretrained(model_id).to(device, dtype=dtype)
        model.eval()
        model.depth_head = None
        _patch_vggt_pos_embed(model)
        should_del_model = True

    try:
        logger.info("Preprocessing %d keyframes for VGGT...", len(img_paths))
        images = load_and_preprocess_images(img_paths).to(device, dtype=dtype)
        if images.dim() == 4:
            images = images.unsqueeze(0)

        logger.info("Executing VGGT forward pass on GPU (Tensor shape: %s)...", tuple(images.shape))
        t_infer = time.time()
        with torch.no_grad():
            aggregated_tokens_list, patch_start_idx = model.aggregator(images)
            
            # Camera head
            pose_enc_list = model.camera_head(aggregated_tokens_list)
            pose_enc = pose_enc_list[-1]
            
            # Point head chunked processing (chunk size = 4 frames for fast parallel throughput)
            B, S, _, H, W = images.shape
            chunk_size = 4
            all_pts = []
            all_conf = []
            all_colors = []
            for s_idx in range(0, S, chunk_size):
                e_idx = min(s_idx + chunk_size, S)
                pts_c, conf_c = model.point_head._forward_impl(
                    aggregated_tokens_list, images, patch_start_idx, s_idx, e_idx
                )
                # Spatial striding (::3, ::3) transfers ~30k points per frame (~750k total) efficiently to CPU
                pts_sub = pts_c[:, :, ::3, ::3, :].reshape(-1, 3).cpu()
                conf_sub = conf_c[:, :, ::3, ::3].reshape(-1).cpu()
                all_pts.append(pts_sub)
                all_conf.append(conf_sub)

                # Extract RGB colors corresponding to spatial striding (::3, ::3)
                img_chunk = images[:, s_idx:e_idx].permute(0, 1, 3, 4, 2)
                col_sub = (img_chunk[:, :, ::3, ::3, :].clamp(0.0, 1.0) * 255.0).round().to(torch.uint8).reshape(-1, 3).cpu()
                all_colors.append(col_sub)

            if device == "cuda":
                torch.cuda.empty_cache()

            pts_valid = torch.cat(all_pts, dim=0).float().numpy()
            conf_valid = torch.cat(all_conf, dim=0).float().numpy()
            colors_valid = torch.cat(all_colors, dim=0).numpy()

        dur_infer = time.time() - t_infer
        logger.info("VGGT forward pass completed in %.2fs. Peak VRAM: %.1f MB",
                    dur_infer, (torch.cuda.max_memory_allocated(0) / 1024**2) if device == "cuda" else 0.0)

        # 4. Decode camera extrinsics & write cameras.json
        extrinsic, intrinsic = pose_encoding_to_extri_intri(pose_enc, images.shape[-2:])
        if extrinsic.dim() == 4:
            extrinsic = extrinsic.squeeze(0)
        ext_np = extrinsic.detach().cpu().float().numpy()

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
            "cameras": cams,
        }
        os.makedirs(vggt_dir, exist_ok=True)
        with open(cam_path, "w", encoding="utf-8") as f:
            json.dump(cameras_doc, f, indent=2)
        logger.info("Saved %d camera poses to %s", len(cams), cam_path)

        # 5. Extract points & write project_point_cloud.npz
        mask = np.isfinite(pts_valid).all(axis=-1) & np.isfinite(conf_valid)
        pts_valid = pts_valid[mask]
        conf_valid = conf_valid[mask]
        colors_valid = colors_valid[mask]

        if len(pts_valid) > max_points:
            stride = int(np.ceil(len(pts_valid) / max_points))
            pts_valid = pts_valid[::stride]
            conf_valid = conf_valid[::stride]
            colors_valid = colors_valid[::stride]

        np.savez_compressed(
            npz_path,
            points=pts_valid,
            confidence=conf_valid,
            colors=colors_valid,
            image_names=np.array([fr["vggt_filename"] for fr in frames[:len(ext_np)]]),
        )
        logger.info("Saved %d 3D points with RGB colors to %s", len(pts_valid), npz_path)

        total_dur = time.time() - t_start
        logger.info("VGGT processing complete for session in %.2fs!", total_dur)
        return True

    except Exception:
        logger.exception("VGGT inference failed for session %s", session_dir)
        return False
    finally:
        # Guarantee full cleanup of model from GPU VRAM
        if should_del_model and "model" in locals():
            del model
        if "images" in locals():
            del images
        if "aggregated_tokens_list" in locals():
            del aggregated_tokens_list
        if "all_pts" in locals():
            del all_pts
        if "all_conf" in locals():
            del all_conf
        if "all_colors" in locals():
            del all_colors
        if "device" in locals() and device == "cuda":
            torch.cuda.empty_cache()
        gc.collect()
