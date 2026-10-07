import json
import os
from typing import Optional

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import uuid
import traceback
import runpod
import torch
from vggt.models.vggt import VGGT

# Define global constants matching backend
MODEL_ID = os.getenv("VGGT_MODEL_ID", "facebook/VGGT-1B")
BASE_DIR = os.getenv("BASE_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "vggt_room3d_jobs"))
DEFAULT_HARD_MAX_POINTS = 300_000_000

# 1. Khởi tạo & Warm-up Model (Global Scope)
print("--> Loading VGGT Model to GPU...", flush=True)
device = "cuda" if torch.cuda.is_available() else "cpu"
if device == "cuda":
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
else:
    dtype = torch.float32

# Load and compile weights to GPU VRAM
model = VGGT.from_pretrained(MODEL_ID).to(device)
model.eval()

# Inject the model into the backend_api module to prevent reloading
import backend_api_extended_manhattan
backend_api_extended_manhattan.model = model
backend_api_extended_manhattan.device = device
backend_api_extended_manhattan.dtype = dtype

print("--> Model loaded successfully!", flush=True)

def find_multisensor_session_dir(root_dir: str) -> Optional[str]:
    """Detect if directory contains multi-sensor LiDAR/odometry scan files."""
    indicators = ("manifest.json", "camera_matrix.csv", "odometry.csv", "lidar.csv")
    for dirpath, _, filenames in os.walk(root_dir):
        if any(f in filenames for f in indicators):
            return dirpath
    return None


def handler(job):
    """
    RunPod Serverless Handler.
    Receives JSON input from the RunPod Queue and processes it.
    Supports both:
    1. Multi-Sensor LiDAR Scan Packages (manifest.json, camera_matrix.csv, odometry.csv, lidar.csv)
    2. Pure Photo Packages (VGGT-1B 3D Point Cloud Reconstruction)
    """
    job_input = job.get("input", {})
    zip_url = job_input.get("zip_url")
    if not zip_url:
        return {"status": "error", "error": "Missing 'zip_url' in input"}
        
    batch_id = job_input.get("batch_id", "batch")
    metadata = job_input.get("metadata", [])
    conf_threshold = float(job_input.get("conf_threshold", 1.0))
    max_points = int(job_input.get("max_points", 3000000))
    hard_max_points = int(job_input.get("hard_max_points", DEFAULT_HARD_MAX_POINTS))
    max_images = job_input.get("max_images")
    if max_images is not None:
        max_images = int(max_images)
    clean_voxel = float(job_input.get("clean_voxel", backend_api_extended_manhattan.DEFAULT_CLEAN_VOXEL))
    clean_stat_neighbors = int(job_input.get("clean_stat_neighbors", backend_api_extended_manhattan.DEFAULT_CLEAN_STAT_NEIGHBORS))
    clean_stat_std = float(job_input.get("clean_stat_std", backend_api_extended_manhattan.DEFAULT_CLEAN_STAT_STD))

    # Generate job IDs and paths
    job_id = str(uuid.uuid4())[:8]
    job_dir = os.path.join(BASE_DIR, job_id)
    img_dir = os.path.join(job_dir, "images")
    out_dir = os.path.join(job_dir, "output")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)

    try:
        # Download and extract the zip file
        image_paths = backend_api_extended_manhattan.download_and_extract_zip(zip_url, job_dir, img_dir)
        object_prefix = f"{backend_api_extended_manhattan.safe_object_part(batch_id)}/{job_id}"

        # ── Branch A: Multi-Sensor LiDAR Session ──
        session_dir = find_multisensor_session_dir(img_dir) or find_multisensor_session_dir(job_dir)
        if session_dir is not None:
            print(f"--> [Job {job_id}] Detected Multi-Sensor LiDAR Session in: {session_dir}. Running standalone pipeline...", flush=True)
            from app.pipeline.runner import run_pipeline
            from app.services.cad.contract import layout_to_metrics_mm

            result = run_pipeline(session_dir)
            if not result.success:
                raise RuntimeError(f"Multi-sensor pipeline execution failed: {result.error_message}")

            # 1. Clean PLY upload
            clean_ply_candidates = [
                os.path.join(session_dir, "reconstructed_visual.ply"),
                os.path.join(session_dir, "whiteflat.ply"),
                os.path.join(session_dir, "reconstructed.ply"),
            ]
            clean_ply_path = next((p for p in clean_ply_candidates if os.path.isfile(p)), None)
            clean_ply_r2 = {}
            if clean_ply_path:
                clean_ply_r2 = backend_api_extended_manhattan.upload_to_r2(
                    clean_ply_path, f"ply_clean/{object_prefix}/project_point_cloud_clean.ply"
                )

            # 2. Floorplan & CAD files upload
            floorplan_files = {}
            file_candidates = [
                ("floorplan_png", ["FloorPlan_A3.png", "FloorPlan_A4.png", "floorplan.png", "ISO_A3_Floorplan.png", "ISO_A4_Floorplan.png"]),
                ("floorplan_pdf", ["FloorPlan_A3.pdf", "FloorPlan_A4.pdf", "floorplan.pdf", "ISO_A3_Floorplan.pdf", "ISO_A4_Floorplan.pdf"]),
                ("wall_elevation_png", ["Walls.png", "walls.png"]),
                ("debug_topdown_png", ["debug_topdown.png"]),
                ("room_model_glb", ["room_model_texture.glb", "room_model.glb", "reconstructed.glb"]),
                ("metrics_json", ["metrics.json"]),
            ]
            for file_key, cands in file_candidates:
                for c in cands:
                    p = os.path.join(session_dir, c)
                    if os.path.isfile(p):
                        target_name = "walls.png" if "wall" in file_key else c
                        floorplan_files[file_key] = backend_api_extended_manhattan.upload_to_r2(
                            p, f"ply_clean/{object_prefix}/{target_name}"
                        )
                        break

            # 3. Room metrics extraction (contract matching new_update.md Section 5)
            room_metrics = None
            metrics_json_path = os.path.join(session_dir, "metrics.json")
            if os.path.isfile(metrics_json_path):
                try:
                    with open(metrics_json_path, "r", encoding="utf-8") as f:
                        room_metrics = json.load(f)
                except Exception:
                    room_metrics = None
            if room_metrics is None and result.floorplan:
                room_metrics = layout_to_metrics_mm(result.floorplan)

            presigned_url = clean_ply_r2.get("presigned_url") or floorplan_files.get("floorplan_png", {}).get("presigned_url")
            r2_key = clean_ply_r2.get("key")

            return {
                "status": "success",
                "job_id": job_id,
                "mode": "multisensor_lidar",
                "download_url": presigned_url,
                "clean_ply": {
                    "presigned_url": presigned_url,
                    "r2_key": r2_key,
                },
                "room_metrics": room_metrics,
                "floorplan_files": floorplan_files,
                "timings": result.metrics.timings if hasattr(result.metrics, "timings") else {},
            }

        # ── Branch B: Pure Photo Reconstruction (VGGT-1B) ──
        if not image_paths:
            return {"status": "error", "error": "No valid images or sensor data found in the zip file"}

        print(f"--> [Job {job_id}] Running VGGT-1B inference on {len(image_paths)} images...", flush=True)
        meta = backend_api_extended_manhattan.run_inference_pipeline(
            image_paths=image_paths,
            batch_id=batch_id,
            metadata=metadata,
            conf_threshold=conf_threshold,
            max_points=max_points,
            hard_max_points=hard_max_points,
            max_images=max_images,
            clean_voxel=clean_voxel,
            clean_stat_neighbors=clean_stat_neighbors,
            clean_stat_std=clean_stat_std,
            job_id=job_id,
            job_dir=job_dir,
            img_dir=img_dir,
            out_dir=out_dir
        )

        presigned_url = meta["clean_ply"]["presigned_url"]
        r2_key = meta["clean_ply"]["key"]

        return {
            "status": "success",
            "job_id": job_id,
            "mode": "vggt_photo",
            "download_url": presigned_url,
            "clean_ply": {
                "presigned_url": presigned_url,
                "r2_key": r2_key
            },
            "room_metrics": meta.get("room_metrics"),
            "floorplan_files": meta.get("floorplan_files", {}),
            "alignment": meta.get("alignment")
        }

    except Exception as e:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        error_text = traceback.format_exc()
        with open(os.path.join(out_dir, "error.txt"), "w", encoding="utf-8") as f:
            f.write(error_text)
        try:
            backend_api_extended_manhattan.zip_job(job_dir, job_id)
        except Exception:
            pass
        return {"status": "error", "job_id": job_id, "error": str(e)}

if __name__ == "__main__":
    print("Starting RunPod Serverless Worker...", flush=True)
    runpod.serverless.start({"handler": handler})
