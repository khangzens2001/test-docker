import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import uuid
import traceback
import runpod
import torch
from vggt.models.vggt import VGGT

# Define global constants matching backend
MODEL_ID = "facebook/VGGT-1B"
BASE_DIR = os.getenv("BASE_DIR", "/app/vggt_room3d_jobs")
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

def handler(job):
    """
    RunPod Serverless Handler.
    Receives JSON input from the RunPod Queue and processes it.
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
        if not image_paths:
            return {"status": "error", "error": "No valid images found in the zip file"}

        # Run the backend inference and R2 upload pipeline
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
            "download_url": presigned_url,
            "clean_ply": {
                "presigned_url": presigned_url,
                "r2_key": r2_key
            }
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
