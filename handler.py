import json
import os
from typing import Optional

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import gc
import shutil
import uuid
import traceback

if __name__ != "__mp_main__":
    import runpod
    import torch
    import backend_api_extended_manhattan

# Define global constants matching backend
MODEL_ID = os.getenv("VGGT_MODEL_ID", "facebook/VGGT-1B")
BASE_DIR = os.getenv("BASE_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "vggt_room3d_jobs"))
DEFAULT_HARD_MAX_POINTS = 300_000_000

class PipelineFailure(RuntimeError):
    def __init__(self, message, error_type=None, retryable=None):
        super().__init__(f"Multi-sensor pipeline execution failed: {message}")
        self.error_type = error_type
        self.retryable = retryable


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
    # Caller-supplied job_id (e.g. Temporal workflow id) makes retries idempotent.
    requested_job_id = job_input.get("job_id")
    if requested_job_id:
        job_id = backend_api_extended_manhattan.safe_object_part(str(requested_job_id))[:64]
    else:
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
            if getattr(backend_api_extended_manhattan, "model", None) is not None:
                backend_api_extended_manhattan.model = None
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            from app.pipeline.runner import run_pipeline
            from app.services.cad.contract import layout_to_metrics_mm

            def _on_stage(idx: int, name: str, pct: int) -> None:
                try:
                    runpod.serverless.progress_update(
                        job, json.dumps({"stage": name, "stage_index": idx, "pct": pct})
                    )
                except Exception:
                    pass

            mode = job_input.get("mode", "legacy_full")
            result = run_pipeline(
                session_dir,
                stage_callback=_on_stage,
                skip_materials=(mode == "reconstruct_only"),
                enable_parallel=job_input.get("enable_parallel"),
            )
            if not result.success:
                raise PipelineFailure(result.error_message, result.error_type, result.retryable)

            from app.services.artifact_upload import upload_session_artifacts

            clean_ply_r2, floorplan_files = upload_session_artifacts(
                session_dir, object_prefix, backend_api_extended_manhattan.upload_to_r2
            )

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
                "pipeline_mode": mode,
                "session_prefix": object_prefix,
                "download_url": presigned_url,
                "clean_ply": {
                    "presigned_url": presigned_url,
                    "r2_key": r2_key,
                },
                "room_metrics": room_metrics,
                "floorplan_files": floorplan_files,
                "timings": result.metrics.timings if hasattr(result.metrics, "timings") else {},
                "warnings": result.warnings,
            }

        # ── Branch B: Pure Photo Reconstruction (VGGT-1B) ──
        if not image_paths:
            return {"status": "error", "job_id": job_id, "error": "No valid images or sensor data found in the zip file", "error_type": "invalid_input", "retryable": False}

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
            out_dir=out_dir,
            create_archive=False,
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
        from app.core.exceptions import classify_pipeline_error

        err_type = getattr(e, "error_type", None)
        retryable = getattr(e, "retryable", None)
        if err_type is None or retryable is None:
            err_type, retryable = classify_pipeline_error(e)
        return {
            "status": "error",
            "job_id": job_id,
            # RunPod only surfaces the `error` string of a failed job, so append the traceback tail.
            "error": f"{e}\n--- traceback (tail) ---\n{error_text[-1800:]}",
            "error_type": err_type,
            "retryable": retryable,
        }
    finally:
        r2_configured = bool(
            getattr(backend_api_extended_manhattan, "R2_ENDPOINT_URL", "")
            and os.getenv("R2_ACCESS_KEY_ID")
            and os.getenv("R2_SECRET_ACCESS_KEY")
        )
        should_cleanup = os.getenv("CLEANUP_JOB_DIR", "1" if r2_configured else "0").strip() != "0"
        if job_input.get("keep_session"):
            should_cleanup = False
        if should_cleanup and os.path.isdir(job_dir):
            shutil.rmtree(job_dir, ignore_errors=True)

if __name__ == "__main__":
    try:
        from app.services.vio_core.numba_accelerated import warmup_numba_jit
        warmup_numba_jit()
    except Exception as e:
        print(f"Warning: Numba JIT warmup skipped: {e}", flush=True)
    print("Starting RunPod Serverless Worker...", flush=True)
    runpod.serverless.start({"handler": handler})
