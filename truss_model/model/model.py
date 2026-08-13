import os
import sys
import uuid
import traceback
import subprocess
import torch

class Model:
    def __init__(self, **kwargs):
        self._data_dir = kwargs.get("data_dir")
        self._model = None
        self._secrets = kwargs.get("secrets", {})

    def load(self):
        # 1. Map Truss secrets to environment variables so that backend_api can read them
        for key, val in self._secrets.items():
            if val:
                os.environ[key] = str(val)

        # 2. Add vggt-space to sys.path so we can import from vggt
        vggt_space_path = "/app/vggt-space"
        if vggt_space_path not in sys.path:
            sys.path.append(vggt_space_path)

        os.environ["PYTHONPATH"] = vggt_space_path
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["BASE_DIR"] = "/app/vggt_room3d_jobs"
        os.makedirs(os.environ["BASE_DIR"], exist_ok=True)

        # 3. Load VGGT model to GPU
        print("--> Loading VGGT Model to GPU...", flush=True)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cuda":
            dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
        else:
            dtype = torch.float32

        self.device = device
        self.dtype = dtype

        from vggt.models.vggt import VGGT
        self._model = VGGT.from_pretrained("facebook/VGGT-1B").to(device)
        self._model.eval()

        # 4. Inject model and device/dtype into backend_api
        from model import backend_api_extended_manhattan
        backend_api_extended_manhattan.model = self._model
        backend_api_extended_manhattan.device = self.device
        backend_api_extended_manhattan.dtype = self.dtype
        
        self.backend = backend_api_extended_manhattan
        print("--> Model loaded successfully!", flush=True)

    def predict(self, model_input):
        """
        Predict endpoint. Handles both nested "input" format or direct flat JSON dict.
        """
        job_input = model_input.get("input", model_input) if isinstance(model_input, dict) else {}
        
        zip_url = job_input.get("zip_url")
        if not zip_url:
            return {"status": "error", "error": "Missing 'zip_url' in input"}
            
        batch_id = job_input.get("batch_id", "batch")
        metadata = job_input.get("metadata", [])
        conf_threshold = float(job_input.get("conf_threshold", 1.0))
        max_points = int(job_input.get("max_points", 3000000))
        hard_max_points = int(job_input.get("hard_max_points", 300_000_000))
        
        max_images = job_input.get("max_images")
        if max_images is not None:
            max_images = int(max_images)
            
        clean_voxel = float(job_input.get("clean_voxel", 0.012))
        clean_stat_neighbors = int(job_input.get("clean_stat_neighbors", 16))
        clean_stat_std = float(job_input.get("clean_stat_std", 2.8))

        # Generate unique job directories
        job_id = str(uuid.uuid4())[:8]
        base_dir = os.environ.get("BASE_DIR", "/app/vggt_room3d_jobs")
        job_dir = os.path.join(base_dir, job_id)
        img_dir = os.path.join(job_dir, "images")
        out_dir = os.path.join(job_dir, "output")
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(out_dir, exist_ok=True)

        try:
            # Download and extract the zip file
            image_paths = self.backend.download_and_extract_zip(zip_url, job_dir, img_dir)
            if not image_paths:
                return {"status": "error", "error": "No valid images found in the zip file"}

            # Run inference pipeline
            meta = self.backend.run_inference_pipeline(
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
                self.backend.zip_job(job_dir, job_id)
            except Exception:
                pass
            return {"status": "error", "job_id": job_id, "error": str(e)}
