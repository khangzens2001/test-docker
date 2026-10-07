# ==============================================================================
# Dockerfile for VGGT 1B AI Server (Pre-packaged Weights)
# Platform: RunPod, FPT GPU Container, or local GPU instances
# ==============================================================================

FROM pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime

# 1. Thiết lập các biến môi trường hệ thống
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app/vggt-space \
    BASE_DIR=/app/vggt_room3d_jobs \
    HF_HOME=/app/hf_cache

# 2. Cài đặt các package hệ thống cần thiết (cho OpenCV, Open3D, font CJK, git, v.v.)
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    libegl1 \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    libusb-1.0-0 \
    libsm6 \
    libice6 \
    libxext6 \
    libxrender1 \
    libx11-6 \
    fonts-noto-cjk \
    && rm -rf /var/lib/apt/lists/*

# 3. Clone repo VGGT của Hugging Face
WORKDIR /app
RUN git clone https://huggingface.co/spaces/JianyuanWang/VGGT /app/vggt-space

# 4. Cài đặt các thư viện Python & nâng cấp libstdc++ cho Open3D
WORKDIR /app/vggt-space
RUN conda install -y -c conda-forge libstdcxx-ng && \
    pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir -r requirements_demo.txt && \
    pip install --no-cache-dir fastapi uvicorn python-multipart aiofiles boto3 open3d gdown runpod huggingface-hub==0.24.0 safetensors opencv-python-headless scipy matplotlib shapely pandas trimesh ezdxf pypdf Pillow onnxruntime pyarrow pydantic-settings cachetools numba requests

# 5. Tải trước trọng số model VGGT-1B từ Hugging Face và lưu vào cache (không load vào RAM để tránh OOM)
ARG HF_TOKEN=""
RUN python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='facebook/VGGT-1B', token='${HF_TOKEN}' if '${HF_TOKEN}' else None)"

# 6. Copy core modules, app, weights, floorplan generator, icons, scripts, file API Server chính và handler vào thư mục space
COPY app /app/vggt-space/app
COPY weights /app/vggt-space/weights
COPY floorplan_generator /app/vggt-space/floorplan_generator
COPY icon /app/vggt-space/icon
COPY scripts /app/vggt-space/scripts
COPY run_pipeline_url.py /app/vggt-space/run_pipeline_url.py
COPY backend_api_extended_manhattan.py /app/vggt-space/backend_api_extended_manhattan.py
COPY handler.py /app/vggt-space/handler.py

# 7. Tải trước trọng số model depthor.onnx từ Hugging Face (nếu chưa có trong context build)
ARG HF_DEPTHOR_TOKEN=""
RUN if [ ! -f /app/vggt-space/weights/depthor.onnx ]; then \
        echo "--> Downloading depthor_plus.onnx from Hugging Face trungshin99/depthor-onnx..." && \
        mkdir -p /app/vggt-space/weights && \
        python -c "import os, shutil; from huggingface_hub import hf_hub_download; p = hf_hub_download(repo_id='trungshin99/depthor-onnx', filename='depthor_plus.onnx', local_dir='/app/vggt-space/weights', token='${HF_DEPTHOR_TOKEN}' if '${HF_DEPTHOR_TOKEN}' else None); t = '/app/vggt-space/weights/depthor.onnx'; shutil.move(p, t) if p != t and os.path.exists(p) else None" && \
        echo "--> depthor.onnx downloaded successfully!"; \
    else \
        echo "--> depthor.onnx already exists in build context, skipping download."; \
    fi

# 8. Tạo thư mục để lưu các jobs tạm thời
RUN mkdir -p /app/vggt_room3d_jobs && chmod -R 777 /app/vggt_room3d_jobs

# Mở port 8000 của API
EXPOSE 8000

# 8. Khởi chạy RunPod handler
CMD ["python", "-u", "handler.py"]
