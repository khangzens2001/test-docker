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

# 2. Cài đặt các package hệ thống cần thiết (cho OpenCV, Open3D, git, v.v.)
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# 3. Clone repo VGGT của Hugging Face
WORKDIR /app
RUN git clone https://huggingface.co/spaces/JianyuanWang/VGGT /app/vggt-space

# 4. Cài đặt các thư viện Python
WORKDIR /app/vggt-space
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir -r requirements_demo.txt && \
    pip install --no-cache-dir fastapi uvicorn python-multipart aiofiles boto3 open3d

# 5. Tải trước trọng số model VGGT-1B từ Hugging Face và lưu vào cache (không load vào RAM để tránh OOM)
RUN python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='facebook/VGGT-1B')"

# 6. Copy file API Server chính vào thư mục space
COPY backend_api_extended_manhattan.py /app/vggt-space/backend_api_extended_manhattan.py

# 7. Tạo thư mục để lưu các jobs tạm thời
RUN mkdir -p /app/vggt_room3d_jobs && chmod -R 777 /app/vggt_room3d_jobs

# Mở port 8000 của API
EXPOSE 8000

# 8. Khởi chạy uvicorn server
CMD ["uvicorn", "backend_api_extended_manhattan:app", "--host", "0.0.0.0", "--port", "8000"]
