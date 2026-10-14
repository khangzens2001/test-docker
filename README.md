# VGGT-1B & Multi-Sensor LiDAR 3D Room Reconstruction AI Server

Docker image đóng gói sẵn trọng số mô hình (`facebook/VGGT-1B` & `depthor.onnx`) cùng toàn bộ pipeline **Multi-Sensor LiDAR Room Scan** (Server VIO, Depthor+ Depth Completion, TSDF Integration, VGGT-1B Topology Prior, Manhattan Alignment, CAD Floorplan & Japanese Material/Tatami Estimation).

---

## ⚡ 1. Tối Ưu Hóa Mặc Định Cho Môi Trường RunPod Serverless

Cấu hình hiện tại của [`Dockerfile`](Dockerfile), [`handler.py`](handler.py) và [`app/core/config.py`](app/core/config.py) được **tối ưu hóa chuyên biệt cho mô hình RunPod Serverless Worker** (mỗi container xử lý tuần tự **1 job tại một thời điểm** trên GPU 16GB–24GB+ VRAM):

1. **Tăng tốc GPU cho cả Depthor+ và VGGT-1B (Adaptive Parallel Reconstruction)**:
   - `onnxruntime-gpu` (`CUDAExecutionProvider`) chạy song song với tiến trình con `ParallelVGGTWorker` (`torch.float16` trên CUDA) khi GPU có VRAM $\ge 11\text{ GB}$.
   - `LD_LIBRARY_PATH` được cấu hình trỏ trực tiếp tới thư viện CUDA 12 & cuDNN 9 bên trong môi trường Conda/PyTorch để `libonnxruntime_providers_cuda.so` luôn kích hoạt được GPU độc lập với thứ tự import.
2. **Ngăn chặn nhân bản VRAM khi `spawn` tiến trình con (`handler.py`)**:
   - Không nạp mô hình `VGGT-1B` ở phạm vi top-level của `handler.py`. Khi `FrontendPipeline` (OpenCV VIO) hoặc `ParallelVGGTWorker` khởi tạo tiến trình con qua `multiprocessing` (`spawn`), tiến trình con khởi động trong `< 2ms` và không tốn VRAM dư thừa.
3. **Mở khóa đa luồng OpenMP (`OMP_NUM_THREADS=4`) cho Open3D**:
   - Vì mỗi RunPod Serverless Worker chỉ xử lý 1 job/lượt, `OMP_NUM_THREADS` được đặt mặc định là `4` để tăng tốc các tác vụ CPU nặng của **Open3D** (Poisson Surface Reconstruction 120.000 tam giác, Voxel Downsampling, Statistical Outlier Removal, RANSAC Plane Segmentation).
4. **Warm-up sẵn Numba JIT & Matplotlib Font Cache từ lúc Build Image**:
   - Các hàm toán học ESKF/SO(3)/RTS Smoother (`numba_accelerated.py`) được biên dịch sẵn vào `/tmp/numba_cache` và cache font CJK của Matplotlib được khởi tạo ngay trong bước `docker build`, giúp loại bỏ ~18–20 giây trễ ở job đầu tiên (cold start).
5. **Tự động dọn dẹp ổ đĩa (`CLEANUP_JOB_DIR`)**:
   - Sau khi đẩy toàn bộ kết quả (`.ply`, `.glb`, `.pdf`, `.png`, `.json`) lên Cloudflare R2, worker tự động xóa thư mục tạm của job (`/app/vggt_room3d_jobs/<job_id>`) để tránh đầy ổ đĩa tạm trên các warm worker chạy lâu dài.

---

## ⚠️ 2. Lưu Ý Quan Trọng Khi Triển Khai Trên Các Loại AI Server Khác (`OMP_NUM_THREADS`)

> **[!IMPORTANT]**
> File [`Dockerfile`](Dockerfile) và [`app/core/config.py`](app/core/config.py) hiện đang đặt mặc định **`OMP_NUM_THREADS=4`** (kèm `MKL_NUM_THREADS=2`, `OPENBLAS_NUM_THREADS=2`, `ORT_NUM_THREADS=2`) vì trên **RunPod Serverless**, mỗi container chỉ phục vụ **1 job duy nhất tại một thời điểm**.
>
> Nếu bạn triển khai image này sang **các loại AI Server khác** (như Dedicated GPU Server chạy nhiều Celery workers, FastAPI/Uvicorn multi-worker, Vast.ai, FPT GPU Container hoặc Kubernetes pod chia sẻ CPU), **bạn CẦN điều chỉnh lại `OMP_NUM_THREADS`** để tránh hiện tượng tranh chấp luồng CPU (CPU thread contention / thrashing) làm suy giảm hiệu năng toàn hệ thống.

### Bảng cấu hình khuyến nghị theo loại AI Server

| Loại Hạ Tầng AI Server | Đặc điểm vận hành | Khuyến nghị `OMP_NUM_THREADS` | Các biến môi trường đi kèm nên chỉnh |
| :--- | :--- | :---: | :--- |
| **RunPod Serverless** *(Mặc định)* | 1 container = 1 job tại một thời điểm (8–16 vCPU, 16–24GB+ VRAM) | `OMP_NUM_THREADS=4` *(Mặc định)* | Giữ nguyên mặc định trong `Dockerfile` |
| **Dedicated Server / Celery Multi-Worker** *(VD: `lidar_room_scan_server`)* | Nhiều worker chạy song song nhiều job trên cùng 1 máy chủ vật lý | `OMP_NUM_THREADS=1` | `MKL_NUM_THREADS=1`<br/>`OPENBLAS_NUM_THREADS=1`<br/>`ORT_NUM_THREADS=1`<br/>`VIO_CV_THREADS=1` |
| **FastAPI / Uvicorn Server** (`backend_api_extended_manhattan.py`) | Chạy API trực tiếp nhận nhiều request đồng thời | `OMP_NUM_THREADS=1` hoặc `2` | `MKL_NUM_THREADS=1`<br/>`OPENBLAS_NUM_THREADS=1`<br/>`PARALLEL_RECONSTRUCTION_ENABLED=0` *(nếu chạy đồng thời nhiều job trên 1 GPU)* |
| **Dedicated GPU Cỡ Lớn (Đơn Job)** *(Vast.ai / AWS g5.4xlarge / A100 chạy 1 job/lần)* | 1 job độc quyền toàn bộ 16–32+ vCPU | `OMP_NUM_THREADS=8` | `VIO_CV_THREADS=4`<br/>`ONNX_INTRA_THREADS=4` |
| **GPU VRAM Thấp (< 12GB VRAM)** *(VD: RTX 3060 8GB / T4)* | Không đủ VRAM chạy song song `Depthor+` và `VGGT-1B` cùng lúc | `OMP_NUM_THREADS=2` hoặc `4` | `PARALLEL_RECONSTRUCTION_ENABLED=0` *(ép chạy tuần tự để tránh OOM)* |

### Cách ghi đè `OMP_NUM_THREADS` mà không cần sửa code

Vì [`app/core/config.py`](app/core/config.py) sử dụng `os.environ.setdefault(...)`, bạn có thể ghi đè trực tiếp qua biến môi trường khi khởi chạy container trên bất kỳ nền tảng nào:

#### Ví dụ 1: Chạy trên Server đa Worker / Đa luồng (hạ `OMP_NUM_THREADS=1` chống nghẽn CPU)
```bash
docker run -d --gpus all \
  -e OMP_NUM_THREADS=1 \
  -e MKL_NUM_THREADS=1 \
  -e OPENBLAS_NUM_THREADS=1 \
  -e ORT_NUM_THREADS=1 \
  -e VIO_CV_THREADS=1 \
  -e PARALLEL_RECONSTRUCTION_ENABLED=0 \
  -p 8000:8000 \
  roomlayout-vggt-server:latest \
  uvicorn backend_api_extended_manhattan:app --host 0.0.0.0 --port 8000
```

#### Ví dụ 2: Chạy trên Dedicated GPU Server nhiều nhân CPU xử lý đơn Job (tăng `OMP_NUM_THREADS=8`)
```bash
docker run --rm --gpus all \
  -e OMP_NUM_THREADS=8 \
  -e ONNX_INTRA_THREADS=4 \
  roomlayout-vggt-server:latest \
  python run_pipeline_url.py "<GOOGLE_DRIVE_OR_PRESIGNED_ZIP_URL>"
```

---

## 🛠️ 3. Danh Sách Các Biến Môi Trường Cấu Hình

| Biến môi trường | Mặc định | Ý nghĩa |
| :--- | :--- | :--- |
| `OMP_NUM_THREADS` | `4` | Số luồng OpenMP cho Open3D (Poisson mesh, RANSAC, KDTree) và C++ backend. Hạ về `1` nếu chạy nhiều worker song song. |
| `MKL_NUM_THREADS` / `OPENBLAS_NUM_THREADS` | `2` | Số luồng tính toán đại số tuyến tính của NumPy / SciPy. |
| `ORT_NUM_THREADS` / `ONNX_INTRA_THREADS` | `2` | Số luồng CPU nội bộ của ONNX Runtime. |
| `VIO_CV_THREADS` | `4` | Số luồng OpenCV trong cả tiến trình chính (`vio.py`) và tiến trình con Visual Frontend (`FrontendPipeline`), ngăn việc `os.cpu_count()` lấy toàn bộ 64–256 nhân vật lý của host RunPod. |
| `ENABLE_VGGT` | `true` | Bật/tắt nhánh suy luận mô hình `facebook/VGGT-1B`. |
| `PARALLEL_RECONSTRUCTION_ENABLED` | `true` | Chạy song song `Depthor+` (ONNX GPU) và `VGGT-1B` (PyTorch GPU) bằng 2 tiến trình độc lập khi VRAM $\ge$ `MIN_PARALLEL_VRAM_GB`. |
| `MIN_PARALLEL_VRAM_GB` | `11.0` | Ngưỡng VRAM tối thiểu (GB) để kích hoạt chế độ chạy song song. |
| `CLEANUP_JOB_DIR` | `1` *(khi có R2)* / `0` *(local)* | Tự động xóa thư mục tạm `/app/vggt_room3d_jobs/<job_id>` sau khi hoàn tất job. |
| `R2_ENDPOINT_URL` / `R2_BUCKET` | `""` / `"3d-ply"` | Cấu hình Cloudflare R2 (S3-compatible) để lưu trữ artifacts đầu ra. |
| `R2_ACCESS_KEY_ID` / `R2_SECRET_ACCESS_KEY` | `""` | Khóa xác thực Cloudflare R2. |

---

## 🚀 4. Quy Trình Đồng Bộ, Build & Kiểm Thử

### Bước 1: Đồng bộ thuật toán mới nhất từ `lidar_room_scan_server`
```bash
./scripts/sync_from_lidar_server.sh /path/to/lidar_room_scan_server
```

### Bước 2: Build Docker Image
```bash
docker build -t roomlayout-vggt-server:latest .
```

### Bước 3: Kiểm thử trực tiếp bằng CLI trong Container (giống máy Local)
```bash
docker run --rm --gpus all \
  -v "$(pwd)/vggt_room3d_jobs:/app/vggt_room3d_jobs" \
  roomlayout-vggt-server:latest \
  python run_pipeline_url.py "https://drive.google.com/file/d/<FILE_ID>/view?usp=drive_link" --batch-id test_local
```
