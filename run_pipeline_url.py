#!/usr/bin/env python3
"""
Run Pipeline Direct from URL
Usage:
    python run_pipeline_url.py "<URL>"
"""
import sys
import os
import json
import uuid
import time
import argparse

import backend_api_extended_manhattan as backend

def run(url: str, batch_id: str = "cli_batch"):
    job_id = f"job_{int(time.time())}_{str(uuid.uuid4())[:6]}"
    job_dir = os.path.join(backend.BASE_DIR, job_id)
    img_dir = os.path.join(job_dir, "input")
    out_dir = os.path.join(job_dir, "output")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)

    print("=" * 80)
    print("🚀 BẮT ĐẦU CHẠY PIPELINE TÁI TẠO 3D & MẶT BẰNG PHÒNG")
    print(f"🔗 URL Gói Dữ Liệu : {url}")
    print(f"📁 Thư mục Job      : {job_dir}")
    print("=" * 80)

    # 1. Download và giải nén dữ liệu
    print("\n[Bước 1/3] ⏳ Đang tải và giải nén dữ liệu từ liên kết...")
    t_start = time.time()
    image_paths = backend.download_and_extract_zip(url, job_dir, img_dir)
    print(f"-> Tải và giải nén hoàn tất ({time.time() - t_start:.2f}s).")

    # 2. Nhận diện cấu trúc dữ liệu
    session_dir = backend.find_multisensor_session_dir(img_dir) or backend.find_multisensor_session_dir(job_dir)
    
    if session_dir is not None:
        print(f"\n[Bước 2/3] 📡 Phát hiện gói dữ liệu Multi-Sensor LiDAR Session tại: {session_dir}")
        print("-> Đang thực thi Standalone Pipeline (Depth Completion, VIO, TSDF Fusion, Manhattan Alignment, CAD Floorplan)...")
        t_pipe = time.time()
        meta = backend.run_multisensor_pipeline(session_dir, batch_id, job_id, out_dir)
        pipe_duration = time.time() - t_pipe
        print(f"-> Pipeline hoàn tất trong {pipe_duration:.2f}s!")
    else:
        print(f"\n[Bước 2/3] 📸 Phát hiện gói dữ liệu ảnh chụp thông thường ({len(image_paths)} ảnh).")
        print("-> Đang thực thi mô hình VGGT-1B 3D Reconstruction...")
        t_pipe = time.time()
        meta = backend.run_inference_pipeline(
            image_paths=image_paths,
            batch_id=batch_id,
            metadata=[],
            conf_threshold=1.0,
            max_points=3_000_000,
            hard_max_points=backend.DEFAULT_HARD_MAX_POINTS,
            max_images=None,
            clean_voxel=backend.DEFAULT_CLEAN_VOXEL,
            clean_stat_neighbors=backend.DEFAULT_CLEAN_STAT_NEIGHBORS,
            clean_stat_std=backend.DEFAULT_CLEAN_STAT_STD,
            job_id=job_id,
            job_dir=job_dir,
            img_dir=img_dir,
            out_dir=out_dir
        )
        pipe_duration = time.time() - t_pipe
        print(f"-> Tái tạo 3D hoàn tất trong {pipe_duration:.2f}s!")

    # 3. Thu thập toàn bộ artifacts vào out_dir
    import shutil
    if session_dir and os.path.exists(session_dir):
        for fname in os.listdir(session_dir):
            if fname.endswith((".png", ".pdf", ".glb", ".ply", ".json")) and not fname.startswith("temp"):
                src = os.path.join(session_dir, fname)
                dst = os.path.join(out_dir, fname)
                if not os.path.exists(dst):
                    shutil.copy2(src, dst)

    # 4. Tổng hợp kết quả
    print("\n[Bước 3/3] 📊 KẾT QUẢ ĐO ĐẠC HÌNH HỌC CĂN PHÒNG (ROOM METRICS):")
    print("=" * 80)
    room_metrics = meta.get("room_metrics")
    if room_metrics:
        print(json.dumps(room_metrics, indent=2, ensure_ascii=False))
    else:
        print("Chưa có metrics chi tiết.")

    print("\n📂 DANH SÁCH FILE KẾT QUẢ TRONG THƯ MỤC OUTPUT:")
    print("-" * 80)
    for fname in sorted(os.listdir(out_dir)):
        fpath = os.path.join(out_dir, fname)
        size_kb = os.path.getsize(fpath) / 1024
        print(f"  • {fname:30s} ({size_kb:8.1f} KB) -> {fpath}")

    print("=" * 80)
    print(f"🎉 TỔNG THỜI GIAN THỰC HIỆN: {time.time() - t_start:.2f}s")
    print(f"📁 Thư mục lưu kết quả: {out_dir}")
    print("=" * 80)
    return meta

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run 3D Reconstruction Pipeline directly from URL")
    parser.add_argument("url", help="Google Drive link or direct ZIP link")
    parser.add_argument("--batch-id", default="cli_test", help="Batch ID for output grouping")
    args = parser.parse_args()

    run(args.url, args.batch_id)
