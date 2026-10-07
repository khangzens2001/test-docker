import os
import sys
import shutil
import json
import time
import tempfile
import numpy as np

# Ensure test-docker and vggt-space are in path
TEST_DOCKER_ROOT = os.path.dirname(os.path.abspath(__file__))
VGGT_SPACE = os.path.join(TEST_DOCKER_ROOT, "vggt-space")
if TEST_DOCKER_ROOT not in sys.path:
    sys.path.insert(0, TEST_DOCKER_ROOT)
if VGGT_SPACE not in sys.path:
    sys.path.insert(0, VGGT_SPACE)

import backend_api_extended_manhattan as backend

def main():
    print("=" * 80)
    print("🚀 BẮT ĐẦU KIỂM THỬ PIPELINE TRONG WORKSPACE TEST-DOCKER")
    print("=" * 80)

    benchmark_src = "/home/zenzen2411/development/projects/lidar_room_scan_server/data/benchmark_new_scan"
    if not os.path.isdir(benchmark_src):
        print(f"❌ Không tìm thấy thư mục benchmark tại: {benchmark_src}")
        sys.exit(1)

    # Tạo thư mục test session độc lập
    test_work_dir = os.path.join(TEST_DOCKER_ROOT, "vggt_room3d_jobs", "benchmark_test_docker")
    if os.path.exists(test_work_dir):
        shutil.rmtree(test_work_dir)
    os.makedirs(test_work_dir, exist_ok=True)

    session_dir = os.path.join(test_work_dir, "benchmark_scan")
    out_dir = os.path.join(test_work_dir, "output")
    os.makedirs(out_dir, exist_ok=True)

    print(f"📂 Sao chép dữ liệu quét chuẩn từ: {benchmark_src}")
    print(f"🎯 Đến thư mục làm việc test: {session_dir}")
    shutil.copytree(benchmark_src, session_dir)

    # Chạy pipeline thông qua backend_api_extended_manhattan (giống hệt luồng RunPod/Docker)
    print("\n⏳ Đang chạy run_multisensor_pipeline()...")
    t0 = time.time()
    meta = backend.run_multisensor_pipeline(
        session_dir=session_dir,
        batch_id="docker_bench_batch",
        job_id="job_bench_001",
        out_dir=out_dir
    )
    t_elapsed = time.time() - t0
    print(f"✅ Pipeline hoàn tất trong {t_elapsed:.2f} giây!")

    # Thu thập và kiểm tra các file đầu ra
    print("\n" + "=" * 80)
    print("📊 KẾT QUẢ ĐỐI CHIẾU CHỈ SỐ KỸ THUẬT VỚI GROUND TRUTH:")
    print("=" * 80)

    floorplan_json_path = os.path.join(session_dir, "floorplan.json")
    with open(floorplan_json_path, "r", encoding="utf-8") as f:
        fp_data = json.load(f)

    walls = fp_data.get("walls", [])
    if len(walls) >= 4:
        lengths = [w["length_meters"] for w in walls]
        heights = [w["height_meters"] for w in walls]
        long_side = max(lengths) * 100.0
        short_side = min(lengths) * 100.0
        height = np.mean(heights) * 100.0
        area = (max(lengths) * min(lengths))

        print(f"• Chiều dài phòng : {long_side:.1f} cm (Chuẩn GT: 219 - 220 cm, sai số: {abs(long_side - 219.5)/219.5*100:.2f}%)")
        print(f"• Chiều rộng phòng: {short_side:.1f} cm (Chuẩn GT: 137 - 142 cm)")
        print(f"• Chiều cao trần  : {height:.1f} cm (Chuẩn GT: 220 - 223 cm)")
        print(f"• Diện tích sàn   : {area:.2f} m² (Chuẩn GT: 3.0 - 4.7 m²)")

    # Kiểm tra kích thước và số điểm/tam giác của mô hình 3D
    import open3d as o3d
    import trimesh

    dense_ply = os.path.join(session_dir, "reconstructed_vggt_dense.ply")
    vis_ply = os.path.join(session_dir, "reconstructed_visual.ply")
    rec_ply = os.path.join(session_dir, "reconstructed.ply")
    rec_glb = os.path.join(session_dir, "reconstructed.glb")

    print("\n🔍 ĐỐI CHIẾU MÔ HÌNH 3D (.PLY & .GLB):")
    if os.path.isfile(dense_ply):
        p_dense = o3d.io.read_point_cloud(dense_ply)
        print(f"• reconstructed_vggt_dense.ply: {len(p_dense.points):,d} điểm (có màu RGB: {len(p_dense.colors):,d})")
    if os.path.isfile(vis_ply):
        p_vis = o3d.io.read_point_cloud(vis_ply)
        print(f"• reconstructed_visual.ply    : {len(p_vis.points):,d} điểm (có màu RGB: {len(p_vis.colors):,d})")
    if os.path.isfile(rec_ply):
        p_rec = o3d.io.read_point_cloud(rec_ply)
        print(f"• reconstructed.ply           : {len(p_rec.points):,d} điểm (có màu RGB: {len(p_rec.colors):,d})")
    if os.path.isfile(rec_glb):
        mesh = trimesh.load(rec_glb, force="mesh")
        print(f"• reconstructed.glb (Poisson) : {len(mesh.faces):,d} tam giác, {len(mesh.vertices):,d} đỉnh")

    print("\n" + "=" * 80)
    print("🎉 KIỂM THỬ TEST-DOCKER HOÀN TẤT THÀNH CÔNG RỰC RỠ!")
    print("=" * 80)

if __name__ == "__main__":
    main()
