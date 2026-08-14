import asyncio
import os
import sys

# Try to import the vastai SDK
try:
    from vastai import Serverless
except ImportError:
    print("❌ Lỗi: Thư viện 'vastai' chưa được cài đặt trong môi trường Python hiện tại.")
    print("👉 Vui lòng cài đặt bằng lệnh: pip install vastai")
    sys.exit(1)

# ==============================================================================
# CẤU HÌNH VAST.AI
# ==============================================================================
# 1. Tên Endpoint mà bạn đã tạo trên Vast.ai (ví dụ: "vggt-1b-serverless")
ENDPOINT_NAME = os.getenv("VAST_ENDPOINT_NAME", "YOUR_VAST_ENDPOINT_NAME")

# 2. Link ZIP chứa các ảnh cần test (ở đây dùng link Google Drive test từ trước)
GDRIVE_ZIP_URL = "https://drive.google.com/uc?id=1Wwmg1uurWJP2KfILFb5SffjubVGW-saJ"

# ==============================================================================
# HÀM CALL API CHÍNH
# ==============================================================================
async def main():
    # Kiểm tra API Key
    api_key = os.getenv("VAST_API_KEY")
    if not api_key:
        print("❌ Lỗi: Thiếu VAST_API_KEY trong biến môi trường!")
        print("👉 Vui lòng thiết lập bằng lệnh:")
        print("   export VAST_API_KEY=\"your_api_key_here\"")
        sys.exit(1)

    if ENDPOINT_NAME == "YOUR_VAST_ENDPOINT_NAME":
        print("⚠️ Cảnh báo: Vui lòng thay thế 'YOUR_VAST_ENDPOINT_NAME' bằng tên Endpoint thực tế của bạn.")
        print("👉 Hoặc chạy script bằng cách đặt biến môi trường: VAST_ENDPOINT_NAME=\"tên_của_bạn\" python test_vast_api.py")
        sys.exit(1)

    print(f"⏳ Đang kết nối tới Vast.ai Serverless...")
    print(f"📡 Target Endpoint: {ENDPOINT_NAME}")
    
    try:
        # Khởi tạo Vast.ai Serverless client (tự động dùng VAST_API_KEY từ env)
        async with Serverless() as client:
            # Lấy thông tin endpoint
            endpoint = await client.get_endpoint(ENDPOINT_NAME)
            
            # Chuẩn bị payload để gửi (giống với RunPod handler)
            payload = {
                "zip_url": GDRIVE_ZIP_URL,
                "batch_id": "vast_test_batch",
                "conf_threshold": 1.0,
                "max_points": 3000000,
                "clean_voxel": 0.012,
                "clean_stat_neighbors": 16,
                "clean_stat_std": 2.8
            }
            
            print("🚀 Đang gửi request xử lý ảnh (predict_zip_url) tới Vast.ai...")
            # Gửi request xử lý tới route '/predict_zip_url'
            response = await endpoint.request("/predict_zip_url", payload)
            
            print("\n✅ Nhận phản hồi thành công từ Vast.ai:")
            print("=================================================================")
            import json
            print(json.dumps(response, indent=2, ensure_ascii=False))
            print("=================================================================")
            
    except Exception as e:
        print(f"\n❌ Lỗi khi gửi request: {e}")
        print("👉 Hãy kiểm tra xem Endpoint của bạn đã active chưa và VAST_API_KEY có chính xác không.")

if __name__ == "__main__":
    asyncio.run(main())
