#!/bin/bash
# ==============================================================================
# Script đồng bộ 1:1 các module thuật toán từ repo lidar_room_scan_server
# sang feat-full-pipeline để phục vụ build Docker và deploy RunPod Serverless.
#
# Usage: ./scripts/sync_from_lidar_server.sh [path_to_lidar_room_scan_server]
# ==============================================================================

set -e

SOURCE_DIR="${1:-/Users/trungshin/product/lidar/lidar_room_scan_server}"
TARGET_DIR="$(cd "$(dirname "$0")/.." && pwd)"

if [ ! -d "$SOURCE_DIR/app/pipeline" ]; then
    echo "❌ Lỗi: Không tìm thấy thư mục nguồn hợp lệ tại: $SOURCE_DIR"
    echo "👉 Vui lòng truyền đường dẫn: ./scripts/sync_from_lidar_server.sh /path/to/lidar_room_scan_server"
    exit 1
fi

echo "================================================================="
echo "🔄 Bắt đầu đồng bộ thuật toán từ lidar_room_scan_server"
echo "  📂 Nguồn : $SOURCE_DIR"
echo "  🎯 Đích  : $TARGET_DIR"
echo "================================================================="

mkdir -p "$TARGET_DIR/app"
mkdir -p "$TARGET_DIR/weights"

echo "📦 1. Đồng bộ app/__init__.py..."
cp "$SOURCE_DIR/app/__init__.py" "$TARGET_DIR/app/"

echo "📦 2. Đồng bộ app/pipeline/..."
rsync -av --delete "$SOURCE_DIR/app/pipeline/" "$TARGET_DIR/app/pipeline/"

echo "📦 3. Đồng bộ app/services/..."
rsync -av --delete "$SOURCE_DIR/app/services/" "$TARGET_DIR/app/services/"

echo "📦 4. Đồng bộ app/schemas/..."
rsync -av --delete "$SOURCE_DIR/app/schemas/" "$TARGET_DIR/app/schemas/"

echo "📦 5. Đồng bộ app/core/ (loại bỏ Celery & SQLite DB)..."
rsync -av "$SOURCE_DIR/app/core/" "$TARGET_DIR/app/core/"
rm -f "$TARGET_DIR/app/core/celery_app.py" "$TARGET_DIR/app/core/database.py" "$TARGET_DIR/app/core/cleanup.py"

echo "📦 6. Đồng bộ app/static/ (icons & fonts)..."
rsync -av "$SOURCE_DIR/app/static/" "$TARGET_DIR/app/static/"

if [ -f "$SOURCE_DIR/app/models/depthor_plus.onnx" ]; then
    echo "📦 7. Cập nhật weights/depthor.onnx..."
    cp "$SOURCE_DIR/app/models/depthor_plus.onnx" "$TARGET_DIR/weights/depthor.onnx"
elif [ -f "$SOURCE_DIR/weights/depthor.onnx" ]; then
    echo "📦 7. Cập nhật weights/depthor.onnx..."
    cp "$SOURCE_DIR/weights/depthor.onnx" "$TARGET_DIR/weights/depthor.onnx"
fi

echo "🔍 8. Kiểm tra cú pháp Python..."
python3 -m py_compile "$TARGET_DIR/handler.py" "$TARGET_DIR/app/pipeline/runner.py"

echo "================================================================="
echo "✅ Đồng bộ hoàn tất 100%! Đã sẵn sàng deploy hoặc build Docker."
echo "================================================================="
