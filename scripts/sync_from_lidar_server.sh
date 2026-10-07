#!/bin/bash
# ==============================================================================
# Script đồng bộ 1:1 các module thuật toán từ repo lidar_room_scan_server
# sang test-docker để phục vụ build Docker và deploy RunPod Serverless.
#
# Usage: ./scripts/sync_from_lidar_server.sh [path_to_lidar_room_scan_server]
# ==============================================================================

set -e

SOURCE_DIR="${1}"
if [ -z "$SOURCE_DIR" ]; then
    if [ -d "/home/zenzen2411/development/projects/lidar_room_scan_server" ]; then
        SOURCE_DIR="/home/zenzen2411/development/projects/lidar_room_scan_server"
    else
        SOURCE_DIR="/Users/trungshin/product/lidar/lidar_room_scan_server"
    fi
fi

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
mkdir -p "$TARGET_DIR/scripts"

echo "📦 1. Đồng bộ app/__init__.py..."
cp "$SOURCE_DIR/app/__init__.py" "$TARGET_DIR/app/"

echo "📦 2. Đồng bộ app/pipeline/..."
rsync -av --delete --exclude="__pycache__" "$SOURCE_DIR/app/pipeline/" "$TARGET_DIR/app/pipeline/"

echo "📦 3. Đồng bộ app/services/..."
rsync -av --delete --exclude="__pycache__" "$SOURCE_DIR/app/services/" "$TARGET_DIR/app/services/"

echo "📦 4. Đồng bộ app/schemas/..."
rsync -av --delete --exclude="__pycache__" "$SOURCE_DIR/app/schemas/" "$TARGET_DIR/app/schemas/"

echo "📦 5. Đồng bộ app/core/ (loại bỏ Celery & SQLite DB, bảo tồn MODEL_WEIGHTS_PATH)..."
rsync -av --exclude="__pycache__" "$SOURCE_DIR/app/core/" "$TARGET_DIR/app/core/"
rm -f "$TARGET_DIR/app/core/celery_app.py" "$TARGET_DIR/app/core/database.py" "$TARGET_DIR/app/core/cleanup.py"

# Đảm bảo MODEL_WEIGHTS_PATH được resolve đường dẫn tuyệt đối chuẩn
python3 -c "
import os
path = '$TARGET_DIR/app/core/config.py'
with open(path, 'r', encoding='utf-8') as f:
    content = f.read()

snippet = '''        if not os.path.isabs(self.MODEL_WEIGHTS_PATH):
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            abs_weights = os.path.join(project_root, self.MODEL_WEIGHTS_PATH)
            if os.path.exists(abs_weights):
                self.MODEL_WEIGHTS_PATH = abs_weights
'''

if 'project_root = os.path.dirname' not in content:
    target = 'DATABASE_URL = (\n                f\"sqlite+aiosqlite:///{os.path.join(self.DATA_DIR, \'server.db\')}\"\n            )\n'
    if target in content:
        content = content.replace(target, target + snippet)
        with open(path, 'w', encoding='utf-8') as f:
            f.write(content)
        print('  -> Đã cập nhật resolve MODEL_WEIGHTS_PATH trong app/core/config.py')
"

echo "📦 6. Đồng bộ app/static/ (icons & fonts)..."
rsync -av --exclude="__pycache__" "$SOURCE_DIR/app/static/" "$TARGET_DIR/app/static/"

echo "📦 7. Đồng bộ scripts/export_vggt_keyframes.py..."
if [ -f "$SOURCE_DIR/scripts/export_vggt_keyframes.py" ]; then
    cp "$SOURCE_DIR/scripts/export_vggt_keyframes.py" "$TARGET_DIR/scripts/export_vggt_keyframes.py"
    touch "$TARGET_DIR/scripts/__init__.py"
    # Điều chỉnh import load_pose_table_for_tsdf từ app.pipeline.runner thay vì app.tasks.pipeline
    python3 -c "
path = '$TARGET_DIR/scripts/export_vggt_keyframes.py'
with open(path, 'r', encoding='utf-8') as f:
    c = f.read()
c = c.replace('from app.tasks.pipeline import load_pose_table_for_tsdf', '''try:
    from app.pipeline.runner import load_pose_table_for_tsdf
except ImportError:
    from app.tasks.pipeline import load_pose_table_for_tsdf''')
with open(path, 'w', encoding='utf-8') as f:
    f.write(c)
"
    echo "  -> Đã cập nhật export_vggt_keyframes.py không phụ thuộc app.tasks.pipeline."
fi

# Tối ưu vggt_runner.py để tái sử dụng preloaded model từ handler/backend và tương thích docker path
python3 -c "
import os
path = '$TARGET_DIR/app/services/vggt_runner.py'
if os.path.exists(path):
    with open(path, 'r', encoding='utf-8') as f:
        c = f.read()
    
    # 1. Tương thích DEFAULT_VGGT_SPACE
    old_space = 'DEFAULT_VGGT_SPACE = \"/home/zenzen2411/development/projects/test-docker/vggt-space\"'
    new_space = 'DEFAULT_VGGT_SPACE = \"/app/vggt-space\" if os.path.isdir(\"/app/vggt-space\") else \"/home/zenzen2411/development/projects/test-docker/vggt-space\"'
    if old_space in c:
        c = c.replace(old_space, new_space)
        
    # 2. Safe import export_vggt_keyframes
    old_imp = 'from scripts.export_vggt_keyframes import export_vggt_keyframes'
    new_imp = '''try:
    from scripts.export_vggt_keyframes import export_vggt_keyframes
except ImportError:
    from export_vggt_keyframes import export_vggt_keyframes'''
    if old_imp in c:
        c = c.replace(old_imp, new_imp)

    # 3. Model preloading reuse
    old_load = '''    model = VGGT.from_pretrained(model_id).to(device, dtype=dtype)
    model.eval()
    # Disable depth head to save VRAM; only point head + camera head are needed for 3D layout prior
    model.depth_head = None
    _patch_vggt_pos_embed(model)'''

    new_load = '''    preloaded_model = None
    if \"backend_api_extended_manhattan\" in sys.modules:
        preloaded_model = getattr(sys.modules[\"backend_api_extended_manhattan\"], \"model\", None)

    if preloaded_model is not None:
        logger.info(\"Reusing preloaded VGGT model from backend_api_extended_manhattan.\")
        model = preloaded_model
        _patch_vggt_pos_embed(model)
        should_del_model = False
    else:
        model = VGGT.from_pretrained(model_id).to(device, dtype=dtype)
        model.eval()
        model.depth_head = None
        _patch_vggt_pos_embed(model)
        should_del_model = True'''
        
    if old_load in c:
        c = c.replace(old_load, new_load)
        c = c.replace('if \"model\" in locals():\n            del model', 'if should_del_model and \"model\" in locals():\n            del model')

    with open(path, 'w', encoding='utf-8') as f:
        f.write(c)
    print('  -> Đã tối ưu vggt_runner.py cho môi trường Docker và tái sử dụng GPU VRAM model')
"

if [ -f "$SOURCE_DIR/app/models/depthor_plus.onnx" ]; then
    echo "📦 8. Cập nhật weights/depthor.onnx..."
    cp "$SOURCE_DIR/app/models/depthor_plus.onnx" "$TARGET_DIR/weights/depthor.onnx"
elif [ -f "$SOURCE_DIR/weights/depthor.onnx" ]; then
    echo "📦 8. Cập nhật weights/depthor.onnx..."
    cp "$SOURCE_DIR/weights/depthor.onnx" "$TARGET_DIR/weights/depthor.onnx"
fi

echo "🔍 9. Kiểm tra cú pháp Python toàn bộ project..."
python3 -m py_compile \
    "$TARGET_DIR/handler.py" \
    "$TARGET_DIR/backend_api_extended_manhattan.py" \
    "$TARGET_DIR/app/pipeline/runner.py" \
    "$TARGET_DIR/app/services/vggt_runner.py" \
    "$TARGET_DIR/scripts/export_vggt_keyframes.py"

echo "================================================================="
echo "✅ Đồng bộ hoàn tất 100%! Đã sẵn sàng deploy hoặc build Docker."
echo "================================================================="
