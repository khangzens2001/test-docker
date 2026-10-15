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
rsync -av --delete --exclude="__pycache__" --exclude="dashboard.py" "$SOURCE_DIR/app/services/" "$TARGET_DIR/app/services/"

echo "📦 4. Đồng bộ app/schemas/..."
rsync -av --delete --exclude="__pycache__" "$SOURCE_DIR/app/schemas/" "$TARGET_DIR/app/schemas/"

echo "📦 5. Đồng bộ app/core/ (loại bỏ Celery, SQLite DB & JWT, bảo tồn MODEL_WEIGHTS_PATH)..."
rsync -av --exclude="__pycache__" --exclude="security.py" "$SOURCE_DIR/app/core/" "$TARGET_DIR/app/core/"
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

old_threads = '''os.environ[\"OMP_NUM_THREADS\"] = \"1\"
os.environ[\"MKL_NUM_THREADS\"] = \"1\"
os.environ[\"OPENBLAS_NUM_THREADS\"] = \"1\"
os.environ[\"ARROW_IO_THREADS\"] = \"1\"
os.environ[\"NUMEXPR_NUM_THREADS\"] = \"1\"
os.environ[\"VECLIB_MAXIMUM_THREADS\"] = \"1\"
os.environ[\"ORT_NUM_THREADS\"] = \"1\"'''

new_threads = '''os.environ.setdefault(\"OMP_NUM_THREADS\", \"4\")
os.environ.setdefault(\"VIO_CV_THREADS\", \"4\")
os.environ.setdefault(\"MKL_NUM_THREADS\", \"2\")
os.environ.setdefault(\"OPENBLAS_NUM_THREADS\", \"2\")
os.environ.setdefault(\"ARROW_IO_THREADS\", \"2\")
os.environ.setdefault(\"NUMEXPR_NUM_THREADS\", \"2\")
os.environ.setdefault(\"VECLIB_MAXIMUM_THREADS\", \"2\")
os.environ.setdefault(\"ORT_NUM_THREADS\", \"2\")'''

if old_threads in content:
    content = content.replace(old_threads, new_threads)

with open(path, 'w', encoding='utf-8') as f:
    f.write(content)
print('  -> Đã cập nhật resolve MODEL_WEIGHTS_PATH, OMP_NUM_THREADS và VIO_CV_THREADS trong app/core/config.py')
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
if 'from app.pipeline.runner import load_pose_table_for_tsdf' not in c:
    c = c.replace('from app.tasks.pipeline import load_pose_table_for_tsdf', '''try:
    from app.pipeline.runner import load_pose_table_for_tsdf
except ImportError:
    from app.tasks.pipeline import load_pose_table_for_tsdf''')
with open(path, 'w', encoding='utf-8') as f:
    f.write(c)
"
    echo "  -> Đã cập nhật export_vggt_keyframes.py không phụ thuộc app.tasks.pipeline."
fi

KEYFRAME_SELECT_SRC="$(dirname "$SOURCE_DIR")/selected_keyframe/src/keyframe_select"
if [ -d "$KEYFRAME_SELECT_SRC" ]; then
    echo "📦 7b. Đồng bộ keyframe_select/..."
    mkdir -p "$TARGET_DIR/keyframe_select"
    rsync -av --delete --exclude="__pycache__" "$KEYFRAME_SELECT_SRC/" "$TARGET_DIR/keyframe_select/"
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
    if old_space in c and '/app/vggt-space' not in c:
        c = c.replace(old_space, new_space)
        
    # 2. Safe import export_vggt_keyframes
    old_imp = 'from scripts.export_vggt_keyframes import export_vggt_keyframes'
    new_imp = '''try:
    from scripts.export_vggt_keyframes import export_vggt_keyframes
except ImportError:
    from export_vggt_keyframes import export_vggt_keyframes'''
    if old_imp in c and 'from export_vggt_keyframes import export_vggt_keyframes' not in c:
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
        c = c.replace(
            'model.aggregator.forward = _optimized_aggregator_forward',
            'orig_aggregator_forward = getattr(model.aggregator, \"forward\", None)\n    model.aggregator.forward = _optimized_aggregator_forward',
        )
        old_ext = 'ext_np = extrinsic.detach().cpu().float().numpy()'
        new_ext = '''ext_np = extrinsic.detach().cpu().float().numpy()

        # Release GPU VRAM immediately before CPU-bound Open3D & NPZ steps
        if \"model\" in locals() and model is not None:
            if \"orig_aggregator_forward\" in locals() and orig_aggregator_forward is not None:
                model.aggregator.forward = orig_aggregator_forward
            if should_del_model:
                model = None
        orig_aggregator_forward = None
        _optimized_aggregator_forward = None
        images = None
        aggregated_tokens_list = None
        cam_tokens = None
        pose_enc_list = None
        pose_enc = None
        extrinsic = None
        intrinsic = None
        chunk_tokens = None
        img_chunk = None
        pts_c = None
        conf_c = None
        img_chunk_perm = None
        all_pts = None
        all_conf = None
        all_colors = None
        gc.collect()
        if device == \"cuda\":
            torch.cuda.empty_cache()'''
        c = c.replace(old_ext, new_ext, 1)

        old_finally = '''        if \"model\" in locals():
            del model
        if \"images\" in locals():
            del images
        if \"aggregated_tokens_list\" in locals():
            del aggregated_tokens_list
        if \"all_pts\" in locals():
            del all_pts
        if \"all_conf\" in locals():
            del all_conf
        if \"all_colors\" in locals():
            del all_colors
        if \"device\" in locals() and device == \"cuda\":
            torch.cuda.empty_cache()
        gc.collect()'''
        new_finally = '''        if \"model\" in locals() and model is not None:
            if \"orig_aggregator_forward\" in locals() and orig_aggregator_forward is not None:
                model.aggregator.forward = orig_aggregator_forward
            if should_del_model:
                model = None
        orig_aggregator_forward = None
        _optimized_aggregator_forward = None
        images = None
        aggregated_tokens_list = None
        all_pts = None
        all_conf = None
        all_colors = None
        gc.collect()
        if \"device\" in locals() and device == \"cuda\":
            torch.cuda.empty_cache()'''
        c = c.replace(old_finally, new_finally)

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
