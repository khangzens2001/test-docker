#!/bin/bash
# ==============================================================================
# ZenS - AI 360 Room Layout
# Script tự động Build, Tag và Push VGGT 1B AI Server Image lên Docker Hub
# Usage: ./push_to_dockerhub.sh <your_dockerhub_username> [tag_version]
# Ví dụ: ./push_to_dockerhub.sh thinhpld0 v2.0.0
# ==============================================================================

set -e

DOCKER_USER=${1}
TAG=${2:-"v2.0.0"}

if [ -z "$DOCKER_USER" ]; then
  echo "❌ Lỗi: Vui lòng nhập Docker Hub username!"
  echo "👉 Cú pháp: ./push_to_dockerhub.sh <dockerhub_username> [tag]"
  echo "👉 Ví dụ:   ./push_to_dockerhub.sh thinhpld0 v2.0.0"
  exit 1
fi

# Tìm kiếm binary docker
DOCKER_BIN="docker"
if ! command -v docker &> /dev/null; then
  if [ -f "/Applications/Docker.app/Contents/Resources/bin/docker" ]; then
    DOCKER_BIN="/Applications/Docker.app/Contents/Resources/bin/docker"
  else
    echo "❌ Lỗi: Không tìm thấy lệnh 'docker' trong hệ thống."
    echo "👉 Vui lòng mở ứng dụng Docker Desktop."
    exit 1
  fi
fi

IMAGE_NAME="vggt-1b"
TARGET_IMAGE="$DOCKER_USER/$IMAGE_NAME"

echo "================================================================="
echo "🚀 Đang chuẩn bị Build và Push VGGT 1B AI Server Image lên Docker Hub"
echo "  👤 Account : $DOCKER_USER"
echo "  🏷️  Tag     : $TAG & latest"
echo "  🖥️  Platform: linux/amd64 (Dùng cho RunPod, FPT GPU Container)"
echo "================================================================="

# 1. Kiểm tra đăng nhập Docker Hub
echo "🔒 Kiểm tra trạng thái đăng nhập Docker Hub..."
if ! $DOCKER_BIN info 2>/dev/null | grep -q "Username:"; then
    echo "🔑 Bạn chưa đăng nhập Docker Hub. Đang gọi 'docker login'..."
    $DOCKER_BIN login
fi

# 2. Build Docker Image (sử dụng platform linux/amd64 để chạy được trên GPU cloud)
echo "🔨 Building VGGT 1B AI Server Image..."
$DOCKER_BIN build --platform linux/amd64 -t $IMAGE_NAME:latest -f Dockerfile .

# 3. Tag Image
echo "🏷️  Tagging Image..."
$DOCKER_BIN tag $IMAGE_NAME:latest $TARGET_IMAGE:$TAG
if [ "$TAG" != "latest" ]; then
  echo "ℹ️  Không ghi đè tag 'latest' để bảo vệ môi trường chạy RunPod hiện tại."
else
  $DOCKER_BIN tag $IMAGE_NAME:latest $TARGET_IMAGE:latest
fi

# 4. Push Image
echo "📤 Pushing Image ($TARGET_IMAGE:$TAG)..."
$DOCKER_BIN push $TARGET_IMAGE:$TAG
if [ "$TAG" == "latest" ]; then
  $DOCKER_BIN push $TARGET_IMAGE:latest
fi

echo ""
echo "================================================================="
echo "✅ Hoàn tất đẩy VGGT 1B AI Server Image lên Docker Hub thành công!"
echo "📦 Repository: https://hub.docker.com/r/$TARGET_IMAGE"
echo "================================================================="
