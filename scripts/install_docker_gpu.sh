#!/usr/bin/env bash
# ==============================================================================
# Script tự động cài đặt Docker Engine và NVIDIA Container Toolkit trên Ubuntu 24.04
# Cho phép Docker container nhận diện GPU NVIDIA RTX 2060
# ==============================================================================
set -e

echo "=== 1. Cập nhật hệ thống và cài đặt gói phụ thuộc ==="
sudo apt-get update
sudo apt-get install -y ca-certificates curl gnupg

echo "=== 2. Cài đặt Docker Engine chính thức ==="
sudo install -m 0755 -d /etc/apt/keyrings
sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
sudo chmod a+r /etc/apt/keyrings/docker.asc

echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu \
  $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | \
  sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

sudo apt-get update
sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

echo "=== 3. Cấu hình phân quyền người dùng (chạy docker không cần sudo) ==="
sudo usermod -aG docker "$USER"

echo "=== 4. Cài đặt NVIDIA Container Toolkit (GPU Passthrough) ==="
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
  && curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit

echo "=== 5. Cấu hình Docker daemon để hỗ trợ NVIDIA runtime ==="
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

echo "=== 6. Kiểm tra GPU trong Docker ==="
sudo docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi

echo "=================================================================="
echo "✅ Cài đặt Docker & NVIDIA GPU Toolkit thành công!"
echo "👉 Lưu ý: Để dùng lệnh 'docker' không cần sudo, hãy logout rồi login lại phiên làm việc Ubuntu."
echo "=================================================================="
