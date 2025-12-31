#!/bin/bash
# Jetson Nano 环境配置脚本 / Environment Setup Script for Jetson Nano
# 适用于 JetPack 4.6.6 (L4T R32.7.4)

set -e

echo "=========================================="
echo "Jetson Nano YOLOv5 环境配置"
echo "Jetson Nano YOLOv5 Environment Setup"
echo "=========================================="

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if running on Jetson
if [ ! -f /etc/nv_tegra_release ]; then
    echo -e "${RED}警告: 此脚本设计用于 NVIDIA Jetson 设备${NC}"
    echo -e "${RED}Warning: This script is designed for NVIDIA Jetson devices${NC}"
    read -p "是否继续? Continue? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Function to print section
print_section() {
    echo ""
    echo -e "${GREEN}===> $1${NC}"
}

# 1. System Update
print_section "1. 更新系统 / Updating system packages"
sudo apt-get update
sudo apt-get install -y python3-pip python3-dev

# 2. Install system dependencies
print_section "2. 安装系统依赖 / Installing system dependencies"
sudo apt-get install -y \
    libopencv-python \
    libgstreamer1.0-dev \
    gstreamer1.0-tools \
    gstreamer1.0-plugins-good \
    gstreamer1.0-plugins-bad \
    gstreamer1.0-plugins-ugly \
    gstreamer1.0-libav \
    libjpeg-dev \
    zlib1g-dev \
    libpython3-dev \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev

# 3. Install Python dependencies
print_section "3. 安装 Python 依赖 / Installing Python dependencies"
pip3 install --upgrade pip

# Install basic dependencies
pip3 install numpy opencv-python scipy pyyaml matplotlib seaborn tqdm psutil

# 4. Install PyCUDA for TensorRT
print_section "4. 安装 PyCUDA / Installing PyCUDA"
pip3 install pycuda

# 5. Install PyTorch (manual step - requires download)
print_section "5. PyTorch 安装说明 / PyTorch Installation Instructions"
echo -e "${YELLOW}PyTorch 需要手动下载安装 / PyTorch requires manual installation${NC}"
echo ""
echo "请执行以下命令下载并安装 PyTorch 1.10.0:"
echo "Please execute the following commands to download and install PyTorch 1.10.0:"
echo ""
echo "  wget https://nvidia.box.com/shared/static/fjtbno0vpo676a25cgvuqc1wty0fkkg6.whl -O torch-1.10.0-cp36-cp36m-linux_aarch64.whl"
echo "  pip3 install torch-1.10.0-cp36-cp36m-linux_aarch64.whl"
echo ""

# 6. Install Torchvision (manual step)
print_section "6. Torchvision 安装说明 / Torchvision Installation Instructions"
echo -e "${YELLOW}Torchvision 需要从源码编译 / Torchvision requires building from source${NC}"
echo ""
echo "请执行以下命令安装 Torchvision 0.11.1:"
echo "Please execute the following commands to install Torchvision 0.11.1:"
echo ""
echo "  git clone --branch v0.11.1 https://github.com/pytorch/vision torchvision"
echo "  cd torchvision"
echo "  export BUILD_VERSION=0.11.1"
echo "  python3 setup.py install --user"
echo "  cd .."
echo ""

# 7. Download YOLOv5 v6.0
print_section "7. 下载 YOLOv5 v6.0 / Downloading YOLOv5 v6.0"
if [ ! -d "yolov5" ]; then
    git clone --branch v6.0 https://github.com/ultralytics/yolov5.git
    cd yolov5
    pip3 install -r requirements.txt
    cd ..
    echo -e "${GREEN}YOLOv5 v6.0 下载完成 / YOLOv5 v6.0 downloaded${NC}"
else
    echo "YOLOv5 目录已存在 / YOLOv5 directory already exists"
fi

# 8. Performance optimization
print_section "8. 性能优化建议 / Performance optimization recommendations"
echo ""
echo "建议执行以下优化 / Recommended optimizations:"
echo ""
echo "1. 切换到 MAXN 模式 (最大性能) / Switch to MAXN mode (maximum performance):"
echo "   sudo nvpmodel -m 0"
echo ""
echo "2. 设置最大 CPU 频率 / Set maximum CPU frequency:"
echo "   sudo jetson_clocks"
echo ""
echo "3. 启用 SWAP (如果内存不足) / Enable SWAP (if memory is insufficient):"
echo "   sudo systemctl disable nvzramconfig"
echo "   sudo fallocate -l 4G /mnt/4GB.swap"
echo "   sudo chmod 600 /mnt/4GB.swap"
echo "   sudo mkswap /mnt/4GB.swap"
echo "   sudo swapon /mnt/4GB.swap"
echo "   # 永久生效 / Make permanent:"
echo "   echo '/mnt/4GB.swap swap swap defaults 0 0' | sudo tee -a /etc/fstab"
echo ""

# 9. Verify installation
print_section "9. 验证安装 / Verifying installation"
echo ""
python3 << EOF
import sys
print("Python version:", sys.version)

try:
    import numpy as np
    print("✓ NumPy:", np.__version__)
except:
    print("✗ NumPy not installed")

try:
    import cv2
    print("✓ OpenCV:", cv2.__version__)
except:
    print("✗ OpenCV not installed")

try:
    import tensorrt as trt
    print("✓ TensorRT:", trt.__version__)
except:
    print("✗ TensorRT not installed")

try:
    import pycuda.driver as cuda
    import pycuda.autoinit
    print("✓ PyCUDA installed")
except:
    print("✗ PyCUDA not installed")

try:
    import torch
    print("✓ PyTorch:", torch.__version__)
    print("  CUDA available:", torch.cuda.is_available())
except:
    print("✗ PyTorch not installed (需要手动安装 / requires manual installation)")

try:
    import torchvision
    print("✓ Torchvision:", torchvision.__version__)
except:
    print("✗ Torchvision not installed (需要手动安装 / requires manual installation)")
EOF

echo ""
print_section "安装完成! / Installation complete!"
echo ""
echo "请参考上述说明完成 PyTorch 和 Torchvision 的安装"
echo "Please refer to the instructions above to complete PyTorch and Torchvision installation"
echo ""
echo "接下来的步骤 / Next steps:"
echo "1. 转换 YOLOv5 模型为 TensorRT engine"
echo "   Convert YOLOv5 model to TensorRT engine"
echo "2. 运行检测 Agent"
echo "   Run detection agent"
echo ""
echo "详细信息请参考 docs/jetson_nano_setup.md"
echo "For more details, see docs/jetson_nano_setup.md"
