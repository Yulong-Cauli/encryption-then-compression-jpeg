# Jetson Nano 环境配置指南 / Jetson Nano Environment Setup Guide

## 设备信息 / Device Information

- **Hardware**: Jetson Nano 4GB
- **JetPack**: 4.6.6-b24 (L4T R32.7.4)
- **CUDA**: 10.2
- **cuDNN**: 8.2
- **TensorRT**: 8.2.1
- **Python**: 3.6.9 (aarch64)

## PyTorch 安装 / PyTorch Installation

### 官方 WHL 下载链接 / Official WHL Download Links

由于 JetPack 4.6.6 使用 Python 3.6 和 CUDA 10.2，需要使用 NVIDIA 官方预编译的 wheel：

**PyTorch 1.10.0 (Python 3.6, aarch64):**
```bash
# Download from NVIDIA official repository
wget https://nvidia.box.com/shared/static/fjtbno0vpo676a25cgvuqc1wty0fkkg6.whl -O torch-1.10.0-cp36-cp36m-linux_aarch64.whl

# Install
pip3 install torch-1.10.0-cp36-cp36m-linux_aarch64.whl
```

**Torchvision 0.11.1 (Compatible with PyTorch 1.10.0):**
```bash
# Install dependencies first
sudo apt-get install libjpeg-dev zlib1g-dev libpython3-dev libavcodec-dev libavformat-dev libswscale-dev

# Download and install
git clone --branch v0.11.1 https://github.com/pytorch/vision torchvision
cd torchvision
export BUILD_VERSION=0.11.1
python3 setup.py install --user
cd ..
```

### 验证安装 / Verify Installation

```python
import torch
import torchvision

print(f"PyTorch Version: {torch.__version__}")
print(f"Torchvision Version: {torchvision.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")
print(f"CUDA Version: {torch.version.cuda}")
print(f"cuDNN Version: {torch.backends.cudnn.version()}")
```

## 依赖安装 / Dependencies Installation

```bash
# System dependencies
sudo apt-get update
sudo apt-get install -y python3-pip python3-dev
sudo apt-get install -y libopencv-python libgstreamer1.0-dev gstreamer1.0-tools

# Python dependencies
pip3 install --upgrade pip
pip3 install numpy opencv-python scipy pyyaml matplotlib seaborn tqdm
pip3 install pycuda  # For TensorRT acceleration
```

## TensorRT Python API

TensorRT Python API is already included in JetPack 4.6.6:
```python
import tensorrt as trt
print(f"TensorRT Version: {trt.__version__}")
```

## 注意事项 / Important Notes

1. **内存管理**: Jetson Nano 4GB RAM 有限，建议启用 SWAP（至少 4GB）
2. **功率模式**: 使用 `sudo nvpmodel -m 0` 切换到 MAXN 模式以获得最佳性能
3. **散热**: 确保有良好的散热环境，推荐使用风扇
4. **相机**: CSI 摄像头需要正确连接到 CSI 接口
   - 支持 IMX219 (Raspberry Pi Camera v2)
   - 支持 IMX477 (Raspberry Pi HQ Camera)
   - 其他兼容 nvarguscamerasrc 的 CSI 摄像头

## 性能优化建议 / Performance Optimization

- 使用 TensorRT FP16 模式以提高推理速度
- 使用 CUDA Stream 进行异步推理
- 批处理时使用合适的 batch size (建议1-2)
- 监控内存使用，避免 OOM
