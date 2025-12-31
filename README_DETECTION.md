# YOLOv5 Detection Agent for Jetson Nano

[中文](#中文) | [English](#english)

---

## 中文

### 项目简介

这是一个专为 NVIDIA Jetson Nano 优化的 YOLOv5 目标检测系统。基于 ultralytics/yolov5 v6.0，使用 TensorRT 加速推理，支持 CSI 摄像头输入，并包含内存监控熔断机制。

### 主要特性

- ✅ **TensorRT 加速**: 使用 FP16 精度，针对 Maxwell 架构 GPU 优化
- ✅ **CUDA Stream 异步处理**: 压榨 GPU 性能，提高推理速度
- ✅ **内存监控熔断**: 当可用内存 < 200MB 时自动触发清理，防止系统卡死
- ✅ **CSI 摄像头支持**: 完整的 GStreamer 管道配置
- ✅ **完整工具链**: 从模型下载到部署的一站式解决方案

### 系统要求

- **硬件**: NVIDIA Jetson Nano 4GB
- **系统**: JetPack 4.6.6 (L4T R32.7.4)
- **CUDA**: 10.2
- **TensorRT**: 8.2.1
- **Python**: 3.6.9

### 快速开始

#### 1. 环境配置

```bash
# 克隆仓库
git clone https://github.com/Yulong-Cauli/encryption-then-compression-jpeg.git
cd encryption-then-compression-jpeg

# 运行自动配置脚本
chmod +x setup_jetson.sh
./setup_jetson.sh

# 手动安装 PyTorch 1.10.0
wget https://nvidia.box.com/shared/static/fjtbno0vpo676a25cgvuqc1wty0fkkg6.whl -O torch-1.10.0-cp36-cp36m-linux_aarch64.whl
pip3 install torch-1.10.0-cp36-cp36m-linux_aarch64.whl

# 手动安装 Torchvision 0.11.1
git clone --branch v0.11.1 https://github.com/pytorch/vision torchvision
cd torchvision
export BUILD_VERSION=0.11.1
python3 setup.py install --user
cd ..
```

#### 2. 性能优化（推荐）

```bash
# 切换到最大性能模式
sudo nvpmodel -m 0

# 锁定最大 CPU 频率
sudo jetson_clocks

# （可选）启用 4GB SWAP
sudo fallocate -l 4G /mnt/4GB.swap
sudo chmod 600 /mnt/4GB.swap
sudo mkswap /mnt/4GB.swap
sudo swapon /mnt/4GB.swap
```

#### 3. 运行检测

##### 方式一：完整流程（推荐新手）

```bash
# 自动下载 YOLOv5s 模型，转换为 TensorRT，并运行检测
python3 run_detection.py --mode full --model yolov5s --source 0 --display
```

##### 方式二：分步执行

```bash
# Step 1: 转换模型
python3 run_detection.py --mode convert --model yolov5s

# Step 2: 运行检测
python3 run_detection.py --mode detect --engine yolov5s_fp16.engine --source 0 --display
```

##### 方式三：直接使用 Agent

```bash
# 使用 CSI 摄像头
python3 src/yolov5_agent.py --engine yolov5s_fp16.engine --source 0 --display

# 使用 USB 摄像头
python3 src/yolov5_agent.py --engine yolov5s_fp16.engine --source 0 --display

# 处理视频文件
python3 src/yolov5_agent.py --engine yolov5s_fp16.engine --source video.mp4 --output output.mp4
```

### 核心组件说明

#### 1. TensorRT 推理引擎 (`src/trt_engine.py`)

高性能 TensorRT 推理引擎，支持：
- CUDA Stream 异步推理
- FP16 精度加速
- 自动内存管理
- 引擎预热

**示例：**

```python
from src.trt_engine import TensorRTInferenceEngine

# 加载引擎
engine = TensorRTInferenceEngine("yolov5s_fp16.engine")

# 预热
engine.warmup(num_iterations=10)

# 推理
output = engine.infer(input_data)
```

#### 2. YOLOv5 检测 Agent (`src/yolov5_agent.py`)

完整的检测系统，包含：
- 图像预处理和后处理
- NMS (非极大值抑制)
- 内存监控熔断机制
- FPS 计数器
- 可视化工具

**示例：**

```python
from src.yolov5_agent import YOLOv5Agent

# 初始化 Agent
agent = YOLOv5Agent(
    engine_path="yolov5s_fp16.engine",
    conf_threshold=0.25,
    iou_threshold=0.45
)

# 检测
detections = agent.detect(image)

# 可视化
result = agent.draw_detections(image, detections)
```

#### 3. GStreamer 摄像头接口

**CSI 摄像头管道：**

支持的摄像头：
- ✅ IMX219 (Raspberry Pi Camera v2) - 8MP
- ✅ IMX477 (Raspberry Pi HQ Camera) - 12MP  
- ✅ 其他兼容 nvarguscamerasrc 的 CSI 摄像头

```python
from src.yolov5_agent import GStreamerCamera

# 打开 CSI 摄像头
cap = GStreamerCamera.open_camera(
    camera_id=0,      # 摄像头 ID (0 或 1)
    width=1280,       # 分辨率宽度
    height=720,       # 分辨率高度
    fps=30            # 帧率
)
```

**完整管道字符串：**

```
nvarguscamerasrc sensor-id=0 ! 
video/x-raw(memory:NVMM), width=(int)1280, height=(int)720, format=(string)NV12, framerate=(fraction)30/1 ! 
nvvidconv flip-method=0 ! 
video/x-raw, width=(int)1280, height=(int)720, format=(string)BGRx ! 
videoconvert ! 
video/x-raw, format=(string)BGR ! 
appsink
```

#### 4. 内存监控熔断机制

自动监控系统内存，防止 OOM：

```python
from src.yolov5_agent import MemoryMonitor

# 初始化监控器（阈值 200MB）
monitor = MemoryMonitor(threshold_mb=200, check_interval=1.0)

# 启动监控
monitor.start()

# 检查是否安全运行
if monitor.is_safe_to_run():
    # 执行推理
    pass
else:
    # 跳过或清理
    monitor.trigger_cleanup()
```

### 性能参考

在 Jetson Nano 上使用 YOLOv5s + TensorRT FP16 的性能：

| 模型 | 输入尺寸 | FP16 | FPS (approx) |
|------|----------|------|--------------|
| YOLOv5n | 640x640 | ✓ | ~15-20 |
| YOLOv5s | 640x640 | ✓ | ~10-15 |
| YOLOv5m | 640x640 | ✓ | ~5-8 |

*注：实际性能取决于场景复杂度和系统负载*

### 故障排除

#### 1. 摄像头无法打开

```bash
# 检查摄像头是否连接
ls /dev/video*

# 测试 CSI 摄像头
gst-launch-1.0 nvarguscamerasrc ! nvoverlaysink

# 如果 CSI 失败，尝试 USB 摄像头
python3 run_detection.py --mode detect --engine yolov5s_fp16.engine --source 0
```

#### 2. 内存不足

```bash
# 检查内存使用
free -h

# 启用 SWAP
sudo fallocate -l 4G /mnt/4GB.swap
sudo mkswap /mnt/4GB.swap
sudo swapon /mnt/4GB.swap
```

#### 3. TensorRT 构建失败

```bash
# 检查 TensorRT 版本
python3 -c "import tensorrt as trt; print(trt.__version__)"

# 确保有足够的工作空间
# 如果失败，尝试减小 max_workspace_size
```

### 项目结构

```
.
├── docs/
│   └── jetson_nano_setup.md          # 详细环境配置文档
├── src/
│   ├── trt_engine.py                 # TensorRT 推理引擎
│   ├── yolov5_agent.py               # YOLOv5 检测 Agent
│   └── etc_tool.py                   # JPEG 加密工具（原项目）
├── setup_jetson.sh                    # 环境配置脚本
├── run_detection.py                   # 检测主程序
└── README_DETECTION.md                # 本文档
```

### 参考资源

- [ultralytics/yolov5 v6.0](https://github.com/ultralytics/yolov5/tree/v6.0)
- [NVIDIA Jetson Nano Developer Kit](https://developer.nvidia.com/embedded/jetson-nano-developer-kit)
- [TensorRT Documentation](https://docs.nvidia.com/deeplearning/tensorrt/)
- [JetPack 4.6.6 Release](https://developer.nvidia.com/embedded/jetpack-sdk-466)

---

## English

### Introduction

A YOLOv5 object detection system optimized for NVIDIA Jetson Nano. Based on ultralytics/yolov5 v6.0, accelerated with TensorRT, supporting CSI camera input, and featuring a memory monitoring circuit breaker.

### Key Features

- ✅ **TensorRT Acceleration**: FP16 precision optimized for Maxwell architecture GPU
- ✅ **CUDA Stream Async Processing**: Maximizes GPU performance and inference speed
- ✅ **Memory Circuit Breaker**: Auto cleanup when free memory < 200MB to prevent system freeze
- ✅ **CSI Camera Support**: Complete GStreamer pipeline configuration
- ✅ **Complete Toolchain**: End-to-end solution from model download to deployment

### System Requirements

- **Hardware**: NVIDIA Jetson Nano 4GB
- **OS**: JetPack 4.6.6 (L4T R32.7.4)
- **CUDA**: 10.2
- **TensorRT**: 8.2.1
- **Python**: 3.6.9

### Quick Start

#### 1. Environment Setup

```bash
# Clone repository
git clone https://github.com/Yulong-Cauli/encryption-then-compression-jpeg.git
cd encryption-then-compression-jpeg

# Run auto setup script
chmod +x setup_jetson.sh
./setup_jetson.sh

# Manually install PyTorch 1.10.0
wget https://nvidia.box.com/shared/static/fjtbno0vpo676a25cgvuqc1wty0fkkg6.whl -O torch-1.10.0-cp36-cp36m-linux_aarch64.whl
pip3 install torch-1.10.0-cp36-cp36m-linux_aarch64.whl

# Manually install Torchvision 0.11.1
git clone --branch v0.11.1 https://github.com/pytorch/vision torchvision
cd torchvision
export BUILD_VERSION=0.11.1
python3 setup.py install --user
cd ..
```

#### 2. Performance Optimization (Recommended)

```bash
# Switch to maximum performance mode
sudo nvpmodel -m 0

# Lock maximum CPU frequency
sudo jetson_clocks

# (Optional) Enable 4GB SWAP
sudo fallocate -l 4G /mnt/4GB.swap
sudo chmod 600 /mnt/4GB.swap
sudo mkswap /mnt/4GB.swap
sudo swapon /mnt/4GB.swap
```

#### 3. Run Detection

##### Option 1: Full Workflow (Recommended for Beginners)

```bash
# Auto download YOLOv5s model, convert to TensorRT, and run detection
python3 run_detection.py --mode full --model yolov5s --source 0 --display
```

##### Option 2: Step-by-Step

```bash
# Step 1: Convert model
python3 run_detection.py --mode convert --model yolov5s

# Step 2: Run detection
python3 run_detection.py --mode detect --engine yolov5s_fp16.engine --source 0 --display
```

##### Option 3: Direct Agent Usage

```bash
# Use CSI camera
python3 src/yolov5_agent.py --engine yolov5s_fp16.engine --source 0 --display

# Use USB camera
python3 src/yolov5_agent.py --engine yolov5s_fp16.engine --source 0 --display

# Process video file
python3 src/yolov5_agent.py --engine yolov5s_fp16.engine --source video.mp4 --output output.mp4
```

### Core Components

#### 1. TensorRT Inference Engine (`src/trt_engine.py`)

High-performance TensorRT inference engine with:
- CUDA Stream async inference
- FP16 precision acceleration
- Automatic memory management
- Engine warmup

**Example:**

```python
from src.trt_engine import TensorRTInferenceEngine

# Load engine
engine = TensorRTInferenceEngine("yolov5s_fp16.engine")

# Warmup
engine.warmup(num_iterations=10)

# Inference
output = engine.infer(input_data)
```

#### 2. YOLOv5 Detection Agent (`src/yolov5_agent.py`)

Complete detection system including:
- Image preprocessing and postprocessing
- NMS (Non-Maximum Suppression)
- Memory monitoring circuit breaker
- FPS counter
- Visualization tools

**Example:**

```python
from src.yolov5_agent import YOLOv5Agent

# Initialize agent
agent = YOLOv5Agent(
    engine_path="yolov5s_fp16.engine",
    conf_threshold=0.25,
    iou_threshold=0.45
)

# Detect
detections = agent.detect(image)

# Visualize
result = agent.draw_detections(image, detections)
```

#### 3. GStreamer Camera Interface

**CSI Camera Pipeline:**

Supported cameras:
- ✅ IMX219 (Raspberry Pi Camera v2) - 8MP
- ✅ IMX477 (Raspberry Pi HQ Camera) - 12MP
- ✅ Other CSI cameras compatible with nvarguscamerasrc

```python
from src.yolov5_agent import GStreamerCamera

# Open CSI camera
cap = GStreamerCamera.open_camera(
    camera_id=0,      # Camera ID (0 or 1)
    width=1280,       # Resolution width
    height=720,       # Resolution height
    fps=30            # Frame rate
)
```

#### 4. Memory Circuit Breaker

Auto monitor system memory to prevent OOM:

```python
from src.yolov5_agent import MemoryMonitor

# Initialize monitor (threshold 200MB)
monitor = MemoryMonitor(threshold_mb=200, check_interval=1.0)

# Start monitoring
monitor.start()

# Check if safe to run
if monitor.is_safe_to_run():
    # Execute inference
    pass
else:
    # Skip or cleanup
    monitor.trigger_cleanup()
```

### Performance Reference

YOLOv5s + TensorRT FP16 on Jetson Nano:

| Model | Input Size | FP16 | FPS (approx) |
|-------|------------|------|--------------|
| YOLOv5n | 640x640 | ✓ | ~15-20 |
| YOLOv5s | 640x640 | ✓ | ~10-15 |
| YOLOv5m | 640x640 | ✓ | ~5-8 |

*Note: Actual performance depends on scene complexity and system load*

### License

MIT License - See [LICENSE](LICENSE) file for details.

### Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

### Contact

For issues and questions, please open an issue on GitHub.
