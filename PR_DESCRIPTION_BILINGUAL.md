# PR 描述 / PR Description

[中文](#中文版本) | [English](#english-version)

---

## 中文版本

为 NVIDIA Jetson Nano（JetPack 4.6.6，Maxwell GPU）实现了生产就绪的目标检测系统。解决了嵌入式 AI 部署中的硬件约束问题：PyTorch 1.10.0/Torchvision 0.11.1 兼容性、TensorRT 加速、CSI 摄像头集成和 OOM 预防。

### 核心组件

- **`src/trt_engine.py`**：TensorRT 推理引擎，具有 CUDA Stream 异步处理和针对 Maxwell 架构的 FP16 精度优化
- **`src/yolov5_agent.py`**：YOLOv5 v6.0 检测 Agent，包含预处理、NMS 和可视化管道
- **`src/yolov5_agent.py::MemoryMonitor`**：后台线程监控可用内存；当内存低于 200MB 阈值时触发 GC 并停止推理
- **`src/yolov5_agent.py::GStreamerCamera`**：CSI 摄像头接口，生成 nvarguscamerasrc 管道

### 使用方法

```python
from src.yolov5_agent import YOLOv5Agent

# 使用 TensorRT 引擎初始化
agent = YOLOv5Agent(
    engine_path="yolov5s_fp16.engine",
    enable_memory_monitor=True  # 低内存时自动停止
)

# 从 CSI 摄像头检测
detections = agent.detect(frame)
result = agent.draw_detections(frame, detections)
```

```bash
# 端到端：下载模型、转换为 TensorRT、运行检测
python3 run_detection.py --mode full --model yolov5s --source 0
```

### GStreamer 管道

为 Jetson CSI 摄像头提供完整的 nvarguscamerasrc 管道，支持可配置的分辨率、FPS 和翻转方法：

```python
pipeline = GStreamerCamera.get_csi_pipeline(
    camera_id=0, width=1280, height=720, fps=30, flip_method=0
)
# 返回: "nvarguscamerasrc sensor-id=0 ! video/x-raw(memory:NVMM), ..."
```

### 文档

- `docs/jetson_nano_setup.md`：PyTorch 1.10.0 wheel 下载链接和 Torchvision 0.11.1 构建说明（Python 3.6/aarch64）
- `README_DETECTION.md`：完整的 API 参考和部署指南（中英双语）
- `setup_jetson.sh`：自动化依赖安装
- `examples/`：3 个示例脚本，包含 TensorRT、检测和摄像头演示

### 实现要点

- **异步推理**：CUDA Stream 管道重叠 H2D 传输、内核执行和 D2H 传输
- **内存安全**：熔断器模式防止 Jetson Nano 因 OOM 而冻结
- **颜色调色板缓存**：为 80 个 COCO 类别预计算，避免重复随机种子设置
- **错误处理**：析构函数防止 TensorRT 引擎部分初始化失败

Jetson Nano 预期性能：YOLOv5s @ 640x640 FP16 约 10-15 FPS。

### 新增文件

- `docs/jetson_nano_setup.md` - Jetson Nano 环境配置指南
- `src/trt_engine.py` - TensorRT 推理引擎（CUDA Stream 异步处理）
- `src/yolov5_agent.py` - YOLOv5 检测 Agent（含内存监控熔断器）
- `setup_jetson.sh` - 自动化环境配置脚本
- `run_detection.py` - 主检测程序（支持模型转换和推理）
- `README_DETECTION.md` - 完整的中英双语用户文档
- `IMPLEMENTATION_SUMMARY.md` - 实施总结和技术细节
- `examples/` - 使用示例（TensorRT、检测、摄像头）

### 快速开始

```bash
# 1. 环境配置
./setup_jetson.sh

# 2. 手动安装 PyTorch/Torchvision（见 docs/jetson_nano_setup.md）

# 3. 运行检测
python3 run_detection.py --mode full --model yolov5s --source 0 --display
```

---

## English Version

Implements a production-ready object detection system optimized for NVIDIA Jetson Nano (JetPack 4.6.6, Maxwell GPU). Addresses requirements for embedded AI deployment with hardware constraints: PyTorch 1.10.0/Torchvision 0.11.1 compatibility, TensorRT acceleration, CSI camera integration, and OOM prevention.

### Core Components

- **`src/trt_engine.py`**: TensorRT inference engine with CUDA Stream async processing and FP16 precision for Maxwell architecture
- **`src/yolov5_agent.py`**: YOLOv5 v6.0 detection agent with preprocessing, NMS, and visualization pipeline
- **`src/yolov5_agent.py::MemoryMonitor`**: Background thread monitoring free memory; triggers GC and halts inference below 200MB threshold
- **`src/yolov5_agent.py::GStreamerCamera`**: CSI camera interface generating nvarguscamerasrc pipelines

### Usage

```python
from src.yolov5_agent import YOLOv5Agent

# Initialize with TensorRT engine
agent = YOLOv5Agent(
    engine_path="yolov5s_fp16.engine",
    enable_memory_monitor=True  # Auto-stops on low memory
)

# Detect from CSI camera
detections = agent.detect(frame)
result = agent.draw_detections(frame, detections)
```

```bash
# End-to-end: download model, convert to TensorRT, run detection
python3 run_detection.py --mode full --model yolov5s --source 0
```

### GStreamer Pipeline

Complete nvarguscamerasrc pipeline for Jetson CSI cameras with configurable resolution, FPS, and flip methods:

```python
pipeline = GStreamerCamera.get_csi_pipeline(
    camera_id=0, width=1280, height=720, fps=30, flip_method=0
)
# Returns: "nvarguscamerasrc sensor-id=0 ! video/x-raw(memory:NVMM), ..."
```

### Documentation

- `docs/jetson_nano_setup.md`: PyTorch 1.10.0 wheel URLs and Torchvision 0.11.1 build instructions for Python 3.6/aarch64
- `README_DETECTION.md`: Complete API reference and deployment guide (bilingual Chinese/English)
- `setup_jetson.sh`: Automated dependency installation
- `examples/`: 3 example scripts with TensorRT, detection, and camera demos

### Implementation Notes

- **Async inference**: CUDA Stream pipeline overlaps H2D transfer, kernel execution, and D2H transfer
- **Memory safety**: Circuit breaker pattern prevents Jetson Nano from freezing on OOM
- **Color palette caching**: Pre-computed for 80 COCO classes to avoid repeated random seeding
- **Error handling**: Destructor guards against partial initialization failures in TensorRT engine

Expected performance on Jetson Nano: YOLOv5s @ 640x640 FP16 ~10-15 FPS.

### New Files

- `docs/jetson_nano_setup.md` - Jetson Nano environment setup guide
- `src/trt_engine.py` - TensorRT inference engine (CUDA Stream async)
- `src/yolov5_agent.py` - YOLOv5 detection agent (with memory circuit breaker)
- `setup_jetson.sh` - Automated environment setup script
- `run_detection.py` - Main detection program (model conversion & inference)
- `README_DETECTION.md` - Complete bilingual user documentation
- `IMPLEMENTATION_SUMMARY.md` - Implementation summary and technical details
- `examples/` - Usage examples (TensorRT, detection, camera)

### Quick Start

```bash
# 1. Setup environment
./setup_jetson.sh

# 2. Install PyTorch/Torchvision manually (see docs/jetson_nano_setup.md)

# 3. Run detection
python3 run_detection.py --mode full --model yolov5s --source 0 --display
```

---

## 使用说明 / Instructions

请将上述内容复制到 PR 描述中，替换现有的英文描述。所有文档文件已经是中英双语的：

Please copy the above content to the PR description to replace the existing English-only description. All documentation files are already bilingual:

- ✅ `README_DETECTION.md` - 中英双语 / Bilingual
- ✅ `IMPLEMENTATION_SUMMARY.md` - 中英双语 / Bilingual  
- ✅ `docs/jetson_nano_setup.md` - 中英双语 / Bilingual
- ✅ `examples/README.md` - 中英双语 / Bilingual
- ✅ 代码注释 / Code comments - 中英双语 / Bilingual
