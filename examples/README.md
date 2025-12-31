# Examples

This directory contains usage examples for the YOLOv5 detection agent on Jetson Nano.

## Available Examples

### 1. TensorRT Inference (`example_trt_inference.py`)

Basic TensorRT inference examples:
- Basic synchronous inference
- Asynchronous inference with CUDA Stream
- Performance benchmark

**Usage:**
```bash
# Example 1: Basic inference
python3 examples/example_trt_inference.py --example 1

# Example 2: Async inference
python3 examples/example_trt_inference.py --example 2

# Example 3: Benchmark
python3 examples/example_trt_inference.py --example 3
```

### 2. YOLOv5 Detection (`example_detection.py`)

YOLOv5 detection and memory monitoring examples:
- Single image detection
- Memory monitoring demonstration
- Batch detection
- FPS counter

**Usage:**
```bash
# Example 1: Image detection
python3 examples/example_detection.py --example 1

# Example 2: Memory monitor
python3 examples/example_detection.py --example 2

# Example 3: Batch detection
python3 examples/example_detection.py --example 3

# Example 4: FPS counter
python3 examples/example_detection.py --example 4
```

### 3. GStreamer Camera (`example_camera.py`)

CSI camera integration examples:
- GStreamer pipeline generation
- Camera opening and testing
- Different resolutions
- Image capture
- Test script generation

**Usage:**
```bash
# Example 1: Show CSI pipeline
python3 examples/example_camera.py --example 1

# Example 2: Open camera
python3 examples/example_camera.py --example 2

# Example 3: Show different resolutions
python3 examples/example_camera.py --example 3

# Example 4: Capture and save image
python3 examples/example_camera.py --example 4

# Example 5: Generate camera test script
python3 examples/example_camera.py --example 5
```

## Prerequisites

Before running the examples, make sure you have:

1. **Converted a YOLOv5 model to TensorRT engine:**
   ```bash
   python3 run_detection.py --mode convert --model yolov5s
   ```

2. **Set up the environment** (see `setup_jetson.sh` and `docs/jetson_nano_setup.md`)

3. **For camera examples:** A CSI camera connected to Jetson Nano

## Quick Start

To run all examples in sequence:

```bash
# Convert model first (if not done)
python3 run_detection.py --mode convert --model yolov5s

# Run TensorRT examples
python3 examples/example_trt_inference.py --example 1
python3 examples/example_trt_inference.py --example 3

# Run detection examples
python3 examples/example_detection.py --example 1
python3 examples/example_detection.py --example 2

# Run camera examples (if CSI camera available)
python3 examples/example_camera.py --example 1
python3 examples/example_camera.py --example 5
```

## Troubleshooting

### Engine not found
If you get "Engine file not found" error:
```bash
python3 run_detection.py --mode convert --model yolov5s
```

### Camera not available
If camera examples fail:
- Ensure CSI camera is properly connected
- Run the generated test script: `./test_camera.sh`
- Try USB camera instead (change camera_id)

### Memory errors
If you encounter memory errors:
- Enable SWAP (see `docs/jetson_nano_setup.md`)
- Reduce batch size or image resolution
- Use the memory monitor (enabled by default)

## Performance Tips

1. **Use MAXN mode** for maximum performance:
   ```bash
   sudo nvpmodel -m 0
   sudo jetson_clocks
   ```

2. **Enable SWAP** to prevent OOM:
   ```bash
   sudo fallocate -l 4G /mnt/4GB.swap
   sudo mkswap /mnt/4GB.swap
   sudo swapon /mnt/4GB.swap
   ```

3. **Use FP16 mode** (default) for better performance on Jetson Nano

4. **Monitor memory** usage with the built-in memory monitor

## More Information

- See `README_DETECTION.md` for complete documentation
- See `docs/jetson_nano_setup.md` for environment setup
- See the main repository README for project overview
