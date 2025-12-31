#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YOLOv5 模型转换和检测演示脚本
YOLOv5 Model Conversion and Detection Demo Script
"""

import os
import sys
import argparse
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def convert_yolov5_to_onnx(weights_path: str, img_size: int = 640):
    """
    将 YOLOv5 PyTorch 模型转换为 ONNX
    Convert YOLOv5 PyTorch model to ONNX
    
    Args:
        weights_path: YOLOv5 weights 文件路径 (e.g., yolov5s.pt)
        img_size: 输入图像尺寸
    
    Returns:
        ONNX 文件路径
    """
    try:
        import torch
        sys.path.insert(0, './yolov5')
        from models.experimental import attempt_load
        from utils.torch_utils import select_device
    except ImportError as e:
        logger.error(f"Failed to import required modules: {e}")
        logger.error("Please make sure YOLOv5 v6.0 is installed in ./yolov5 directory")
        sys.exit(1)
    
    logger.info(f"Converting {weights_path} to ONNX...")
    
    # Load model
    device = select_device('cpu')  # Use CPU for export
    model = attempt_load(weights_path, map_location=device)
    model.eval()
    
    # Dummy input
    dummy_input = torch.randn(1, 3, img_size, img_size)
    
    # Output path
    onnx_path = weights_path.replace('.pt', '.onnx')
    
    # Export
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        opset_version=11,
        input_names=['images'],
        output_names=['output'],
        dynamic_axes=None
    )
    
    logger.info(f"ONNX model saved to: {onnx_path}")
    return onnx_path


def convert_onnx_to_trt(onnx_path: str, engine_path: str = None, fp16: bool = True):
    """
    将 ONNX 模型转换为 TensorRT Engine
    Convert ONNX model to TensorRT Engine
    
    Args:
        onnx_path: ONNX 模型路径
        engine_path: 输出 engine 路径 (如果为 None，自动生成)
        fp16: 是否使用 FP16 精度
    
    Returns:
        Engine 文件路径
    """
    from src.trt_engine import TensorRTEngineBuilder
    
    if engine_path is None:
        engine_path = onnx_path.replace('.onnx', '_fp16.engine' if fp16 else '_fp32.engine')
    
    logger.info(f"Converting {onnx_path} to TensorRT engine...")
    logger.info(f"FP16 mode: {fp16}")
    
    TensorRTEngineBuilder.build_engine(
        onnx_path=onnx_path,
        engine_path=engine_path,
        fp16_mode=fp16,
        max_batch_size=1,
        max_workspace_size=1 << 30  # 1GB
    )
    
    logger.info(f"TensorRT engine saved to: {engine_path}")
    return engine_path


def download_yolov5_weights(model_name: str = 'yolov5s'):
    """
    下载 YOLOv5 预训练权重
    Download YOLOv5 pretrained weights
    
    Args:
        model_name: 模型名称 (yolov5s, yolov5m, yolov5l, yolov5x)
    
    Returns:
        权重文件路径
    """
    import urllib.request
    
    weights_url = f"https://github.com/ultralytics/yolov5/releases/download/v6.0/{model_name}.pt"
    weights_path = f"{model_name}.pt"
    
    if os.path.exists(weights_path):
        logger.info(f"Weights already exist: {weights_path}")
        return weights_path
    
    logger.info(f"Downloading {model_name} weights from {weights_url}...")
    
    try:
        urllib.request.urlretrieve(weights_url, weights_path)
        logger.info(f"Weights downloaded to: {weights_path}")
        return weights_path
    except Exception as e:
        logger.error(f"Failed to download weights: {e}")
        sys.exit(1)


def run_detection_demo(engine_path: str, source: str = "0", display: bool = False):
    """
    运行检测演示
    Run detection demo
    
    Args:
        engine_path: TensorRT engine 文件路径
        source: 视频源 (摄像头 ID 或视频文件)
        display: 是否显示窗口
    """
    from src.yolov5_agent import YOLOv5Agent, GStreamerCamera
    import cv2
    
    logger.info("Initializing YOLOv5 Agent...")
    agent = YOLOv5Agent(
        engine_path=engine_path,
        conf_threshold=0.25,
        iou_threshold=0.45
    )
    
    # Open video source
    logger.info(f"Opening video source: {source}")
    
    if source.isdigit():
        # CSI camera
        try:
            cap = GStreamerCamera.open_camera(camera_id=int(source))
        except:
            logger.warning("Failed to open CSI camera, trying USB camera...")
            cap = cv2.VideoCapture(int(source))
    else:
        # Video file
        cap = cv2.VideoCapture(source)
    
    if not cap.isOpened():
        logger.error("Failed to open video source")
        sys.exit(1)
    
    logger.info("Starting detection loop... Press 'q' to quit")
    
    frame_count = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                logger.warning("Failed to read frame")
                break
            
            # Detect
            detections = agent.detect(frame)
            
            # Draw
            result = agent.draw_detections(frame, detections)
            
            # Display
            if display:
                cv2.imshow("YOLOv5 Detection", result)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            # Log every 30 frames
            frame_count += 1
            if frame_count % 30 == 0 and detections:
                logger.info(f"Frame {frame_count}: Detected {len(detections)} objects")
    
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    
    finally:
        cap.release()
        if display:
            cv2.destroyAllWindows()
        agent.cleanup()
        logger.info("Demo complete")


def main():
    parser = argparse.ArgumentParser(
        description="YOLOv5 模型转换和检测工具 / YOLOv5 Model Conversion and Detection Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例 / Examples:

1. 完整流程 (下载模型 -> 转换 -> 检测) / Complete workflow:
   python run_detection.py --mode full --model yolov5s --source 0 --display

2. 仅转换模型 / Convert model only:
   python run_detection.py --mode convert --model yolov5s

3. 仅运行检测 / Run detection only:
   python run_detection.py --mode detect --engine yolov5s_fp16.engine --source 0 --display

4. 从视频文件检测 / Detect from video file:
   python run_detection.py --mode detect --engine yolov5s_fp16.engine --source video.mp4
        """
    )
    
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=['full', 'convert', 'detect'],
        help="运行模式 / Run mode: full (完整流程), convert (仅转换), detect (仅检测)"
    )
    
    parser.add_argument(
        "--model",
        type=str,
        default="yolov5s",
        choices=['yolov5n', 'yolov5s', 'yolov5m', 'yolov5l', 'yolov5x'],
        help="YOLOv5 模型 / YOLOv5 model (default: yolov5s)"
    )
    
    parser.add_argument(
        "--weights",
        type=str,
        help="PyTorch 权重文件路径 / PyTorch weights path (optional)"
    )
    
    parser.add_argument(
        "--engine",
        type=str,
        help="TensorRT engine 文件路径 / TensorRT engine path (for detect mode)"
    )
    
    parser.add_argument(
        "--source",
        type=str,
        default="0",
        help="视频源 / Video source: camera ID (0) or video file path"
    )
    
    parser.add_argument(
        "--display",
        action="store_true",
        help="显示检测窗口 / Display detection window"
    )
    
    parser.add_argument(
        "--fp16",
        action="store_true",
        default=True,
        help="使用 FP16 精度 / Use FP16 precision (default: True)"
    )
    
    parser.add_argument(
        "--img-size",
        type=int,
        default=640,
        help="输入图像尺寸 / Input image size (default: 640)"
    )
    
    args = parser.parse_args()
    
    # Mode: full
    if args.mode == 'full':
        logger.info("=== 完整流程 / Full Workflow ===")
        
        # Step 1: Download weights
        if args.weights:
            weights_path = args.weights
        else:
            weights_path = download_yolov5_weights(args.model)
        
        # Step 2: Convert to ONNX
        onnx_path = convert_yolov5_to_onnx(weights_path, img_size=args.img_size)
        
        # Step 3: Convert to TensorRT
        engine_path = convert_onnx_to_trt(onnx_path, fp16=args.fp16)
        
        # Step 4: Run detection
        run_detection_demo(engine_path, source=args.source, display=args.display)
    
    # Mode: convert
    elif args.mode == 'convert':
        logger.info("=== 模型转换 / Model Conversion ===")
        
        # Download or use provided weights
        if args.weights:
            weights_path = args.weights
        else:
            weights_path = download_yolov5_weights(args.model)
        
        # Convert to ONNX
        onnx_path = convert_yolov5_to_onnx(weights_path, img_size=args.img_size)
        
        # Convert to TensorRT
        engine_path = convert_onnx_to_trt(onnx_path, fp16=args.fp16)
        
        logger.info(f"Conversion complete! Engine: {engine_path}")
    
    # Mode: detect
    elif args.mode == 'detect':
        logger.info("=== 运行检测 / Run Detection ===")
        
        if not args.engine:
            logger.error("Please specify --engine for detect mode")
            sys.exit(1)
        
        if not os.path.exists(args.engine):
            logger.error(f"Engine file not found: {args.engine}")
            sys.exit(1)
        
        run_detection_demo(args.engine, source=args.source, display=args.display)


if __name__ == "__main__":
    main()
