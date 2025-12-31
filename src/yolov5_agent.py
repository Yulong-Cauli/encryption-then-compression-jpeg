#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YOLOv5 Detection Agent for Jetson Nano
基于 ultralytics/yolov5 v6.0 的目标检测 Agent
包含内存监控和熔断机制
"""

import os
import sys
import time
import psutil
import logging
import threading
import numpy as np
import cv2
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass

# Import TensorRT engine
from trt_engine import TensorRTInferenceEngine

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class Detection:
    """检测结果"""
    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2)
    confidence: float
    class_id: int
    class_name: str


class MemoryMonitor:
    """
    内存监控器 - 熔断机制
    当可用内存 < 200MB 时触发清理或停止推理
    """
    
    def __init__(self, threshold_mb: int = 200, check_interval: float = 1.0):
        """
        Args:
            threshold_mb: 内存阈值 (MB)
            check_interval: 检查间隔 (秒)
        """
        self.threshold_bytes = threshold_mb * 1024 * 1024
        self.check_interval = check_interval
        self.should_stop = False
        self.monitoring = False
        self.monitor_thread = None
        
    def start(self):
        """启动监控"""
        if self.monitoring:
            return
        
        self.monitoring = True
        self.should_stop = False
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        logger.info(f"Memory monitor started (threshold: {self.threshold_bytes / 1024 / 1024:.0f} MB)")
    
    def stop(self):
        """停止监控"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2.0)
        logger.info("Memory monitor stopped")
    
    def _monitor_loop(self):
        """监控循环"""
        while self.monitoring:
            mem_info = psutil.virtual_memory()
            free_mem = mem_info.available
            
            if free_mem < self.threshold_bytes:
                logger.warning(f"LOW MEMORY! Free: {free_mem / 1024 / 1024:.1f} MB < {self.threshold_bytes / 1024 / 1024:.0f} MB")
                self.should_stop = True
                self.trigger_cleanup()
            
            time.sleep(self.check_interval)
    
    def trigger_cleanup(self):
        """触发内存清理"""
        import gc
        logger.info("Triggering garbage collection...")
        gc.collect()
        
        # Log memory status after cleanup
        mem_info = psutil.virtual_memory()
        logger.info(f"Memory after cleanup - Free: {mem_info.available / 1024 / 1024:.1f} MB, Used: {mem_info.percent}%")
    
    def is_safe_to_run(self) -> bool:
        """检查是否安全运行"""
        return not self.should_stop
    
    def reset(self):
        """重置熔断状态"""
        self.should_stop = False
        logger.info("Memory monitor reset")


class YOLOv5Agent:
    """
    YOLOv5 检测 Agent
    基于 ultralytics/yolov5 v6.0
    """
    
    # COCO dataset class names (YOLOv5 v6.0 default)
    COCO_CLASSES = [
        'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
        'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
        'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
        'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
        'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
        'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
        'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
        'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
        'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
        'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
        'toothbrush'
    ]
    
    def __init__(self, 
                 engine_path: str,
                 conf_threshold: float = 0.25,
                 iou_threshold: float = 0.45,
                 input_size: Tuple[int, int] = (640, 640),
                 enable_memory_monitor: bool = True):
        """
        初始化 YOLOv5 Agent
        
        Args:
            engine_path: TensorRT engine 文件路径
            conf_threshold: 置信度阈值
            iou_threshold: NMS IoU 阈值
            input_size: 输入尺寸 (width, height)
            enable_memory_monitor: 是否启用内存监控
        """
        self.engine_path = engine_path
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.input_size = input_size
        
        # Initialize TensorRT engine
        logger.info(f"Loading TensorRT engine: {engine_path}")
        self.engine = TensorRTInferenceEngine(engine_path, max_batch_size=1)
        
        # Warmup
        self.engine.warmup(num_iterations=5)
        
        # Memory monitor
        self.memory_monitor = None
        if enable_memory_monitor:
            self.memory_monitor = MemoryMonitor(threshold_mb=200)
            self.memory_monitor.start()
        
        # FPS counter
        self.fps_counter = FPSCounter()
        
        # Pre-compute color palette for better performance
        self._color_palette = self._generate_color_palette()
        
        logger.info("YOLOv5 Agent initialized")
    
    def preprocess(self, image: np.ndarray) -> Tuple[np.ndarray, float, Tuple[int, int]]:
        """
        预处理图像
        
        Args:
            image: BGR image (H, W, C)
            
        Returns:
            (preprocessed_image, scale, pad)
        """
        # Get original shape
        h, w = image.shape[:2]
        
        # Resize with aspect ratio preservation
        scale = min(self.input_size[0] / w, self.input_size[1] / h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
        # Pad to input size
        pad_w = (self.input_size[0] - new_w) // 2
        pad_h = (self.input_size[1] - new_h) // 2
        
        padded = cv2.copyMakeBorder(
            resized, pad_h, self.input_size[1] - new_h - pad_h,
            pad_w, self.input_size[0] - new_w - pad_w,
            cv2.BORDER_CONSTANT, value=(114, 114, 114)
        )
        
        # Convert to RGB and normalize
        rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
        normalized = rgb.astype(np.float32) / 255.0
        
        # Transpose to CHW format
        chw = np.transpose(normalized, (2, 0, 1))
        
        # Add batch dimension
        batch = np.expand_dims(chw, axis=0)
        
        return batch, scale, (pad_w, pad_h)
    
    def postprocess(self, 
                    output: np.ndarray,
                    scale: float,
                    pad: Tuple[int, int],
                    orig_shape: Tuple[int, int]) -> List[Detection]:
        """
        后处理输出
        
        Args:
            output: Model output (1, 25200, 85) for YOLOv5s
            scale: Scale factor
            pad: Padding (pad_w, pad_h)
            orig_shape: Original image shape (H, W)
            
        Returns:
            List of Detection objects
        """
        # Remove batch dimension
        output = output[0]  # (25200, 85)
        
        # Filter by confidence
        confidences = output[:, 4]
        mask = confidences >= self.conf_threshold
        output = output[mask]
        
        if len(output) == 0:
            return []
        
        # Extract boxes and class scores
        boxes = output[:, :4]  # (x_center, y_center, w, h)
        class_scores = output[:, 5:]
        
        # Get class predictions
        class_ids = np.argmax(class_scores, axis=1)
        class_confidences = np.max(class_scores, axis=1)
        
        # Final confidence
        final_confidences = confidences[mask] * class_confidences
        
        # Convert to (x1, y1, x2, y2)
        x_center, y_center, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        x1 = x_center - w / 2
        y1 = y_center - h / 2
        x2 = x_center + w / 2
        y2 = y_center + h / 2
        
        # Adjust for padding and scale
        pad_w, pad_h = pad
        x1 = (x1 - pad_w) / scale
        y1 = (y1 - pad_h) / scale
        x2 = (x2 - pad_w) / scale
        y2 = (y2 - pad_h) / scale
        
        # Clip to image bounds
        h_orig, w_orig = orig_shape
        x1 = np.clip(x1, 0, w_orig)
        y1 = np.clip(y1, 0, h_orig)
        x2 = np.clip(x2, 0, w_orig)
        y2 = np.clip(y2, 0, h_orig)
        
        # NMS
        boxes_xyxy = np.stack([x1, y1, x2, y2], axis=1)
        keep_indices = self._nms(boxes_xyxy, final_confidences, self.iou_threshold)
        
        # Build Detection objects
        detections = []
        for idx in keep_indices:
            detection = Detection(
                bbox=(int(x1[idx]), int(y1[idx]), int(x2[idx]), int(y2[idx])),
                confidence=float(final_confidences[idx]),
                class_id=int(class_ids[idx]),
                class_name=self.COCO_CLASSES[class_ids[idx]]
            )
            detections.append(detection)
        
        return detections
    
    def _nms(self, boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> List[int]:
        """Non-Maximum Suppression"""
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        
        areas = (x2 - x1) * (y2 - y1)
        order = scores.argsort()[::-1]
        
        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            
            w = np.maximum(0.0, xx2 - xx1)
            h = np.maximum(0.0, yy2 - yy1)
            inter = w * h
            
            iou = inter / (areas[i] + areas[order[1:]] - inter)
            
            inds = np.where(iou <= iou_threshold)[0]
            order = order[inds + 1]
        
        return keep
    
    def detect(self, image: np.ndarray) -> List[Detection]:
        """
        检测图像中的目标
        
        Args:
            image: BGR image
            
        Returns:
            List of Detection objects
        """
        # Check memory
        if self.memory_monitor and not self.memory_monitor.is_safe_to_run():
            logger.error("Memory threshold exceeded! Skipping detection.")
            return []
        
        # Preprocess
        input_tensor, scale, pad = self.preprocess(image)
        
        # Inference
        outputs = self.engine.infer(input_tensor)
        
        # Postprocess
        detections = self.postprocess(outputs[0], scale, pad, image.shape[:2])
        
        # Update FPS
        self.fps_counter.update()
        
        return detections
    
    def draw_detections(self, image: np.ndarray, detections: List[Detection]) -> np.ndarray:
        """
        在图像上绘制检测结果
        
        Args:
            image: BGR image
            detections: List of Detection objects
            
        Returns:
            Image with drawn detections
        """
        result = image.copy()
        
        for det in detections:
            x1, y1, x2, y2 = det.bbox
            
            # Draw bounding box
            color = self._get_color(det.class_id)
            cv2.rectangle(result, (x1, y1), (x2, y2), color, 2)
            
            # Draw label
            label = f"{det.class_name} {det.confidence:.2f}"
            (label_w, label_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(result, (x1, y1 - label_h - baseline), (x1 + label_w, y1), color, -1)
            cv2.putText(result, label, (x1, y1 - baseline), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Draw FPS
        fps_text = f"FPS: {self.fps_counter.get_fps():.1f}"
        cv2.putText(result, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        
        return result
    
    def _generate_color_palette(self) -> Dict[int, Tuple[int, int, int]]:
        """Pre-generate color palette for all classes"""
        palette = {}
        for class_id in range(len(self.COCO_CLASSES)):
            np.random.seed(class_id)
            palette[class_id] = tuple(map(int, np.random.randint(0, 255, 3)))
        return palette
    
    def _get_color(self, class_id: int) -> Tuple[int, int, int]:
        """Get color for class from pre-computed palette"""
        return self._color_palette.get(class_id, (255, 255, 255))
    
    def cleanup(self):
        """清理资源"""
        if self.memory_monitor:
            self.memory_monitor.stop()
        logger.info("YOLOv5 Agent cleaned up")
    
    def __del__(self):
        self.cleanup()


class FPSCounter:
    """FPS 计数器"""
    
    def __init__(self, window_size: int = 30):
        self.window_size = window_size
        self.timestamps = []
    
    def update(self):
        """更新时间戳"""
        self.timestamps.append(time.time())
        if len(self.timestamps) > self.window_size:
            self.timestamps.pop(0)
    
    def get_fps(self) -> float:
        """获取 FPS"""
        if len(self.timestamps) < 2:
            return 0.0
        return (len(self.timestamps) - 1) / (self.timestamps[-1] - self.timestamps[0])


class GStreamerCamera:
    """
    GStreamer CSI 摄像头接口
    适配 Jetson Nano nvarguscamerasrc
    """
    
    @staticmethod
    def get_csi_pipeline(
            camera_id: int = 0,
            width: int = 1280,
            height: int = 720,
            fps: int = 30,
            flip_method: int = 0) -> str:
        """
        获取 CSI 摄像头 GStreamer 管道字符串
        
        Args:
            camera_id: 摄像头 ID (0 或 1)
            width: 视频宽度
            height: 视频高度
            fps: 帧率
            flip_method: 翻转方法 (0-7)
                0: none
                1: counterclockwise
                2: rotate-180
                3: clockwise
                4: horizontal-flip
                5: upper-right-diagonal
                6: vertical-flip
                7: upper-left-diagonal
            
        Returns:
            GStreamer pipeline string
        """
        pipeline = (
            f"nvarguscamerasrc sensor-id={camera_id} ! "
            f"video/x-raw(memory:NVMM), width=(int){width}, height=(int){height}, "
            f"format=(string)NV12, framerate=(fraction){fps}/1 ! "
            f"nvvidconv flip-method={flip_method} ! "
            f"video/x-raw, width=(int){width}, height=(int){height}, format=(string)BGRx ! "
            f"videoconvert ! "
            f"video/x-raw, format=(string)BGR ! "
            f"appsink"
        )
        return pipeline
    
    @staticmethod
    def open_camera(camera_id: int = 0, width: int = 1280, height: int = 720, fps: int = 30) -> cv2.VideoCapture:
        """
        打开 CSI 摄像头
        
        Returns:
            cv2.VideoCapture object
        """
        pipeline = GStreamerCamera.get_csi_pipeline(camera_id, width, height, fps)
        logger.info(f"Opening CSI camera with pipeline:\n{pipeline}")
        
        cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
        
        if not cap.isOpened():
            raise RuntimeError("Failed to open CSI camera")
        
        logger.info(f"CSI camera opened: {width}x{height} @ {fps} FPS")
        return cap


if __name__ == "__main__":
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description="YOLOv5 Detection Agent for Jetson Nano")
    parser.add_argument("--engine", type=str, required=True, help="Path to TensorRT engine file")
    parser.add_argument("--source", type=str, default="0", help="Video source (camera ID or video file)")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.45, help="NMS IoU threshold")
    parser.add_argument("--display", action="store_true", help="Display output window")
    parser.add_argument("--output", type=str, help="Output video file")
    
    args = parser.parse_args()
    
    # Initialize agent
    agent = YOLOv5Agent(
        engine_path=args.engine,
        conf_threshold=args.conf,
        iou_threshold=args.iou
    )
    
    # Open video source
    if args.source.isdigit():
        # CSI camera
        cap = GStreamerCamera.open_camera(camera_id=int(args.source))
    else:
        # Video file
        cap = cv2.VideoCapture(args.source)
    
    if not cap.isOpened():
        logger.error("Failed to open video source")
        sys.exit(1)
    
    # Output writer
    writer = None
    if args.output:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = cv2.VideoWriter(args.output, fourcc, fps, (width, height))
    
    logger.info("Starting detection loop...")
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Detect
            detections = agent.detect(frame)
            
            # Draw
            result = agent.draw_detections(frame, detections)
            
            # Display
            if args.display:
                cv2.imshow("YOLOv5 Detection", result)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            # Write
            if writer:
                writer.write(result)
            
            # Log detections
            if detections:
                logger.info(f"Detected {len(detections)} objects: {[d.class_name for d in detections]}")
    
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    
    finally:
        cap.release()
        if writer:
            writer.release()
        if args.display:
            cv2.destroyAllWindows()
        agent.cleanup()
        logger.info("Cleanup complete")
