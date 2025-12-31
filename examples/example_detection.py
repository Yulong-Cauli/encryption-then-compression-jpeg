#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Example 2: YOLOv5 Detection with Memory Monitoring
演示 YOLOv5 检测和内存监控功能
"""

import sys
import os
import cv2
import numpy as np

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from yolov5_agent import YOLOv5Agent, MemoryMonitor


def example_image_detection():
    """图像检测示例"""
    print("=== Example 1: Image Detection ===\n")
    
    engine_path = "yolov5s_fp16.engine"
    image_path = "../assets/input.jpg"
    
    if not os.path.exists(engine_path):
        print(f"Error: Engine file not found: {engine_path}")
        print("Please run: python3 run_detection.py --mode convert --model yolov5s")
        return
    
    # Initialize agent
    print("Initializing YOLOv5 Agent...")
    agent = YOLOv5Agent(
        engine_path=engine_path,
        conf_threshold=0.25,
        iou_threshold=0.45,
        enable_memory_monitor=True
    )
    
    # Load image
    if not os.path.exists(image_path):
        # Create a dummy image if test image doesn't exist
        print(f"Test image not found, creating dummy image...")
        image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
    else:
        print(f"Loading image: {image_path}")
        image = cv2.imread(image_path)
    
    # Detect
    print("\nRunning detection...")
    detections = agent.detect(image)
    
    print(f"\n✓ Detected {len(detections)} objects:")
    for i, det in enumerate(detections):
        print(f"  {i+1}. {det.class_name} ({det.confidence:.2f}) at {det.bbox}")
    
    # Draw results
    result = agent.draw_detections(image, detections)
    
    # Save result
    output_path = "detection_result.jpg"
    cv2.imwrite(output_path, result)
    print(f"\n✓ Result saved to: {output_path}")
    
    # Cleanup
    agent.cleanup()


def example_memory_monitor():
    """内存监控示例"""
    print("\n=== Example 2: Memory Monitoring ===\n")
    
    import time
    import psutil
    
    # Create monitor with 500MB threshold (easier to trigger for demo)
    monitor = MemoryMonitor(threshold_mb=500, check_interval=1.0)
    
    # Show current memory
    mem = psutil.virtual_memory()
    print(f"Current memory status:")
    print(f"  Total: {mem.total / 1024 / 1024:.0f} MB")
    print(f"  Available: {mem.available / 1024 / 1024:.0f} MB")
    print(f"  Used: {mem.percent:.1f}%")
    print(f"\nMonitor threshold: 500 MB")
    
    # Start monitoring
    print("\nStarting memory monitor...")
    monitor.start()
    
    # Simulate work
    print("Simulating work for 5 seconds...")
    for i in range(5):
        time.sleep(1)
        
        # Check status
        if monitor.is_safe_to_run():
            print(f"  [{i+1}/5] Memory OK")
        else:
            print(f"  [{i+1}/5] ⚠️  Memory threshold exceeded!")
            monitor.trigger_cleanup()
            monitor.reset()
    
    # Stop monitoring
    monitor.stop()
    print("\n✓ Monitor test complete")


def example_batch_detection():
    """批量检测示例"""
    print("\n=== Example 3: Batch Detection ===\n")
    
    engine_path = "yolov5s_fp16.engine"
    
    if not os.path.exists(engine_path):
        print(f"Error: Engine file not found: {engine_path}")
        return
    
    # Initialize agent
    print("Initializing YOLOv5 Agent...")
    agent = YOLOv5Agent(
        engine_path=engine_path,
        conf_threshold=0.25,
        iou_threshold=0.45,
        enable_memory_monitor=True
    )
    
    # Create dummy images
    print("\nCreating test images...")
    num_images = 10
    images = [np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8) for _ in range(num_images)]
    
    # Batch detection
    print(f"\nProcessing {num_images} images...")
    total_detections = 0
    
    for i, image in enumerate(images):
        # Check memory before processing
        if not agent.memory_monitor.is_safe_to_run():
            print(f"\n⚠️  Memory threshold exceeded! Stopping at image {i+1}/{num_images}")
            agent.memory_monitor.trigger_cleanup()
            break
        
        detections = agent.detect(image)
        total_detections += len(detections)
        
        if (i + 1) % 2 == 0:
            print(f"  Processed {i+1}/{num_images} images, detected {len(detections)} objects")
    
    print(f"\n✓ Total detections: {total_detections}")
    print(f"✓ Average: {total_detections / (i + 1):.1f} objects per image")
    
    # Cleanup
    agent.cleanup()


def example_fps_counter():
    """FPS 计数器示例"""
    print("\n=== Example 4: FPS Counter ===\n")
    
    from yolov5_agent import FPSCounter
    import time
    
    fps_counter = FPSCounter(window_size=30)
    
    print("Simulating video stream (30 frames)...")
    for i in range(30):
        # Simulate processing time
        time.sleep(0.03)  # ~33 FPS
        
        fps_counter.update()
        
        if (i + 1) % 10 == 0:
            print(f"  Frame {i+1}/30, FPS: {fps_counter.get_fps():.1f}")
    
    print(f"\n✓ Final FPS: {fps_counter.get_fps():.1f}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="YOLOv5 Detection Examples")
    parser.add_argument(
        "--example",
        type=int,
        choices=[1, 2, 3, 4],
        default=1,
        help="Example number: 1=image, 2=memory, 3=batch, 4=fps"
    )
    
    args = parser.parse_args()
    
    try:
        if args.example == 1:
            example_image_detection()
        elif args.example == 2:
            example_memory_monitor()
        elif args.example == 3:
            example_batch_detection()
        elif args.example == 4:
            example_fps_counter()
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
