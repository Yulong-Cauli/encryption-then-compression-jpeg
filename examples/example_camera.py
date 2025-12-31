#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Example 3: GStreamer Camera Integration
演示 GStreamer CSI 摄像头集成

支持的摄像头 / Supported Cameras:
- IMX219 (Raspberry Pi Camera v2) - 8MP
- IMX477 (Raspberry Pi HQ Camera) - 12MP
- 其他兼容 nvarguscamerasrc 的 CSI 摄像头 / Other CSI cameras compatible with nvarguscamerasrc
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from yolov5_agent import GStreamerCamera
import cv2


def example_csi_pipeline():
    """CSI 摄像头管道示例"""
    print("=== Example 1: CSI Camera Pipeline ===\n")
    
    # Get pipeline string
    pipeline = GStreamerCamera.get_csi_pipeline(
        camera_id=0,
        width=1280,
        height=720,
        fps=30,
        flip_method=0
    )
    
    print("GStreamer Pipeline for CSI Camera:\n")
    print(pipeline)
    print("\n" + "="*80)
    
    # Show different flip methods
    print("\nAvailable flip methods:")
    flip_methods = {
        0: "none",
        1: "counterclockwise",
        2: "rotate-180",
        3: "clockwise",
        4: "horizontal-flip",
        5: "upper-right-diagonal",
        6: "vertical-flip",
        7: "upper-left-diagonal"
    }
    
    for method, description in flip_methods.items():
        print(f"  {method}: {description}")


def example_open_camera():
    """打开摄像头示例"""
    print("\n=== Example 2: Open CSI Camera ===\n")
    
    try:
        # Try to open CSI camera
        print("Attempting to open CSI camera...")
        cap = GStreamerCamera.open_camera(camera_id=0, width=1280, height=720, fps=30)
        
        print("✓ Camera opened successfully!")
        
        # Read a few frames
        print("\nReading 5 frames...")
        for i in range(5):
            ret, frame = cap.read()
            if ret:
                print(f"  Frame {i+1}: {frame.shape}")
            else:
                print(f"  Frame {i+1}: Failed to read")
        
        # Release
        cap.release()
        print("\n✓ Camera released")
        
    except Exception as e:
        print(f"\n✗ Failed to open CSI camera: {e}")
        print("\nNote: CSI camera only works on Jetson devices with connected CSI camera.")
        print("Trying USB camera as fallback...")
        
        try:
            cap = cv2.VideoCapture(0)
            if cap.isOpened():
                print("✓ USB camera opened successfully!")
                cap.release()
            else:
                print("✗ No camera available")
        except Exception as e2:
            print(f"✗ USB camera also failed: {e2}")


def example_different_resolutions():
    """不同分辨率示例"""
    print("\n=== Example 3: Different Resolutions ===\n")
    
    resolutions = [
        (640, 480, "VGA"),
        (1280, 720, "HD"),
        (1920, 1080, "Full HD"),
    ]
    
    print("Available resolutions for CSI camera:\n")
    for width, height, name in resolutions:
        pipeline = GStreamerCamera.get_csi_pipeline(
            camera_id=0,
            width=width,
            height=height,
            fps=30
        )
        print(f"{name} ({width}x{height}):")
        print(f"  Pipeline: nvarguscamerasrc sensor-id=0 ! video/x-raw(memory:NVMM), "
              f"width=(int){width}, height=(int){height}, format=(string)NV12, "
              f"framerate=(fraction)30/1 ! ...")
        print()


def example_capture_and_save():
    """捕获并保存图像示例"""
    print("\n=== Example 4: Capture and Save Image ===\n")
    
    try:
        print("Opening camera...")
        cap = GStreamerCamera.open_camera(camera_id=0, width=1280, height=720, fps=30)
        
        print("Capturing image...")
        ret, frame = cap.read()
        
        if ret:
            output_path = "camera_capture.jpg"
            cv2.imwrite(output_path, frame)
            print(f"✓ Image saved to: {output_path}")
            print(f"  Image shape: {frame.shape}")
        else:
            print("✗ Failed to capture image")
        
        cap.release()
        
    except Exception as e:
        print(f"✗ Error: {e}")
        print("\nNote: This example requires a CSI camera connected to a Jetson device.")


def example_camera_test_script():
    """生成摄像头测试脚本"""
    print("\n=== Example 5: Camera Test Script ===\n")
    
    script = """#!/bin/bash
# Jetson Nano CSI Camera Test Script

echo "Testing CSI Camera on Jetson Nano..."
echo ""

# Test 1: Check if nvarguscamerasrc is available
echo "Test 1: Checking nvarguscamerasrc..."
gst-inspect-1.0 nvarguscamerasrc > /dev/null 2>&1
if [ $? -eq 0 ]; then
    echo "  ✓ nvarguscamerasrc is available"
else
    echo "  ✗ nvarguscamerasrc not found"
    exit 1
fi

# Test 2: Test camera with gst-launch
echo ""
echo "Test 2: Testing camera with gst-launch..."
echo "  (Press Ctrl+C to stop after you see the video)"
gst-launch-1.0 nvarguscamerasrc sensor-id=0 ! \\
    'video/x-raw(memory:NVMM), width=(int)1280, height=(int)720, format=(string)NV12, framerate=(fraction)30/1' ! \\
    nvvidconv ! \\
    'video/x-raw, format=(string)BGRx' ! \\
    videoconvert ! \\
    'video/x-raw, format=(string)BGR' ! \\
    autovideosink

echo ""
echo "✓ Camera test complete!"
"""
    
    output_path = "test_camera.sh"
    with open(output_path, 'w') as f:
        f.write(script)
    
    # Make executable
    os.chmod(output_path, 0o755)
    
    print(f"Camera test script generated: {output_path}")
    print("\nUsage:")
    print(f"  chmod +x {output_path}")
    print(f"  ./{output_path}")
    print("\n✓ Script created successfully!")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="GStreamer Camera Examples")
    parser.add_argument(
        "--example",
        type=int,
        choices=[1, 2, 3, 4, 5],
        default=1,
        help="Example number: 1=pipeline, 2=open, 3=resolutions, 4=capture, 5=test_script"
    )
    
    args = parser.parse_args()
    
    try:
        if args.example == 1:
            example_csi_pipeline()
        elif args.example == 2:
            example_open_camera()
        elif args.example == 3:
            example_different_resolutions()
        elif args.example == 4:
            example_capture_and_save()
        elif args.example == 5:
            example_camera_test_script()
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
