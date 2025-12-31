#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Example 1: Basic TensorRT Inference
演示基础 TensorRT 推理功能
"""

import sys
import os
import numpy as np

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from trt_engine import TensorRTInferenceEngine


def example_basic_inference():
    """基础推理示例"""
    print("=== Example 1: Basic TensorRT Inference ===\n")
    
    # 1. Load engine
    engine_path = "yolov5s_fp16.engine"
    
    if not os.path.exists(engine_path):
        print(f"Error: Engine file not found: {engine_path}")
        print("Please run: python3 run_detection.py --mode convert --model yolov5s")
        return
    
    print(f"Loading engine: {engine_path}")
    engine = TensorRTInferenceEngine(engine_path, max_batch_size=1)
    
    # 2. Warmup
    print("\nWarming up engine...")
    engine.warmup(num_iterations=10)
    
    # 3. Create dummy input
    print("\nCreating dummy input...")
    input_shape = engine.input_shapes[0]
    dummy_input = np.random.randn(*input_shape).astype(np.float32)
    print(f"Input shape: {input_shape}")
    
    # 4. Inference
    print("\nRunning inference...")
    outputs = engine.infer(dummy_input)
    
    print(f"\nOutput shapes:")
    for i, output in enumerate(outputs):
        print(f"  Output {i}: {output.shape}")
    
    print("\n✓ Inference successful!")
    

def example_async_inference():
    """异步推理示例"""
    print("\n=== Example 2: Async Inference ===\n")
    
    engine_path = "yolov5s_fp16.engine"
    
    if not os.path.exists(engine_path):
        print(f"Error: Engine file not found: {engine_path}")
        return
    
    print(f"Loading engine: {engine_path}")
    engine = TensorRTInferenceEngine(engine_path, max_batch_size=1)
    
    # Warmup
    engine.warmup(num_iterations=5)
    
    # Create dummy input
    input_shape = engine.input_shapes[0]
    dummy_input = np.random.randn(*input_shape).astype(np.float32)
    
    # Async inference
    print("\nStarting async inference...")
    engine.infer_async(dummy_input)
    
    # Do other work here...
    print("Doing other work while inference runs...")
    
    # Get results
    print("Waiting for results...")
    outputs = engine.get_async_results()
    
    print(f"\nAsync inference complete!")
    print(f"Output shape: {outputs[0].shape}")


def example_benchmark():
    """性能测试示例"""
    print("\n=== Example 3: Performance Benchmark ===\n")
    
    import time
    
    engine_path = "yolov5s_fp16.engine"
    
    if not os.path.exists(engine_path):
        print(f"Error: Engine file not found: {engine_path}")
        return
    
    print(f"Loading engine: {engine_path}")
    engine = TensorRTInferenceEngine(engine_path, max_batch_size=1)
    
    # Warmup
    print("Warming up...")
    engine.warmup(num_iterations=10)
    
    # Benchmark
    print("\nRunning benchmark (100 iterations)...")
    input_shape = engine.input_shapes[0]
    dummy_input = np.random.randn(*input_shape).astype(np.float32)
    
    num_iterations = 100
    start_time = time.time()
    
    for i in range(num_iterations):
        outputs = engine.infer(dummy_input)
        if (i + 1) % 20 == 0:
            print(f"  Progress: {i + 1}/{num_iterations}")
    
    end_time = time.time()
    elapsed = end_time - start_time
    fps = num_iterations / elapsed
    latency = (elapsed / num_iterations) * 1000  # ms
    
    print(f"\n=== Benchmark Results ===")
    print(f"Total time: {elapsed:.2f} seconds")
    print(f"Average FPS: {fps:.2f}")
    print(f"Average latency: {latency:.2f} ms")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="TensorRT Inference Examples")
    parser.add_argument(
        "--example",
        type=int,
        choices=[1, 2, 3],
        default=1,
        help="Example number: 1=basic, 2=async, 3=benchmark"
    )
    
    args = parser.parse_args()
    
    if args.example == 1:
        example_basic_inference()
    elif args.example == 2:
        example_async_inference()
    elif args.example == 3:
        example_benchmark()
