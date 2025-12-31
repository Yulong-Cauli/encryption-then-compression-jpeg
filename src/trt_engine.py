#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TensorRT Inference Engine with CUDA Stream for High-Performance Inference
针对 NVIDIA Jetson Nano (Maxwell Architecture) 优化的 TensorRT 推理引擎
"""

import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
from typing import List, Tuple, Optional
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TensorRTInferenceEngine:
    """
    High-performance TensorRT inference engine with CUDA Stream async processing.
    为 Maxwell 架构 GPU 优化的异步推理引擎。
    """
    
    def __init__(self, engine_path: str, max_batch_size: int = 1):
        """
        初始化 TensorRT 引擎
        
        Args:
            engine_path: TensorRT engine 文件路径 (.engine)
            max_batch_size: 最大批处理大小
        """
        self.engine_path = engine_path
        self.max_batch_size = max_batch_size
        
        # TensorRT Logger
        self.trt_logger = trt.Logger(trt.Logger.WARNING)
        
        # Load engine
        self.engine = self._load_engine()
        self.context = self.engine.create_execution_context()
        
        # CUDA Stream for async processing
        self.stream = cuda.Stream()
        
        # Allocate buffers
        self.inputs, self.outputs, self.bindings, self.input_shapes, self.output_shapes = self._allocate_buffers()
        
        logger.info(f"TensorRT Engine loaded: {engine_path}")
        logger.info(f"Input shapes: {self.input_shapes}")
        logger.info(f"Output shapes: {self.output_shapes}")
        
    def _load_engine(self) -> trt.ICudaEngine:
        """从文件加载 TensorRT engine"""
        with open(self.engine_path, 'rb') as f:
            runtime = trt.Runtime(self.trt_logger)
            engine = runtime.deserialize_cuda_engine(f.read())
            if engine is None:
                raise RuntimeError(f"Failed to load TensorRT engine: {self.engine_path}")
            return engine
    
    def _allocate_buffers(self) -> Tuple[List, List, List, List, List]:
        """
        分配 GPU 显存缓冲区
        Returns: (inputs, outputs, bindings, input_shapes, output_shapes)
        """
        inputs = []
        outputs = []
        bindings = []
        input_shapes = []
        output_shapes = []
        
        for binding in self.engine:
            # Get binding dimensions
            shape = self.engine.get_binding_shape(binding)
            # Note: For explicit batch engines, shape already includes batch dimension
            # We multiply by max_batch_size for compatibility with dynamic batching
            size = trt.volume(shape) * self.max_batch_size
            dtype = trt.nptype(self.engine.get_binding_dtype(binding))
            
            # Allocate host and device buffers
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            
            # Append to the appropriate list
            bindings.append(int(device_mem))
            
            if self.engine.binding_is_input(binding):
                inputs.append({'host': host_mem, 'device': device_mem})
                input_shapes.append(shape)
                logger.info(f"Input binding '{binding}': shape={shape}, dtype={dtype}")
            else:
                outputs.append({'host': host_mem, 'device': device_mem})
                output_shapes.append(shape)
                logger.info(f"Output binding '{binding}': shape={shape}, dtype={dtype}")
        
        return inputs, outputs, bindings, input_shapes, output_shapes
    
    def infer(self, input_data: np.ndarray) -> List[np.ndarray]:
        """
        同步推理
        
        Args:
            input_data: 输入数据 numpy array
            
        Returns:
            输出列表
        """
        # Copy input data to host buffer
        np.copyto(self.inputs[0]['host'], input_data.ravel())
        
        # Transfer input data to GPU
        cuda.memcpy_htod_async(self.inputs[0]['device'], self.inputs[0]['host'], self.stream)
        
        # Execute inference
        self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)
        
        # Transfer predictions back from GPU
        for output in self.outputs:
            cuda.memcpy_dtoh_async(output['host'], output['device'], self.stream)
        
        # Synchronize stream
        self.stream.synchronize()
        
        # Return outputs
        return [output['host'].reshape(shape) for output, shape in zip(self.outputs, self.output_shapes)]
    
    def infer_async(self, input_data: np.ndarray) -> None:
        """
        异步推理 (非阻塞)
        
        Args:
            input_data: 输入数据 numpy array
        """
        # Copy input data to host buffer
        np.copyto(self.inputs[0]['host'], input_data.ravel())
        
        # Transfer input data to GPU (async)
        cuda.memcpy_htod_async(self.inputs[0]['device'], self.inputs[0]['host'], self.stream)
        
        # Execute inference (async)
        self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)
        
        # Transfer predictions back from GPU (async)
        for output in self.outputs:
            cuda.memcpy_dtoh_async(output['host'], output['device'], self.stream)
    
    def get_async_results(self) -> List[np.ndarray]:
        """
        获取异步推理结果 (阻塞直到完成)
        
        Returns:
            输出列表
        """
        # Wait for async operations to complete
        self.stream.synchronize()
        
        # Return outputs
        return [output['host'].reshape(shape) for output, shape in zip(self.outputs, self.output_shapes)]
    
    def warmup(self, num_iterations: int = 10) -> None:
        """
        预热引擎，优化性能
        
        Args:
            num_iterations: 预热迭代次数
        """
        logger.info(f"Warming up TensorRT engine ({num_iterations} iterations)...")
        
        # Create dummy input
        dummy_input = np.random.randn(*self.input_shapes[0]).astype(np.float32)
        
        for _ in range(num_iterations):
            self.infer(dummy_input)
        
        logger.info("Warmup complete!")
    
    def __del__(self):
        """清理资源"""
        try:
            # Free CUDA memory
            if hasattr(self, 'inputs'):
                for inp in self.inputs:
                    if 'device' in inp:
                        inp['device'].free()
            if hasattr(self, 'outputs'):
                for out in self.outputs:
                    if 'device' in out:
                        out['device'].free()
            
            logger.info("TensorRT resources freed")
        except Exception as e:
            logger.warning(f"Error during cleanup: {e}")


class TensorRTEngineBuilder:
    """
    TensorRT Engine 构建器 (从 ONNX 转换)
    """
    
    @staticmethod
    def build_engine(onnx_path: str, 
                     engine_path: str,
                     fp16_mode: bool = True,
                     max_batch_size: int = 1,
                     max_workspace_size: int = 1 << 30) -> None:
        """
        从 ONNX 模型构建 TensorRT Engine
        
        Args:
            onnx_path: ONNX 模型路径
            engine_path: 输出 engine 文件路径
            fp16_mode: 是否启用 FP16 精度 (推荐 Jetson Nano)
            max_batch_size: 最大批处理大小
            max_workspace_size: 最大工作空间大小 (bytes)
        """
        logger.info(f"Building TensorRT engine from: {onnx_path}")
        
        # Create builder
        trt_logger = trt.Logger(trt.Logger.WARNING)
        builder = trt.Builder(trt_logger)
        network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        parser = trt.OnnxParser(network, trt_logger)
        
        # Parse ONNX
        with open(onnx_path, 'rb') as model:
            if not parser.parse(model.read()):
                logger.error("Failed to parse ONNX model")
                for error in range(parser.num_errors):
                    logger.error(parser.get_error(error))
                raise RuntimeError("ONNX parsing failed")
        
        # Build config
        config = builder.create_builder_config()
        config.max_workspace_size = max_workspace_size
        
        # Enable FP16 for better performance on Jetson
        if fp16_mode and builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)
            logger.info("FP16 mode enabled")
        
        # Build engine
        logger.info("Building engine (this may take a while)...")
        engine = builder.build_engine(network, config)
        
        if engine is None:
            raise RuntimeError("Failed to build TensorRT engine")
        
        # Serialize engine
        with open(engine_path, 'wb') as f:
            f.write(engine.serialize())
        
        logger.info(f"Engine saved to: {engine_path}")


if __name__ == "__main__":
    # Example usage
    import sys
    
    if len(sys.argv) < 2:
        print("Usage:")
        print("  1. Build engine: python trt_engine.py build <onnx_path> <engine_path>")
        print("  2. Test engine: python trt_engine.py test <engine_path>")
        sys.exit(1)
    
    mode = sys.argv[1]
    
    if mode == "build" and len(sys.argv) == 4:
        onnx_path = sys.argv[2]
        engine_path = sys.argv[3]
        TensorRTEngineBuilder.build_engine(onnx_path, engine_path)
        
    elif mode == "test" and len(sys.argv) == 3:
        engine_path = sys.argv[2]
        engine = TensorRTInferenceEngine(engine_path)
        engine.warmup(num_iterations=10)
        logger.info("Engine test successful!")
        
    else:
        print("Invalid arguments")
        sys.exit(1)
