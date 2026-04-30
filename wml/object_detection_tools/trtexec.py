import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import time
import argparse
import os
from wml.wtorch.trt import build_engine,TRTLogger,load_engine,benchmark_inference

def main():
    parser = argparse.ArgumentParser(description="Python equivalent of trtexec")
    parser.add_argument('--onnx', type=str, required=True, help='Path to ONNX model')
    parser.add_argument('--engine', type=str, default='model.engine', help='Path to save/load TensorRT engine')
    parser.add_argument('--fp16', action='store_true', help='Enable FP16 precision')
    parser.add_argument('--workspace', type=int, default=4096, help='Workspace size in MiB')
    parser.add_argument('--minShapes', type=str, default='1,3,224,224', help='Min input shapes e.g. 1,3,224,224')
    parser.add_argument('--optShapes', type=str, default='1,3,224,224', help='Opt input shapes e.g. 4,3,224,224')
    parser.add_argument('--maxShapes', type=str, default='1,3,224,224', help='Max input shapes e.g. 8,3,224,224')
    parser.add_argument('--iterations', type=int, default=100, help='Number of inference iterations for benchmark')
    parser.add_argument('--warmup', type=int, default=10, help='Number of warmup iterations')
    parser.add_argument('--buildOnly', action='store_true', help='Only build engine, do not benchmark')
    
    args = parser.parse_args()
    
    # 解析形状字符串
    def parse_shape(s):
        return tuple(map(int, s.split(',')))
        
    min_shape = parse_shape(args.minShapes)
    opt_shape = parse_shape(args.optShapes)
    max_shape = parse_shape(args.maxShapes)
    workspace_bytes = args.workspace * (1 << 20)
    
    # 1. 构建或加载引擎
    if not os.path.exists(args.engine):
        print(f"Engine not found, building from {args.onnx}...")
        build_engine(
            onnx_path=args.onnx,
            engine_path=args.engine,
            fp16=args.fp16,
            workspace_size=workspace_bytes,
            min_shape=min_shape,
            opt_shape=opt_shape,
            max_shape=max_shape
        )
    else:
        print(f"Loading existing engine from {args.engine}")
        
    engine = load_engine(args.engine)
    
    if args.buildOnly:
        print("Build only mode. Exiting.")
        return
        
    return
    # 2. 基准测试
    print(f"Running benchmark with batch size {opt_shape}...")
    avg_latency, throughput = benchmark_inference(
        engine=engine,
        input_shape=opt_shape,
        iterations=args.iterations,
        warmup=args.warmup
    )
    
    print("-" * 30)
    print(f"Average Latency: {avg_latency:.4f} ms")
    print(f"Throughput: {throughput:.2f} FPS")
    print("-" * 30)

if __name__ == '__main__':
    main()
