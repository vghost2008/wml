
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import time
import argparse
import os

class TRTLogger(trt.ILogger):
    """自定义日志记录器，用于捕获TensorRT构建和推理过程中的信息"""
    def __init__(self, severity=trt.Logger.WARNING):
        super().__init__()
        self.severity = severity

    def log(self, severity, msg):
        if severity <= self.severity:
            print(f"[{severity}] {msg}")

def build_engine(onnx_path, engine_path, fp16=False, workspace_size=1 << 30, 
                 min_shape=(1, 3, 224, 224), opt_shape=(4, 3, 224, 224), max_shape=(8, 3, 224, 224)):
    """
    构建TensorRT引擎，等效于trtexec的构建阶段
    :param onnx_path: ONNX模型路径
    :param engine_path: 保存的Engine路径
    :param fp16: 是否启用FP16
    :param workspace_size: 工作空间大小
    :param min_shape: 最小输入形状
    :param opt_shape: 最佳输入形状
    :param max_shape: 最大输入形状
    :return: ICudaEngine对象
    """
    logger = TRTLogger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    
    # 创建网络定义，必须使用EXPLICIT_BATCH以支持动态形状
    network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(network_flags)
    
    # 解析ONNX模型
    parser = trt.OnnxParser(network, logger)
    with open(onnx_path, 'rb') as model_file:
        if not parser.parse(model_file.read()):
            for error in range(parser.num_errors):
                print(f"Parser Error: {parser.get_error(error)}")
            raise ValueError("Failed to parse ONNX model")

    # 配置构建参数
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_size)
    
    if fp16 and builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)
        
    # 配置动态形状优化配置文件 (Optimization Profile)
    profile = builder.create_optimization_profile()
    input_name = network.get_input(0).name
    profile.set_shape(input_name, min_shape, opt_shape, max_shape)
    config.add_optimization_profile(profile)
    
    # 构建并序列化引擎
    print("Building engine... This may take a while.")
    serialized_engine = builder.build_serialized_network(network, config)
    
    if serialized_engine is None:
        raise RuntimeError("Failed to build engine")
        
    # 保存引擎到磁盘
    with open(engine_path, 'wb') as f:
        f.write(serialized_engine)
    print(f"Engine saved to {engine_path}")
    
    return serialized_engine

def load_engine(engine_path):
    """加载序列化的Engine文件"""
    logger = TRTLogger(trt.Logger.WARNING)
    runtime = trt.Runtime(logger)
    with open(engine_path, 'rb') as f:
        serialized_engine = f.read()
    engine = runtime.deserialize_cuda_engine(serialized_engine)
    return engine

def benchmark_inference(engine, input_shape, iterations=100, warmup=10):
    """
    执行推理基准测试，等效于trtexec的推理测速阶段
    :param engine: ICudaEngine对象
    :param input_shape: 输入形状 (B, C, H, W)
    :param iterations: 正式测试迭代次数
    :param warmup: 预热次数
    :return: 平均延迟 (ms), 吞吐量 (FPS)
    """
    context = engine.create_execution_context()
    
    # 设置输入形状
    input_name = engine.get_binding_name(0)
    context.set_binding_shape(0, input_shape)
    
    # 分配内存
    h_input = np.random.random(input_shape).astype(np.float32)
    d_input = cuda.mem_alloc(h_input.nbytes)
    
    # 获取输出绑定信息
    output_bindings = []
    h_outputs = []
    d_outputs = []
    for i in range(1, engine.num_bindings):
        shape = context.get_binding_shape(i)
        dtype = trt.nptype(engine.get_binding_dtype(i))
        size = trt.volume(shape) * np.dtype(dtype).itemsize
        h_output = np.empty(shape, dtype=dtype)
        d_output = cuda.mem_alloc(size)
        h_outputs.append(h_output)
        d_outputs.append(d_output)
        output_bindings.append(int(d_output))
        
    # 创建CUDA流
    stream = cuda.Stream()
    
    # 预热
    for _ in range(warmup):
        cuda.memcpy_htod_async(d_input, h_input, stream)
        context.execute_async_v2(bindings=[int(d_input)] + output_bindings, stream_handle=stream.handle)
        for h_out, d_out in zip(h_outputs, d_outputs):
            cuda.memcpy_dtoh_async(h_out, d_out, stream)
        stream.synchronize()
        
    # 正式测试
    start_time = time.time()
    for _ in range(iterations):
        cuda.memcpy_htod_async(d_input, h_input, stream)
        context.execute_async_v2(bindings=[int(d_input)] + output_bindings, stream_handle=stream.handle)
        for h_out, d_out in zip(h_outputs, d_outputs):
            cuda.memcpy_dtoh_async(h_out, d_out, stream)
        stream.synchronize()
    end_time = time.time()
    
    total_time = end_time - start_time
    avg_latency_ms = (total_time / iterations) * 1000
    throughput = iterations / total_time
    
    return avg_latency_ms, throughput

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
