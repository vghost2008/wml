import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.onnx import _type_utils, errors, symbolic_helper
from torch.onnx._internal import _beartype, jit_utils, registration
import functools
from torch import _C
from torch._C import _onnx as _C_onnx

def fft(x,dim):
    """
    使用基础算子 (MatMul, Sin, Cos) 实现一维离散傅里叶变换 (DFT)
    
    Args:
        x (torch.Tensor): 输入信号，形状为 (..., N)，支持批量处理
        
    Returns:
        torch.Tensor: 复数频谱，形状为 (..., N)
    """
    # 获取序列长度 N
    N = x.shape[dim]
    
    # 创建索引向量 n 和 k
    # n: [0, 1, ..., N-1], 形状 (N,)
    # k: [0, 1, ..., N-1], 形状 (N,)
    n = torch.arange(N, dtype=torch.float32, device=x.device)
    k = torch.arange(N, dtype=torch.float32, device=x.device)
    
    # 计算外积矩阵 kn，形状 (N, N)
    # kn[i, j] = i * j
    kn = torch.outer(k, n)
    
    # 计算旋转因子的角度: theta = 2 * pi * k * n / N
    # 形状 (N, N)
    theta = 2 * math.pi * kn / N
    
    # 计算旋转因子的实部和虚部
    # W_N^{kn} = cos(theta) - j*sin(theta)
    cos_matrix = torch.cos(theta)  # 实部系数
    sin_matrix = torch.sin(theta)  # 虚部系数 (注意负号在公式中)
    
    # 确保输入是复数类型，如果是实数则转换
    if not x.is_complex():
        x_real = x
        x_imag = torch.zeros_like(x)
    else:
        x_complex = x
        x_real = x_complex.real
        x_imag = x_complex.imag
    
    # 执行矩阵乘法来实现求和 sum_{n}
    # 输出实部: Re(X[k]) = sum( x_real[n]*cos - x_imag[n]*(-sin) ) 
    #          = sum( x_real[n]*cos + x_imag[n]*sin )
    # 输出虚部: Im(X[k]) = sum( x_real[n]*(-sin) + x_imag[n]*cos )
    #          = sum( -x_real[n]*sin + x_imag[n]*cos )
    
    # 使用 matmul 进行批量矩阵乘法
    # x_real: (..., N), cos_matrix: (N, N) -> result: (..., N)
    if dim != -1 and dim !=len(x_real.shape)-1:
        x_real = torch.transpose(x_real,-1,dim)
        x_imag = torch.transpose(x_imag,-1,dim)
    real_part = torch.matmul(x_real, cos_matrix) + torch.matmul(x_imag, sin_matrix)
    imag_part = -torch.matmul(x_real, sin_matrix) + torch.matmul(x_imag, cos_matrix)
    if dim != -1 and dim !=len(x_real.shape)-1:
        real_part = torch.transpose(real_part,-1,dim)
        imag_part = torch.transpose(imag_part,-1,dim)
    
    # 组合成复数张量
    X = torch.complex(real_part, imag_part)
    print("wfft")
    return X

def ifft(X,s=None,dim=-1):
    """
    使用基础算子实现一维离散傅里叶逆变换 (IDFT)
    
    Args:
        X (torch.Tensor): 频域复数信号，形状为 (..., N)
        
    Returns:
        torch.Tensor: 时域复数信号，形状为 (..., N)
    """
    N = X.shape[dim]
    
    # 创建索引向量
    n = torch.arange(N, dtype=torch.float32, device=X.device)
    k = torch.arange(N, dtype=torch.float32, device=X.device)
    
    # 计算外积
    kn = torch.outer(k, n)
    
    # 计算角度: theta = 2 * pi * k * n / N
    # 注意：逆变换指数为正，所以 sin 前面没有负号变化，但在欧拉展开时 e^{j\theta} = cos + j*sin
    theta = 2 * math.pi * kn / N
    
    
    # 提取频域信号的实部和虚部
    x_real = X.real
    x_imag = X.imag
    cos_matrix = torch.cos(theta).to(x_real.dtype)
    sin_matrix = torch.sin(theta).to(x_real.dtype)
    
    # 逆变换公式: x[n] = (1/N) * sum( X[k] * (cos + j*sin) )
    # 实部: sum( X_real*cos - X_imag*sin )
    # 虚部: sum( X_real*sin + X_imag*cos )
    if dim != -1 and dim !=len(x_real.shape)-1:
        x_real = torch.transpose(x_real,-1,dim)
        x_imag = torch.transpose(x_imag,-1,dim)
    
    real_part = torch.matmul(x_real, cos_matrix) - torch.matmul(x_imag, sin_matrix)
    imag_part = torch.matmul(x_real, sin_matrix) + torch.matmul(x_imag, cos_matrix)
    if dim != -1 and dim !=len(x_real.shape)-1:
        real_part = torch.transpose(real_part,-1,dim)
        imag_part = torch.transpose(imag_part,-1,dim)
    
    # 组合并归一化 (除以 N)
    x = torch.complex(real_part, imag_part) / N
    
    return x

_onnx_symbolic = functools.partial(registration.onnx_symbolic, opset=11)



@_onnx_symbolic("aten::real")
@symbolic_helper.parse_args("v")
def real(
    g: jit_utils.GraphContext,
    input: _C.Value,
):
    index = g.op("Constant", value_t=torch.tensor(0))
    output = symbolic_helper._select_helper(g, input, -1, index)
    return symbolic_helper._squeeze_helper(g, output, [-1])
   
    
@_onnx_symbolic("aten::imag")
@symbolic_helper.parse_args("v")
def imag(
    g: jit_utils.GraphContext,
    input: _C.Value,
):
    index = g.op("Constant", value_t=torch.tensor(1))
    output = symbolic_helper._select_helper(g, input, -1, index)
    return symbolic_helper._squeeze_helper(g, output, [-1])


@_onnx_symbolic("aten::complex")
@symbolic_helper.parse_args("v", "v")
def complex(
    g: jit_utils.GraphContext,
    input1: _C.Value,
    input2: _C.Value,
):
    input1 = symbolic_helper._unsqueeze_helper(g, input1, [-1])
    input2 = symbolic_helper._unsqueeze_helper(g, input2, [-1])
    
    output = g.op("Concat", input1, input2, axis_i=-1)
    return output


def irfft(input,s,dim):
    end = (s+1)//2
    device = input.device 
    input2 = torch.index_select(input, dim, torch.flip(torch.arange(1,end),dims=(0,)).to(device))
    r0 = torch.real(input)
    i0 = torch.imag(input)
    r1 = torch.real(input2)
    i1 = torch.imag(input2)
    rr = torch.cat([r0,r1],dim=dim)
    ii = torch.cat([i0,-i1],dim=dim)
    input = torch.complex(rr,ii)

    return ifft(input,dim)
                      
def rfft(input, dim):
    input = input.float()
    input = fft(input, dim=dim)
    device = input.device
    # 取一半
    input = torch.index_select(input, dim, torch.arange(input.shape[dim]//2+1).to(device))
    return input

'''
input: [B,C,H,W]
'''
def rfftn(input, dim):
    input = input.float()
    new_dim  = []
    for d in dim:
        if d<0:
            d = input.ndim+d
        new_dim.append(d)
    dim = new_dim
    input = rfft(input,dim[-1])
    for d in dim[::-1][1:]:
        input = fft(input, dim=d)
    return input

'''
input: complex,[B,C,H,W]
'''
def irfftn(input, s, dim):
    new_dim  = []
    for d in dim:
        if d<0:
            d = input.ndim+d
        new_dim.append(d)
    dim = new_dim

    output = input
    for _s, d in zip(s[:-1], dim[:-1]):
        output= ifft(output, _s, d)
    output = irfft(output,s[-1],dim[-1])
    return torch.real(output)