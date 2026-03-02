import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.onnx import _type_utils, errors, symbolic_helper
from torch.onnx._internal import _beartype, jit_utils, registration
import functools
from torch import _C
from torch._C import _onnx as _C_onnx



_onnx_symbolic = functools.partial(registration.onnx_symbolic, opset=17)


@_onnx_symbolic("aten::fft_fft")
@symbolic_helper.parse_args("v", "i", "i", "s")
def fft_fft(
    g: jit_utils.GraphContext,
    input: _C.Value,
    n: int,
    dim: int,
    norm: str,
):
    
    rank = symbolic_helper._get_tensor_rank(input)
    if rank ==4:
        input = symbolic_helper._unsqueeze_helper(g, input, [-1])
        
    output = g.op(
        "DFT",
        input,
        axis_i=dim, # last dim represent complex
        inverse_i=0,
        onesided_i=0
    )
    
    return output


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

@_onnx_symbolic("aten::fft_ifft")
@symbolic_helper.parse_args("v", "v", "i", "s")
def fft_ifft(
    g: jit_utils.GraphContext,
    input: _C.Value,
    n: _C.Value,
    dim: int,
    norm: str,
):
    return g.op(
        "DFT",
        input,
        n,
        axis_i=dim, 
        inverse_i=1,
        onesided_i=0
    )


def irfft(input,s,dim):
    end = (s+1)//2
    
    input2 = torch.index_select(input, dim, torch.flip(torch.arange(1,end),dims=(0,)))
    r0 = torch.real(input)
    i0 = torch.imag(input)
    r1 = torch.real(input2)
    i1 = torch.imag(input2)
    rr = torch.cat([r0,r1],dim=dim)
    ii = torch.cat([i0,-i1],dim=dim)
    input = torch.complex(rr,ii)

    return torch.fft.ifft(input,s,dim)
                      
def rfft(input, dim):
    input = torch.fft.fft(input, dim=dim)
    # 取一半
    input = torch.index_select(input, dim, torch.arange(input.shape[dim]//2+1))
    return input


def rfftn(input, dim):
    new_dim  = []
    for d in dim:
        if d<0:
            d = input.ndim+d
        new_dim.append(d)
    dim = new_dim
    input = rfft(input,dim[0])
    for d in dim[1:]:
        input = torch.fft.fft(input, dim=d)
    return input

def irfftn(input, sl, dim):
    new_dim  = []
    for d in dim:
        if d<0:
            d = input.ndim+d
        new_dim.append(d)
    dim = new_dim

    output = input
    for s, d in zip(sl[1:][::-1], dim[1:][::-1]):
        output= torch.fft.ifft(output, s, d)
    output = irfft(output,sl[0],dim[0])
    return torch.real(output)