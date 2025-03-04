import torch
import numpy as np
from collections.abc import Iterable
import torch.nn.functional as F
import random
import sys
from functools import wraps
from collections.abc import Mapping, Sequence
import wml.wml_utils as wmlu
import wml.img_utils as wmli
import cv2
from wml.thirdparty.config import CfgNode 
from wml.thirdparty.pyconfig.config import ConfigDict
from wml.wstructures import WPolygonMasks,WBitmapMasks, WMCKeypoints
from wml.semantic.basic_toolkit import *
from itertools import repeat
import collections.abc
import math
import onnx
import pickle
import types
from contextlib import contextmanager
import gc
import time
import torch.nn as nn
import colorama
from .parallel import DataContainer as DC


try:
    import thop
except ImportError:
    thop = None

def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable) and not isinstance(x, str):
            return tuple(x)
        return tuple(repeat(x, n))
    return parse


to_1tuple = _ntuple(1)
to_2tuple = _ntuple(2)
to_3tuple = _ntuple(3)
to_4tuple = _ntuple(4)
to_ntuple = _ntuple

def unnormalize(x:torch.Tensor,mean=[0.0,0.0,0.0],std=[1.0,1.0,1.0]):
    if len(x.size())==4:
        C = x.shape[1]
        scale = np.reshape(np.array(std,dtype=np.float32),[1,C,1,1])
        offset = np.reshape(np.array(mean,dtype=np.float32),[1,C,1,1])
    elif len(x.size())==5:
        C = x.shape[2]
        scale = np.reshape(np.array(std, dtype=np.float32), [1, 1,C, 1, 1])
        offset = np.reshape(np.array(mean, dtype=np.float32), [1,1, C, 1, 1])
    elif len(x.size())==3:
        C = x.shape[0]
        scale = np.reshape(np.array(std, dtype=np.float32), [C, 1, 1])
        offset = np.reshape(np.array(mean, dtype=np.float32), [C, 1, 1])

    offset = torch.from_numpy(offset).to(x.device)
    scale = torch.from_numpy(scale).to(x.device)
    x = x*scale+offset
    return x

def normalize(x:torch.Tensor,mean=[0.0,0.0,0.0],std=[1.0,1.0,1.0]):
    channel = len(mean)
    if len(x.size())==4:
        scale = np.reshape(np.array(std,dtype=np.float32),[1,channel,1,1])
        offset = np.reshape(np.array(mean,dtype=np.float32),[1,channel,1,1])
    elif len(x.size())==5:
        scale = np.reshape(np.array(std, dtype=np.float32), [1, 1,channel, 1, 1])
        offset = np.reshape(np.array(mean, dtype=np.float32), [1,1, channel, 1, 1])
    elif len(x.size())==3:
        scale = np.reshape(np.array(std, dtype=np.float32), [channel, 1, 1])
        offset = np.reshape(np.array(mean, dtype=np.float32), [channel, 1, 1])

    offset = torch.from_numpy(offset).to(x.device)
    scale = torch.from_numpy(scale).to(x.device)
    x = (x-offset)/scale
    return x

def npnormalize(x:np.ndarray,mean=[0.0,0.0,0.0],std=[1.0,1.0,1.0]):
    if len(x.shape)==4:
        scale = np.reshape(np.array(std,dtype=np.float32),[1,3,1,1])
        offset = np.reshape(np.array(mean,dtype=np.float32),[1,3,1,1])
    elif len(x.shape)==5:
        scale = np.reshape(np.array(std, dtype=np.float32), [1, 1,3, 1, 1])
        offset = np.reshape(np.array(mean, dtype=np.float32), [1,1, 3, 1, 1])
    elif len(x.shape)==3:
        scale = np.reshape(np.array(std, dtype=np.float32), [3, 1, 1])
        offset = np.reshape(np.array(mean, dtype=np.float32), [3, 1, 1])

    x = (x.astype(np.float32)-offset)/scale

    return x

def rgb2gray(img):
    '''
    img: [B,3,H,W]/[3,H,W] (R,G,B) order
    '''
    if len(img.shape)==3:
        s = np.reshape(np.array([0.299, 0.587, 0.114], dtype=np.float32),[3,1,1])
        s = img.new_tensor(s)
        img = img*s
        img = torch.sum(img,dim=0,keepdim=True)
    else:
        s = np.reshape(np.array([0.299, 0.587, 0.114], dtype=np.float32),[1,3,1,1])
        s = img.new_tensor(s)
        img = img*s
        img = torch.sum(img,dim=1,keepdim=True)
    
    return img

def remove_prefix_from_state_dict(state_dict,prefix="module."):
    res = {}
    for k,v in state_dict.items():
        if k.startswith(prefix):
            k = k[len(prefix):]
        res[k] = v
    return res

def forgiving_state_restore(net, loaded_dict,verbose=False,strict=False):
    """
    Handle partial loading when some tensors don't match up in size.
    Because we want to use models that were trained off a different
    number of classes.
    """
    colorama.init(autoreset=True)
    ignore_key = ['num_batches_tracked']
    def _is_ignore_key(k):
        for v in ignore_key:
            if v in k:
                return True
            return False

    if 'state_dict' in loaded_dict:
        loaded_dict = loaded_dict['state_dict']
    if hasattr(net,'module'):
        net = net.module
    net_state_dict = net.state_dict()
    new_loaded_dict = {}
    used_loaded_dict_key = []
    unloaded_net_state_key = [] #模型中没有加载的参数
    unused_ckpt_key = [] #ckpt中没有使用的参数
    for k in net_state_dict:
        new_k = k
        if new_k in loaded_dict and net_state_dict[k].size() == loaded_dict[new_k].size():
            new_loaded_dict[k] = loaded_dict[new_k]
        elif (not k.startswith('module.')) and 'module.'+k in loaded_dict and net_state_dict[k].size() == loaded_dict['module.'+new_k].size():
            new_loaded_dict[k] = loaded_dict['module.'+new_k]
            used_loaded_dict_key.append('module.'+new_k)
        elif 'BN' in k and new_k.replace("BN","bn") in loaded_dict:
            new_k = new_k.replace("BN","bn")
            if net_state_dict[k].size() == loaded_dict[new_k].size():
                new_loaded_dict[k] = loaded_dict[new_k]
                used_loaded_dict_key.append(new_k)
        elif 'GN' in k and new_k.replace("GN","gn") in loaded_dict:
            new_k = new_k.replace("GN","gn")
            if net_state_dict[k].size() == loaded_dict[new_k].size():
                new_loaded_dict[k] = loaded_dict[new_k]
                used_loaded_dict_key.append(new_k)
        elif ".num_batches_tracked" not in k:
            print(f"Skipped loading parameter {k} {net_state_dict[k].shape}")
            unloaded_net_state_key.append(k)

    print(f"---------------------------------------------------")
    for k in loaded_dict:
        if k not in new_loaded_dict and k not in used_loaded_dict_key and not _is_ignore_key(k):
            if k in net_state_dict:
                print(colorama.Fore.YELLOW+f"Skip {k} in loaded dict, shape={loaded_dict[k].shape} vs {net_state_dict[k].shape} in model")
            else:
                print(colorama.Fore.BLUE+f"Skip {k} in loaded dict, shape={loaded_dict[k].shape}")
            unused_ckpt_key.append(k)
    if verbose:
        print(f"---------------------------------------------------")
        for k in new_loaded_dict:
            print(f"Load {k}, shape={new_loaded_dict[k].shape}")
    net_state_dict.update(new_loaded_dict)
    net.load_state_dict(net_state_dict)
    sys.stdout.flush()
    print(f"Load checkpoint finish.")
    if strict and (len(unused_ckpt_key)>0 or len(unloaded_net_state_key)>0):
        raise RuntimeError(f"Load ckpt faild.")
    print(colorama.Style.RESET_ALL)
    return net,list(new_loaded_dict.keys()),unloaded_net_state_key

def load_checkpoint(
                module,
                checkpoint,
                map_location=None,
                strict=False):
    state_dict = torch.load(checkpoint,map_location=map_location)
    return forgiving_state_restore(module,state_dict,strict=strict)

def sequence_mask(lengths,maxlen=None,dtype=torch.bool):
    if not isinstance(lengths,torch.Tensor):
        lengths = torch.from_numpy(np.array(lengths))
    if maxlen is None:
        maxlen = lengths.max()
    if len(lengths.shape)==1:
        lengths = torch.unsqueeze(lengths,axis=-1)
    matrix = torch.arange(maxlen,dtype=lengths.dtype)[None,:]
    mask = matrix<lengths
    return mask


class TraceAmpWrape(torch.nn.Module):
    def __init__(self, model) -> None:
        super().__init__()
        self.model = model

    def forward(self, x):
        with torch.no_grad():
            with torch.cuda.amp.autocast():
                return self.model(x)

def get_tensor_info(tensor):
    tensor = tensor.detach().cpu().to(torch.float32)
    return torch.mean(tensor).item(),torch.min(tensor).item(),torch.max(tensor).item(),torch.std(tensor).item()

def merge_imgs_heatmap(imgs,heat_map,scale=1.0,alpha=0.4,channel=None,min=None,max=None):
    if not isinstance(heat_map,torch.Tensor):
        heat_map = torch.from_numpy(heat_map)
    if not isinstance(imgs,torch.Tensor):
        imgs = torch.from_numpy(imgs)
    if min is None:
        min = torch.min(heat_map)
    else:
        heat_map = torch.maximum(heat_map,torch.Tensor([min]))

    if max is None:
        max = torch.max(heat_map)
    else:
        heat_map = torch.minimum(heat_map,torch.Tensor([max]))
    heat_map = (heat_map-min)*scale/(max-min+1e-8)
    if channel is not None and heat_map.shape[channel]==1:
        t_zeros = torch.zeros_like(heat_map)
        heat_map = torch.cat([heat_map,t_zeros,t_zeros],dim=channel)
    new_imgs = imgs*(1-alpha)+heat_map*alpha
    mask = heat_map>(scale*0.01)
    #imgs = torch.where(mask,new_imgs,imgs)
    imgs = new_imgs
    return imgs

def module_parameters_numel(net,only_training=False):
    total = 0
    for param in net.parameters():
        if only_training and param.requires_grad or not only_training:
            total += torch.numel(param)
    return total


def concat_datas(datas,dim=0):
    if isinstance(datas[0], Mapping):
        new_data = {}
        for k,v in datas[0].items():
            new_data[k] = [v]
        for data in datas[1:]:
            for k,v in data.items():
                new_data[k].append(v)
        keys = list(new_data.keys())
        for k in keys:
            new_data[k] = concat_datas(new_data[k],dim=dim)
        return new_data

    if torch.is_tensor(datas[0]):
        return torch.cat(datas,dim=dim)
    elif isinstance(datas[0],DC):
        return concat_dc_datas(datas,dim)
    elif isinstance(datas[0],Iterable):
        res = []
        try:
            for x in zip(*datas):
                if torch.is_tensor(x[0]):
                    res.append(torch.cat(x,dim=dim))
                else:
                    res.append(concat_datas(x))
        except Exception as e:
            print(e)
            for i,x in enumerate(datas):
                print(i,type(x),x)
            print(f"--------------------------")
            for i,x in enumerate(datas):
                print(i,type(x))
            sys.stdout.flush()
            raise e
        return res
    else:
        return torch.cat(datas,dim=dim)

def concat_dc_datas(datas,cat_dim=0):
    if isinstance(datas[0], DC):
        stacked = []
        if datas[0].cpu_only:
            for i in range(0, len(datas)):
                for sample in datas[i].data:
                    stacked.extend(sample)
            return DC(
                [stacked], datas[0].stack, datas[0].padding_value, cpu_only=True)
        elif datas[0].stack:
            batch = []
            for d in datas:
                batch.extend(d.data)
            pad_dims = datas[0].pad_dims
            padding_value =datas[0].padding_value
            max_shape = [0 for _ in range(pad_dims)]
            for sample in batch:
                for dim in range(1, pad_dims + 1):
                    max_shape[dim - 1] = max(max_shape[dim - 1],
                                             sample.size(-dim))

            for i in range(0, len(batch)):
                assert isinstance(batch[i], torch.Tensor)

                if pad_dims is not None:
                    pad = [0 for _ in range(pad_dims * 2)]
                    sample = batch[i]
                    for dim in range(1, pad_dims + 1):
                        pad[2 * dim - 1] = max_shape[dim - 1] - sample.size(-dim)
                    stacked.append(
                            F.pad(sample, pad, value=padding_value))
                elif pad_dims is None:
                    stacked.append(batch)
                else:
                    raise ValueError(
                        'pad_dims should be either None or integers (1-3)')
            stacked = torch.cat(stacked,dim=cat_dim)
            return DC([stacked], datas[0].stack, datas[0].padding_value)
        else:
            for i in range(0, len(datas)):
                for sample in datas[i].data:
                    stacked.extend(sample)
            return DC([stacked], datas[0].stack, datas[0].padding_value)
    else:
        raise RuntimeError(f"ERROR concat dc type {type(datas[0])}")


def get_model(model):
    if hasattr(model, "module"):
        model = model.module
    return model

TORCH_VERSION = tuple(int(x) for x in torch.__version__.split(".")[:2])

'''
fea:[B,C,H,W]
size:(w,h)
'''
CENTER_PAD = 0
RANDOM_PAD = 1
TOPLEFT_PAD = 2
def pad_feature(fea, size, pad_value=0, pad_type=TOPLEFT_PAD, return_pad_value=False):
    '''
    pad_type: 0, center pad
    pad_type: 1, random pad
    pad_type: 2, topleft_pad
    '''
    w = fea.shape[-1]
    h = fea.shape[-2]
    if pad_type == 0:
        if h < size[1]:
            py0 = (size[1] - h) // 2
            py1 = size[1] - h - py0
        else:
            py0 = 0
            py1 = 0
        if w < size[0]:
            px0 = (size[0] - w) // 2
            px1 = size[0] - w - px0
        else:
            px0 = 0
            px1 = 0
    elif pad_type == 1:
        if h < size[1]:
            py0 = random.randint(0, size[1] - h)
            py1 = size[1] - h - py0
        else:
            py0 = 0
            py1 = 0
        if w < size[0]:
            px0 = random.randint(0, size[0] - w)
            px1 = size[0] - w - px0
        else:
            px0 = 0
            px1 = 0
    elif pad_type == 2:
        if h < size[1]:
            py0 = 0
            py1 = size[1] - h - py0
        else:
            py0 = 0
            py1 = 0
        if w < size[0]:
            px0 = 0
            px1 = size[0] - w - px0
        else:
            px0 = 0
            px1 = 0

    if isinstance(pad_value,Iterable):
        pad_value = pad_value[0]
    fea = F.pad(fea, [px0, px1,py0,py1], "constant", pad_value)

    if return_pad_value:
        return fea, px0, px1, py0, py1
    return fea

def split_forward_batch32(func):
    @wraps(func)
    def wrapper(self, data):
        step = 32
        res = []
        cur_idx = 0
        while cur_idx<data.shape[0]:
            ret_val = func(self, data[cur_idx:cur_idx+step])
            cur_idx += step
            res.append(ret_val)
        if len(res)==1:
            return res[0]
        if torch.is_tensor(res[0]):
            return torch.cat(res,dim=0)
        else:
            return np.concatenate(res,axis=0)
    return wrapper

def to(data,device=torch.device("cpu")):
    if torch.is_tensor(data):
        return data.to(device)
    elif isinstance(data,(CfgNode,ConfigDict)):
        return data
    elif isinstance(data,dict):
        keys = list(data.keys())
        new_data = {}
        for k in keys:
            new_data[k] = to(data[k],device)
    elif isinstance(data,(list,tuple)):
        new_data = []
        for v in data:
            new_data.append(to(v,device))
        new_data = type(data)(new_data)
    elif not isinstance(data,Iterable):
        if not isinstance(data,torch.nn.Module) and hasattr(data,"to"):
            data = data.to(device)
        return data
    elif isinstance(data,(np.ndarray,str,bytes)):
        return data
    else:
        print(f"Unsupport type {type(data)}")

    return new_data

def cpu(data):
    return to(data,device=torch.device("cpu"))

def cuda(data):
    return to(data,device=torch.device("cuda"))

def cpu_wraps(func):
    @wraps(func)
    def wraps_func(*args,**kwargs):
        args = cpu(args)
        kwargs = cpu(kwargs)
        res = func(*args,**kwargs)
        res = cuda(res)
        return res
    return wraps_func

def cpu_cpu_wraps(func):
    @wraps(func)
    def wraps_func(*args,**kwargs):
        args = cpu(args)
        kwargs = cpu(kwargs)
        res = func(*args,**kwargs)
        return res
    return wraps_func

def numpy(data):
    if torch.is_tensor(data):
        return data.cpu().numpy()
    elif isinstance(data,dict):
        keys = list(data.keys())
        new_data = {}
        for k in keys:
            new_data[k] = numpy(data[k])
    elif isinstance(data,(list,tuple)):
        new_data = []
        for v in data:
            new_data.append(numpy(v))
        new_data = type(data)(new_data)
    elif not isinstance(data,Iterable):
        return data
    elif isinstance(data,np.ndarray):
        return data
    else:
        print(f"Unsupport type {type(data)}")

    return new_data

def sparse_gather(data,index,return_tensor=True):
    '''
    data: list of tensor (mybe different length)
    '''
    res = []
    for i,d in enumerate(data):
        res.append(d[index[i]])
    if return_tensor:
        return torch.stack(res,dim=0)
    else:
        return res
    
def simple_model_device(model):
     return next(model.parameters()).device

def simple_model_dtype(model):
     return next(model.parameters()).dtype

def resize_mask(mask,size=None,r=None):
    '''
    mask: [N,H,W]
    size: (new_w,new_h)
    '''
    if size is None:
        size = (int(mask.shape[2]*r),int(mask.shape[1]*r))
    if mask.numel()==0:
        return mask.new_zeros([mask.shape[0],size[1],size[0]])

    mask = torch.unsqueeze(mask,dim=0)
    mask =  torch.nn.functional.interpolate(mask,size=(size[1],size[0]),mode='nearest')
    mask = torch.squeeze(mask,dim=0)
    return mask

def npresize_mask(mask,size=None,r=None):
    '''
    mask: [N,H,W]
    size: (new_w,new_h)
    '''
    if mask.shape[0]==0:
        return np.zeros([0,size[1],size[0]],dtype=mask.dtype)
    if mask.shape[0]==1:
        cur_m = cv2.resize(mask[0],dsize=(size[0],size[1]),interpolation=cv2.INTER_NEAREST)
        return np.expand_dims(cur_m,axis=0)
    mask = resize_mask(torch.from_numpy(mask),size,r)
    return mask.numpy()



def __correct_bboxes(bboxes,h,w):
    old_type = bboxes.dtype
    bboxes = np.maximum(bboxes,0)
    bboxes = np.minimum(bboxes,np.array([[w,h,w,h]]))
    return bboxes.astype(old_type)

def npresize_mask_in_bboxes(mask,bboxes,size=None,r=None):
    '''
    mask: [N,H,W]
    bboxes: [N,4](x0,y0,x1,y1)
    size: (new_w,new_h)
    '''
    if isinstance(mask,(WPolygonMasks,WBitmapMasks,WMCKeypoints)):
        return mask.resize_mask_in_bboxes(bboxes,size=size,r=r)
    if mask.shape[0]==0:
        return np.zeros([0,size[1],size[0]],dtype=mask.dtype),np.zeros([0,4],dtype=bboxes.dtype)
    x_scale = size[0]/mask.shape[2]
    y_scale = size[1]/mask.shape[1]
    bboxes = __correct_bboxes(bboxes,h=mask.shape[1],w=mask.shape[2])
    resized_bboxes = (bboxes*np.array([[x_scale,y_scale,x_scale,y_scale]])).astype(np.int32)
    resized_bboxes = __correct_bboxes(resized_bboxes,h=size[1],w=size[0])
    bboxes = np.array(bboxes).astype(np.int32)
    res_mask = np.zeros([mask.shape[0],size[1],size[0]],dtype=mask.dtype)
    for i in range(mask.shape[0]):
        dbbox = resized_bboxes[i]
        dsize = (dbbox[2]-dbbox[0],dbbox[3]-dbbox[1])
        if dsize[0]<=1 or dsize[1]<=1:
            continue
        sub_mask = wmli.crop_img_absolute_xy(mask[i],bboxes[i])
        cur_m = cv2.resize(sub_mask,dsize=dsize,interpolation=cv2.INTER_NEAREST)
        wmli.set_subimg(res_mask[i],cur_m,dbbox[:2])
    return res_mask,resized_bboxes

def __time_npresize_mask_in_bboxes(mask,bboxes,size=None,r=None):
    t = wmlu.TimeThis()
    b = npresize_mask(mask,size,r)
    t0 = t.time(reset=True)
    a = npresize_mask_in_bboxes(mask,bboxes,size,r)
    t1 = t.time(reset=True)
    c = __npresize_mask(mask,size,r)
    t2 = t.time(reset=True)
    print(f"RM,{t0},{t1},{t2}")
    return a

def clone_tensors(x):
    if isinstance(x,(list,tuple)):
        return [v.clone() for v in x]
    return x.clone()

def _trunc_normal_(tensor, mean, std, a, b):
    # Cut & paste from PyTorch official master until it's in a few official releases - RW
    # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        print("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.")

    # Values are generated by using a truncated uniform distribution and
    # then using the inverse CDF for the normal distribution.
    # Get upper and lower cdf values
    l = norm_cdf((a - mean) / std)
    u = norm_cdf((b - mean) / std)

    # Uniformly fill tensor with values from [l, u], then translate to
    # [2l-1, 2u-1].
    tensor.uniform_(2 * l - 1, 2 * u - 1)

    # Use inverse cdf transform for normal distribution to get truncated
    # standard normal
    tensor.erfinv_()

    # Transform to proper mean, std
    tensor.mul_(std * math.sqrt(2.))
    tensor.add_(mean)

    # Clamp to ensure it's in the proper range
    tensor.clamp_(min=a, max=b)
    return tensor

def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    # type: (Tensor, float, float, float, float) -> Tensor
    r"""Fills the input Tensor with values drawn from a truncated
    normal distribution. The values are effectively drawn from the
    normal distribution :math:`\mathcal{N}(\text{mean}, \text{std}^2)`
    with values outside :math:`[a, b]` redrawn until they are within
    the bounds. The method used for generating the random values works
    best when :math:`a \leq \text{mean} \leq b`.

    NOTE: this impl is similar to the PyTorch trunc_normal_, the bounds [a, b] are
    applied while sampling the normal with mean/std applied, therefore a, b args
    should be adjusted to match the range of mean, std args.

    Args:
        tensor: an n-dimensional `torch.Tensor`
        mean: the mean of the normal distribution
        std: the standard deviation of the normal distribution
        a: the minimum cutoff value
        b: the maximum cutoff value
    Examples:
        >>> w = torch.empty(3, 5)
        >>> nn.init.trunc_normal_(w)
    """
    with torch.no_grad():
        return _trunc_normal_(tensor, mean, std, a, b)


def embedding_version2scores(scores,version,exponent=2):
    assert version>=0 and version<100,f"ERROR: version need in range [0,100)"
    scale = math.pow(10,exponent)
    scores = (scores*scale).to(torch.int32).to(torch.float32)
    version = version/100
    scores = (scores+version)/scale
    return scores

def embedding_version2coord(coord,version,exponent=0):
    assert version>=0 and version<100,f"ERROR: version need in range [0,100)"
    scale = math.pow(10,exponent)
    coord = (coord*scale).to(torch.int32).to(torch.float32)
    version = version/100
    coord = (coord+version)/scale
    return coord


def add_version2onnx(onnx_path,save_path,version):
    model_proto = onnx.load(onnx_path)
    #graph_proto = model_proto.graph
    #model_metadata = {}
    # 添加元数据
    model_proto.metadata_props.extend([
        onnx.helper.make_string_initializer(
            'model_version',
            onnx.mapping.NP_TYPE_TO_TENSOR_TYPE[type(version)],
            [1],
            [version],
        )
    ])
    if save_path is None:
        save_path = onnx_path
    onnx.save(model_proto,save_path)
    
def add_metadta2onnx(model_onnx,metadata):
    #model_onnx = onnx.load(f)  # load onnx model
    for k, v in metadata.items():
        meta = model_onnx.metadata_props.add()
        meta.key, meta.value = k, str(v)
    
    return model_onnx


class SafeClass:
    """A placeholder class to replace unknown classes during unpickling."""

    def __init__(self, *args, **kwargs):
        """Initialize SafeClass instance, ignoring all arguments."""
        pass

    def __call__(self, *args, **kwargs):
        """Run SafeClass instance, ignoring all arguments."""
        pass


class SafeUnpickler(pickle.Unpickler):
    """Custom Unpickler that replaces unknown classes with SafeClass."""

    def find_class(self, module, name):
        """Attempt to find a class, returning SafeClass if not among safe modules."""
        safe_modules = (
            "torch",
            "collections",
            "collections.abc",
            "builtins",
            "math",
            "numpy",
            # Add other modules considered safe
        )
        if module in safe_modules:
            return super().find_class(module, name)
        else:
            return SafeClass

def safe_load(file,*args,**kwargs):
    # Load via custom pickle module
    safe_pickle = types.ModuleType("safe_pickle")
    safe_pickle.Unpickler = SafeUnpickler
    safe_pickle.load = lambda file_obj: SafeUnpickler(file_obj).load()
    with open(file, "rb") as f:
        ckpt = torch.load(f, pickle_module=safe_pickle,*args,**kwargs)
    return ckpt

def load(file,*args,**kwargs):
    try:
        return torch.load(file,*args,**kwargs)
    except Exception as e:
        print(f"WARNING: load ckpt {file} faild, info: {e}, try safe load...")
        return safe_load(file)

def time_sync():
    """PyTorch-accurate time."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.time()

@contextmanager
def cuda_memory_usage(device=None):
    """
    Monitor and manage CUDA memory usage.

    This function checks if CUDA is available and, if so, empties the CUDA cache to free up unused memory.
    It then yields a dictionary containing memory usage information, which can be updated by the caller.
    Finally, it updates the dictionary with the amount of memory reserved by CUDA on the specified device.

    Args:
        device (torch.device, optional): The CUDA device to query memory usage for. Defaults to None.

    Yields:
        (dict): A dictionary with a key 'memory' initialized to 0, which will be updated with the reserved memory.
    """
    cuda_info = dict(memory=0)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        try:
            yield cuda_info
        finally:
            cuda_info["memory"] = torch.cuda.memory_reserved(device)
    else:
        yield cuda_info


def profile(input, ops, n=10, device=None, max_num_obj=0):
    """
    Ultralytics speed, memory and FLOPs profiler.

    Example:
        ```python
        from ultralytics.utils.torch_utils import profile

        input = torch.randn(16, 3, 640, 640)
        m1 = lambda x: x * torch.sigmoid(x)
        m2 = nn.SiLU()
        profile(input, [m1, m2], n=100)  # profile over 100 iterations
        ```
    """
    results = []
    print(
        f"{'Params':>12s}{'GFLOPs':>12s}{'GPU_mem (GB)':>14s}{'forward (ms)':>14s}{'backward (ms)':>14s}"
        f"{'input':>24s}{'output':>24s}"
    )
    gc.collect()  # attempt to free unused memory
    torch.cuda.empty_cache()
    for x in input if isinstance(input, list) else [input]:
        x = x.to(device)
        #x.requires_grad = True
        for m in ops if isinstance(ops, list) else [ops]:
            m = m.to(device) if hasattr(m, "to") else m  # device
            m = m.half() if hasattr(m, "half") and isinstance(x, torch.Tensor) and x.dtype is torch.float16 else m
            tf, tb, t = 0, 0, [0, 0, 0]  # dt forward, backward
            try:
                flops = thop.profile(m, inputs=[x], verbose=False)[0] / 1e9 * 2 if thop else 0  # GFLOPs
            except Exception:
                flops = 0

            try:
                mem = 0
                for _ in range(n):
                    with cuda_memory_usage(device) as cuda_info:
                        t[0] = time_sync()
                        y = m(x)
                        t[1] = time_sync()
                        try:
                            (sum(yi.sum() for yi in y) if isinstance(y, list) else y).sum().backward()
                            t[2] = time_sync()
                        except Exception:  # no backward method
                            # print(e)  # for debug
                            t[2] = float("nan")
                    mem += cuda_info["memory"] / 1e9  # (GB)
                    tf += (t[1] - t[0]) * 1000 / n  # ms per op forward
                    tb += (t[2] - t[1]) * 1000 / n  # ms per op backward
                    if max_num_obj:  # simulate training with predictions per image grid (for AutoBatch)
                        with cuda_memory_usage(device) as cuda_info:
                            torch.randn(
                                x.shape[0],
                                max_num_obj,
                                int(sum((x.shape[-1] / s) * (x.shape[-2] / s) for s in m.stride.tolist())),
                                device=device,
                                dtype=torch.float32,
                            )
                        mem += cuda_info["memory"] / 1e9  # (GB)
                s_in, s_out = (tuple(x.shape) if isinstance(x, torch.Tensor) else "list" for x in (x, y))  # shapes
                p = sum(x.numel() for x in m.parameters()) if isinstance(m, nn.Module) else 0  # parameters
                print(f"{float(p):,}{flops:12.4g}{mem:>14.3f}{tf:14.4g}{tb:14.4g}{str(s_in):>24s}{str(s_out):>24s}")
                results.append([p, flops, mem, tf, tb, s_in, s_out])
            except Exception as e:
                print(e)
                results.append(None)
            finally:
                gc.collect()  # attempt to free unused memory
                torch.cuda.empty_cache()
    return results


def torch_profile(model,inputs,log_path="./log"):
    # 使用 torch.profiler 捕获性能数据
    with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA],  # 分析 CPU 和 CUDA 活动
            schedule=torch.profiler.schedule(
                wait=1,  # 前1步不采样
                warmup=1,  # 第2步作为热身，不计入结果
                active=3,  # 采集后面3步的性能数据
                repeat=2),  # 重复2轮
            on_trace_ready=torch.profiler.tensorboard_trace_handler(log_path),  # 保存日志以供 TensorBoard 可视化
            record_shapes=True,  # 记录输入张量的形状
            profile_memory=True,  # 分析内存分配
            with_stack=True  # 记录操作的调用堆栈信息
        ) as profiler:
    
        for step in range(10):
            outputs = model(inputs)
            profiler.step() 
    
    print(f"Log path: {log_path}")

def model_parameters_detail(model,level=0,total_nr=-1,thr=0.001,simple=False):
    if total_nr<0:
        total_nr = sum(x.numel() for x in model.parameters()) if isinstance(model, nn.Module) else 0  # parameters
        cur_total_nr = total_nr
    else:
        cur_total_nr = sum(x.numel() for x in model.parameters()) if isinstance(model, nn.Module) else 0  # parameters
    
    if simple and cur_total_nr==0:
        return ""

    def _addindent(s_, numSpaces):
        s = s_.split('\n')
        # don't do anything for single-line stuff
        if len(s) == 1:
            return s_
        first = s.pop(0)
        s = [(numSpaces * ' ') + line for line in s]
        s = '\n'.join(s)
        s = first + '\n' + s
        return s

    extra_lines = []
    extra_repr = ''
    # empty string will be split into list ['']
    if extra_repr:
        extra_lines = extra_repr.split('\n')
    child_lines = []
    if cur_total_nr > total_nr*thr:
            for key, module in model._modules.items():
                mod_str = model_parameters_detail(module,level=level+1,total_nr=total_nr,thr=thr,simple=simple)
                if len(mod_str) == 0:
                    continue
                mod_str = _addindent(mod_str, 2)
                child_lines.append('(' + key + '): ' + mod_str)
    lines = extra_lines + child_lines

    main_str = model._get_name() + f' {cur_total_nr:,}   {cur_total_nr*100/total_nr:.1f}% ('
    if lines:
        # simple one-liner info, which most builtin Modules will use
        if len(extra_lines) == 1 and not child_lines:
            main_str += extra_lines[0]
        else:
            main_str += '\n  ' + '\n  '.join(lines) + '\n'

    main_str += ')'
    return main_str
