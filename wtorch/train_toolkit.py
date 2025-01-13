import os
import torch
import math
from functools import partial
import torch.nn as nn
import time
import inspect
import sys
from .wlr_scheduler import *
from collections import OrderedDict
from .nn import LayerNorm,LayerNorm2d,EvoNormS0,EvoNormS01D,FrozenBatchNorm2d
import traceback
from typing import Union, Iterable
import re


_NORMS = (
    nn.BatchNorm1d,
    nn.BatchNorm2d,
    nn.BatchNorm3d,
    nn.InstanceNorm1d,
    nn.InstanceNorm2d,
    nn.InstanceNorm3d,
    nn.SyncBatchNorm,
    nn.GroupNorm,
    LayerNorm,
    LayerNorm2d,
    EvoNormS0,
    EvoNormS01D,
    FrozenBatchNorm2d,
)

def is_norm(model):
    return isinstance(model,_NORMS)

def __is_name_of(name, names):
    for x in names:
        if name.startswith(x) or name.startswith("module."+x):
            return True
    return False

def is_in_scope(name, scopes):
    for x in scopes:
        if name.startswith(x) or name.startswith("module."+x):
            return True
    return False

def _get_tensor_or_tensors_shape(x):
    if isinstance(x,(list,tuple)):
        res = []
        for v in x:
            if v is not None:
                res.append(v.shape)
        return res
    if x is not None:
        return x.shape
    else:
        return None

def grad_norm(parameters, norm_type: float = 2.0) -> torch.Tensor:
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = [p for p in parameters if p.grad is not None]
    norm_type = float(norm_type)
    if len(parameters) == 0:
        return torch.tensor(0.)
    device = parameters[0].grad.device
    if norm_type == math.inf:
        total_norm = max(p.grad.detach().abs().max().to(device) for p in parameters)
    else:
        total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), norm_type).to(device) for p in parameters]), norm_type)
    return total_norm
def _add_to_dict(v,dicts):
    for i,c in enumerate(dicts):
        if v in c:
            print(f"ERROR: {v} already in dict {i}")
    dicts[0].add(v)

def simple_split_parameters(model,filter=None,return_unused=False,silent=False):
    '''
    Example:
    bn_weights,weights,biases = simple_split_parameters(model)
    optimizer = optim.AdamW(weights, lr=lr,weight_decay=1e-4)
    optimizer.add_param_group(
                    {"params": bias, "weight_decay": 0.0}
                )  # add pg1 with weight_decay
    optimizer.add_param_group({"params": bn_weights,"weight_decay":0.0})
    '''
    bn_weights, weights, biases = [], [], []
    unbn_weights, unweights, unbiases = [], [], []
    parameters_set = set()
    unused_parameters_set = set()
    print(f"Split model parameters")
    print(f"------------------------------------------")
    total_skip = 0
    for k, v in model.named_modules():
        if len(k)==0:
            continue
        if filter is not None and not(filter(k,v)):
            continue
        if hasattr(v, "bias") and isinstance(v.bias, (torch.Tensor,nn.Parameter)):
            if v.bias.requires_grad is False:
                print(f"{k}.bias requires grad == False, skip.")
                unbiases.append(v.bias)
                _add_to_dict(k+".bias",[unused_parameters_set,parameters_set])
                total_skip += 1
            else:
                biases.append(v.bias)  # biases
                parameters_set.add(k+".bias")
        if (isinstance(v, _NORMS) or "bn" in k) and hasattr(v,'weight'):
            if v.weight is None:
                continue
            elif v.weight.requires_grad is False:
                print(f"{k}.weight requires grad == False, skip.")
                unbn_weights.append(v.weight)
                _add_to_dict(k+".weight",[unused_parameters_set,parameters_set])
                total_skip += 1
            else:
                bn_weights.append(v.weight)  # no decay
                parameters_set.add(k+".weight")
        elif hasattr(v, "weight") and isinstance(v.weight, (torch.Tensor,nn.Parameter)):
            if v.weight.requires_grad is False:
                print(f"{k}.weight requires grad == False, skip.")
                unweights.append(v.weight)
                _add_to_dict(k+".weight",[unused_parameters_set,parameters_set])
                total_skip += 1
            else:
                weights.append(v.weight)  # apply decay
                parameters_set.add(k+".weight")
        for k1,p in v.named_parameters(recurse=False):
            if k1 in ["weight","bias"]:
                continue
            if p.requires_grad == False:
                print(f"{k}.{k1} requires grad == False, skip.")
                total_skip += 1
                if "weight" in k:
                    unweights.append(p)
                    _add_to_dict(k+f".{k1}",[unused_parameters_set,parameters_set])
                elif "bias" in k:
                    unbiases.append(p)
                    _add_to_dict(k+f".{k1}",[unused_parameters_set,parameters_set])
                else:
                    if p.ndim>1:
                        unweights.append(p)
                        _add_to_dict(k+f".{k1}",[unused_parameters_set,parameters_set])
                    else:
                        unbiases.append(p)
                        _add_to_dict(k+f".{k1}",[unused_parameters_set,parameters_set])
                continue
            if "weight" in k:
                weights.append(p)
                parameters_set.add(k+f".{k1}")
            elif "bias" in k:
                biases.append(p)
                parameters_set.add(k+f".{k1}")
            else:
                if p.ndim>1:
                    weights.append(p)
                else:
                    biases.append(p)
                parameters_set.add(k+f".{k1}")

    print(f"------------------------------------------")
    if not silent:
        for k,p in model.named_parameters():
            if p.requires_grad == False:
                continue
            if k not in parameters_set:
                print(f"ERROR: {k} not in any parameters set.")
    #batch norm weight, weight, bias
    print(f"Total have {len(list(model.named_parameters()))} parameters.")
    print(f"Finaly find {len(bn_weights)} bn weights, {len(weights)} weights, {len(biases)} biases, total {len(bn_weights)+len(weights)+len(biases)}, total skip {total_skip}.")
    if not return_unused:
        return bn_weights,weights,biases
    else:
        return bn_weights,weights,biases,unbn_weights,unweights,unbiases

def freeze_model(model,freeze_bn=True):
    if freeze_bn:
        model.eval()
    for name, param in model.named_parameters():
        print(name, param.size(), "freeze")
        param.requires_grad = False

def defrost_model(model,defrost_bn=True,silent=False):
    if defrost_bn:
        model.train()
    for name, param in model.named_parameters():
        if not silent:
            print(name, param.size(), "defrost")
        param.requires_grad = True

def defrost_scope(model,scope,defrost_bn=True,silent=False):
    if defrost_bn:
        defrost_bn(model,scope)
    for name, param in model.named_parameters():
        if not is_in_scope(name,scope):
            continue
        if not silent:
            print(name, param.size(), "defrost")
        param.requires_grad = True

def __set_bn_momentum(m,momentum=0.1):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        m.momentum = momentum

def __set_bn_eps(m,eps=1e-3):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        m.eps = eps 

def __fix_bn(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        m.eval()

def defrost_bn(model:torch.nn.Module,scopes=None):

    _nr = 0
    _nr_skip = 0
    for name, ms in model.named_modules():
        if not isinstance(ms, nn.BatchNorm2d):
            continue
        if __is_name_of(name, scopes):
            ms.train()
            print(f"defrost bn {name}")
            _nr += 1
        else:
            _nr_skip += 1
            continue
    print(f"Total defrost {_nr} bn, total {_nr_skip} bn not defrost.")
    sys.stdout.flush()
    return model

def __freeze_bn(model:torch.nn.Module,names2freeze=None):

    _nr = 0
    _nr_skip = 0
    for name, ms in model.named_modules():
        if not isinstance(ms, nn.BatchNorm2d):
            continue
        if __is_name_of(name, names2freeze):
            ms.apply(__fix_bn)
            print(f"Freeze bn {name}")
            _nr += 1
        else:
            _nr_skip += 1
            continue
    print(f"Total freeze {_nr} bn, total {_nr_skip} bn not freeze.")
    sys.stdout.flush()
    return model

def __freeze_bn2(model,names2freeze=None):
    '''
    names2freeze: str/list[str] names to freeze
    '''
    for name in names2freeze:
        child = getattr(model,name)
        FrozenBatchNorm2d.convert_frozen_batchnorm(child)

def freeze_bn(model,names2freeze=None):
    '''
    names2freeze: str/list[str] names to freeze
    '''
    if names2freeze is None:
        model.apply(__fix_bn)
    else:
        if isinstance(names2freeze,(str,bytes)):
            names2freeze = [names2freeze]
        model = __freeze_bn(model,names2freeze)
    
    return model

def freeze_bn2(model,names2freeze=None):
    '''
    names2freeze: str/list[str] names to freeze
    '''
    if names2freeze is None:
        #model.apply(__fix_bn)
        model = FrozenBatchNorm2d.convert_frozen_batchnorm(model)
    else:
        if isinstance(names2freeze,(str,bytes)):
            names2freeze = [names2freeze]
        model = __freeze_bn2(model,names2freeze)
    
    return model

def set_bn_momentum(model,momentum):
    fn = partial(__set_bn_momentum,momentum=momentum)
    model.apply(fn)

def set_bn_eps(model,eps):
    fn = partial(__set_bn_eps,eps=eps)
    model.apply(fn)

def get_gpus_str(gpus):
    gpus_str = ""
    for g in gpus:
        gpus_str += str(g) + ","
    gpus_str = gpus_str[:-1]

    return gpus_str

def show_model_parameters_info(net):
    print("Training parameters.")
    total_train_parameters = 0
    freeze_parameters = []
    unfreeze_parameters = []
    for name, param in net.named_parameters():
        if param.requires_grad:
            print(name, list(param.size()), param.device,'unfreeze')
            total_train_parameters += param.numel()
            unfreeze_parameters.append(name)
    print(f"Total train parameters {total_train_parameters:,}")
    print("Not training parameters.")
    total_not_train_parameters = 0
    for name, param in net.named_parameters():
        if not param.requires_grad:
            print(name, list(param.size()), param.device,'freeze')
            total_not_train_parameters += param.numel()
            freeze_parameters.append(name)
    print(f"Total not train parameters {total_not_train_parameters:,}")

    _nr = 0
    not_freeze_nr =0
    for name, ms in net.named_modules():
        if not isinstance(ms, (nn.BatchNorm2d,FrozenBatchNorm2d)):
            continue
        if not ms.training or isinstance(ms,FrozenBatchNorm2d):
            _nr += 1
        else:
            not_freeze_nr += 1
    print(f"Total freeze {_nr} batch normal layers, {not_freeze_nr} batch normal layer not freeze.")

    return freeze_parameters,unfreeze_parameters

def show_async_norm_states(module):
    for name, child in module.named_modules():
        if isinstance(child, _NORMS):
            info = ""
            for k,v in child.named_parameters():
                if hasattr(v,"requires_grad"):
                    info += f"{k}:{v.requires_grad}, "
            print(f"{name}: {type(child)}: training: {child.training}, requires_grad: {info}")

def get_total_and_free_memory_in_Mb(cuda_device):
    devices_info_str = os.popen(
        "nvidia-smi --query-gpu=memory.total,memory.used --format=csv,nounits,noheader"
    )
    devices_info = devices_info_str.read().strip().split("\n")
    total, used = devices_info[int(cuda_device)].split(",")
    return int(total), int(used)


def occupy_mem(cuda_device, mem_ratio=0.9):
    """
    pre-allocate gpu memory for training to avoid memory Fragmentation.
    """
    total, used = get_total_and_free_memory_in_Mb(cuda_device)
    max_mem = int(total * mem_ratio)
    block_mem = max_mem - used
    x = torch.cuda.FloatTensor(256, 1024, block_mem)
    del x
    time.sleep(5)

def isfinite_hook(module,fea_in,fea_out):
    '''
    register_forward_hook(net,isfinite_hook)
    '''
    if isinstance(fea_in,(tuple,list)):
        if len(fea_in)==1:
            fea_in = fea_in[0]
        elif len(fea_in)==0:
            return None
    #if not torch.all(torch.isfinite(fea_in)):
        #return None
    if not torch.all(torch.isfinite(fea_out)):
        print("Find NaN or infininite")
        #print(f"{inspect.stack()}")
        traceback.print_exc(file=sys.stdout)
        print(f"Input : {torch.min(fea_in).item(),torch.max(fea_in).item(),torch.mean(fea_in).item()}")
        print(f"Output: {torch.min(fea_out).item(),torch.max(fea_out).item(),torch.mean(fea_out).item()}")
        for name, param in module.named_parameters():
            print(f"{name}: {torch.min(param).item(),torch.max(param).item(),torch.mean(param).item()}")

def islarge_hook(module,fea_in,fea_out,max_v=60000):
    '''
    register_forward_hook(net,isfinite_hook)
    '''
    if isinstance(fea_in,(tuple,list)):
        if len(fea_in)==1:
            fea_in = fea_in[0]
        elif len(fea_in)==0:
            return None
    #if not torch.all(torch.isfinite(fea_in)):
        #return None
    if islarge(fea_out,max_v=max_v):
        print("Find Large value")
        #print(f"{inspect.stack()}")
        traceback.print_exc(file=sys.stdout)
        print(f"Input : {torch.min(fea_in).item(),torch.max(fea_in).item(),torch.mean(fea_in).item()}")
        print(f"Output: {torch.min(fea_out).item(),torch.max(fea_out).item(),torch.mean(fea_out).item()}")
        for name, param in module.named_parameters():
            print(f"{name}: {torch.min(param).item(),torch.max(param).item(),torch.mean(param).item()}")


def islarge(x,max_v=65535):
    if x is None:
        return False
    if isinstance(x,(tuple,list)):
        for v in x :
            if islarge(v,max_v=max_v):
                return True
        return False
    return torch.any(torch.abs(x)>max_v)

def isfinite(x):
    if x is None:
        return True
    if isinstance(x,(tuple,list)):
        for v in x :
            if not isfinite(v):
                return False
        return True 
    return torch.all(torch.isfinite(x))

def register_forward_hook(net,hook):
    nr = 0
    for module in net.children():
        register_forward_hook(module,hook)
        nr += 1
    if nr == 0:
        net.register_forward_hook(hook=hook)

def register_backward_hook(net,hook):
    nr = 0
    for module in net.children():
        register_backward_hook(module,hook)
        nr += 1
    if True:
    #if nr == 0:
        #net.register_full_backward_hook(hook=hook)
        net.register_backward_hook(hook=hook)

def tensor_fix_grad(grad):
    '''
    tensor.register_hook(net,isfinite_hook)
    '''
    max_v = 16000.0
    if not torch.all(torch.isfinite(grad)):
        #print(f"infinite grad:",grad.shape,grad)
        #raise RuntimeError(f"infinite grad")
        return torch.zeros_like(grad)
    elif islarge(grad,max_v):
        #print(f"large grad:",grad.shape,torch.min(grad),torch.max(grad))
        return torch.clamp(grad,min=-max_v,max=max_v)
    return grad
    

def tensor_isfinite_hook(grad):
    '''
    tensor.register_hook(net,isfinite_hook)
    '''
    if not torch.all(torch.isfinite(grad)):
        print(f"Find NaN or infininite grad, {grad.shape}")
        #print(f"{inspect.stack()}")
        traceback.print_exc(file=sys.stdout)
        print(f"grad: {torch.min(grad).item(),torch.max(grad).item(),torch.mean(grad).item()}")
        #print("value:",grad)

def tensor_islarge_hook(grad,max_v=60000):
    '''
    tensor.register_hook(net,isfinite_hook)
    '''
    if islarge(grad,max_v=max_v):
        print("Find Large value grad")
        #print(f"{inspect.stack()}")
        traceback.print_exc(file=sys.stdout)
        print(f"Output: {torch.min(grad).item(),torch.max(grad).item(),torch.mean(grad).item()}")

def register_tensor_hook(model,hook):
    '''
    register_tensor_hook(model,tensor_isfinite_hook)
    '''
    for param in model.parameters():
        if param.requires_grad:
            param.register_hook(hook)

def is_any_grad_infinite(model):
    '''
    register_tensor_hook(model,tensor_isfinite_hook)
    '''
    res = False
    for name,param in model.named_parameters():
        if param.requires_grad and param.grad is not None and \
            (not torch.all(torch.isfinite(param.grad))  or islarge(param.grad,max_v=32768.0)):
            print(f"ERROR: {name}: unnormal grad")
            res = True

    return res

def backward_grad_normal_hook(module,grad_input,grad_output):
    '''
    tensor.register_hook(net,isfinite_hook)
    '''
    if not isfinite(grad_output) or islarge(grad_output,max_v=32768.0):
        print("Find NaN or infininite grad")
        #print(f"{inspect.stack()}")
        print(module,_get_tensor_or_tensors_shape(grad_input),_get_tensor_or_tensors_shape(grad_output),grad_input,grad_output)
        #traceback.print_exc(file=sys.stdout)
        #print(f"grad_output: {torch.min(grad_output).item(),torch.max(grad_output).item(),torch.mean(grad_output).item()}")

def finetune_model(model,names_not2train=None,names2train=None):
    if names_not2train is not None:
        finetune_model_nottrain(model,names_not2train)
        if names2train is not None:
            finetune_model_train(model,names2train)
        return

    def is_name_of(name, names):
        for x in names:
            if name.startswith(x) or name.startswith("module."+x):
                return True
        return False

    for name, param in model.named_parameters():
        if is_name_of(name, names2train):
            continue
        param.requires_grad = False

    param_to_update = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            param_to_update.append(param)

    _nr = 0
    for name, ms in model.named_modules():
        if not isinstance(ms, nn.BatchNorm2d):
            continue
        if is_name_of(name, names2train):
            continue
        else:
            ms.eval()
            _nr += 1

def finetune_model_train(model,names2train=None):

    def is_name_of(name, names):
        for x in names:
            if name.startswith(x) or name.startswith("module."+x):
                return True
        return False

    for name, param in model.named_parameters():
        if is_name_of(name, names2train):
            param.requires_grad = True

    _nr = 0
    for name, ms in model.named_modules():
        if not isinstance(ms, nn.BatchNorm2d):
            continue
        if is_name_of(name, names2train):
            ms.train()
            _nr += 1

def finetune_model_nottrain(model:torch.nn.Module,names_not2train):

    if not isinstance(names_not2train,(list,tuple)):
        names_not2train = [names_not2train]

    patterns = [re.compile(x) for x in names_not2train]

    def is_name_of(name, names):
        for x in names:
            if name.startswith(x) or name.startswith("module."+x):
                return True
        for x in patterns:
            if x.match(name) is not None:
                return True
        return False

    for name, param in model.named_parameters():
        if is_name_of(name, names_not2train):
            param.requires_grad = False
    

    param_to_update = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            param_to_update.append(param)

    _nr = 0
    for name, ms in model.named_modules():
        if not isinstance(ms, nn.BatchNorm2d):
            continue
        if is_name_of(name, names_not2train):
            ms.eval()
            _nr += 1
        else:
            continue
    sys.stdout.flush()

