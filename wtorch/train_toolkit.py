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

_NORMS = (
    nn.BatchNorm1d,
    nn.BatchNorm2d,
    nn.BatchNorm3d,
    nn.InstanceNorm1d,
    nn.InstanceNorm2d,
    nn.InstanceNorm3d,
    nn.SyncBatchNorm,
)

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

def simple_split_parameters(model,filter=None):
    bn_weights, weights, biases = [], [], []
    parameters_set = set()
    print(f"------------------------------------------")
    total_skip = 0
    for k, v in model.named_modules():
        if filter is not None and not(filter(k,v)):
            continue
        if hasattr(v, "bias") and isinstance(v.bias, nn.Parameter):
            if v.bias.requires_grad is False:
                print(f"{k}.bias requires grad == False, skip.")
                total_skip += 1
            else:
                biases.append(v.bias)  # biases
                parameters_set.add(k+".bias")
        if isinstance(v, nn.BatchNorm2d) or "bn" in k:
            if v.weight.requires_grad is False:
                print(f"{k}.weight requires grad == False, skip.")
                total_skip += 1
            else:
                bn_weights.append(v.weight)  # no decay
                parameters_set.add(k+".weight")
        elif hasattr(v, "weight") and isinstance(v.weight, nn.Parameter):
            if v.weight.requires_grad is False:
                print(f"{k}.weight requires grad == False, skip.")
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
    for k,p in model.named_parameters():
        if p.requires_grad == False:
            continue
        if k not in parameters_set:
            print(f"ERROR: {k} not in any parameters set.")
    #batch norm weight, weight, bias
    '''
    optimizer = optim.AdamW(pg0, lr=lr,weight_decay=0.0)
    optimizer.add_param_group(
                    {"params": pg1, "weight_decay": 4e-5}
                )  # add pg1 with weight_decay
    optimizer.add_param_group({"params": pg2,"weight_decay":0.0})
    '''
    print(f"Total have {len(list(model.named_parameters()))} parameters.")
    print(f"Finaly find {len(bn_weights)} bn weights, {len(weights)} weights, {len(biases)} biases, total {len(bn_weights)+len(weights)+len(biases)}, total skip {total_skip}.")
    return bn_weights,weights,biases

def freeze_model(model,freeze_bn=True):
    if freeze_bn:
        model.eval()
    for name, param in model.named_parameters():
        print(name, param.size(), "freeze")
        param.requires_grad = False

def defrost_model(model,defrost_bn=True):
    if defrost_bn:
        model.train()
    for name, param in model.named_parameters():
        print(name, param.size(), "defrost")
        param.requires_grad = True

def __set_bn_momentum(m,momentum=0.1):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        m.momentum = momentum

def __fix_bn(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        m.eval()

def freeze_bn(model):
    model.apply(__fix_bn)

def set_bn_momentum(model,momentum):
    fn = partial(__set_bn_momentum,momentum=momentum)
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
    for name, param in net.named_parameters():
        if param.requires_grad:
            print(name, list(param.size()), param.device,'unfreeze')
            total_train_parameters += param.numel()
    print(f"Total train parameters {total_train_parameters:,}")
    print("Not training parameters.")
    total_not_train_parameters = 0
    for name, param in net.named_parameters():
        if not param.requires_grad:
            print(name, list(param.size()), param.device,'freeze')
            total_not_train_parameters += param.numel()
    print(f"Total not train parameters {total_not_train_parameters:,}")

    _nr = 0
    not_freeze_nr =0
    for name, ms in net.named_modules():
        if not isinstance(ms, nn.BatchNorm2d):
            continue
        if not ms.training:
            _nr += 1
        else:
            not_freeze_nr += 1
    print(f"Total freeze {_nr} batch normal layers, {not_freeze_nr} batch normal layer not freeze.")

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
    if isinstance(fea_in,(tuple,list)):
        if len(fea_in)==1:
            fea_in = fea_in[0]
        elif len(fea_in)==0:
            return None
    #if not torch.all(torch.isfinite(fea_in)):
        #return None
    if not torch.all(torch.isfinite(fea_out)):
        print("Find NaN or infininite")
        print(f"{inspect.stack()}")
        print(f"Input : {torch.min(fea_in).item(),torch.max(fea_in).item(),torch.mean(fea_in).item()}")
        print(f"Output: {torch.min(fea_out).item(),torch.max(fea_out).item(),torch.mean(fea_out).item()}")
        for name, param in module.named_parameters():
            print(f"{name}: {torch.min(param).item(),torch.max(param).item(),torch.mean(param).item()}")

def islarge(x,max_v=65535):
    return torch.any(torch.abs(x)>max_v)

def islarge_hook(module,fea_in,fea_out):
    if isinstance(fea_in,(tuple,list)):
        if len(fea_in)==1:
            fea_in = fea_in[0]
        elif len(fea_in)==0:
            return None
    #if islarge(fea_in):
        #return None
    if islarge(fea_out):
        print("Find large value")
        print(f"{inspect.stack()}")
        print(f"Input : {torch.min(fea_in).item(),torch.max(fea_in).item(),torch.mean(fea_in).item()}")
        print(f"Output: {torch.min(fea_out).item(),torch.max(fea_out).item(),torch.mean(fea_out).item()}")
        for name, param in module.named_parameters():
            print(f"{name}: {torch.min(param).item(),torch.max(param).item(),torch.mean(param).item()}")

def register_forward_hook(net,hook):
    nr = 0
    for module in net.children():
        register_forward_hook(module,hook)
        nr += 1
    if nr == 0:
        net.register_forward_hook(hook=hook)

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

    print(f"Finetune model.")
    for name, param in model.named_parameters():
        if is_name_of(name, names2train):
            continue
        print(name, param.size(), "freeze")
        param.requires_grad = False

    param_to_update = []
    print("Training parameters.")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name, list(param.size()), 'unfreeze')
            param_to_update.append(param)

    _nr = 0
    for name, ms in model.named_modules():
        if not isinstance(ms, nn.BatchNorm2d):
            continue
        if is_name_of(name, names2train):
            print(f"{name}:{ms} unfreeze.")
            continue
        else:
            print(f"{name}:{ms} freeze.")
            ms.eval()
            _nr += 1
    print(f"Total freeze {_nr} batch normal layers.")

def finetune_model_train(model,names2train=None):

    def is_name_of(name, names):
        for x in names:
            if name.startswith(x) or name.startswith("module."+x):
                return True
        return False

    print(f"Finetune model.")
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
    print(f"Total unfreeze {_nr} batch normal layers.")

def finetune_model_nottrain(model:torch.nn.Module,names_not2train):
    def is_name_of(name, names):
        for x in names:
            if name.startswith(x) or name.startswith("module."+x):
                return True
        return False

    print(f"Finetune model.")
    for name, param in model.named_parameters():
        if is_name_of(name, names_not2train):
            print(name, param.size(), "freeze")
            param.requires_grad = False

    param_to_update = []
    print("Training parameters.")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name, list(param.size()), 'unfreeze')
            param_to_update.append(param)

    _nr = 0
    for name, ms in model.named_modules():
        if not isinstance(ms, nn.BatchNorm2d):
            continue
        if is_name_of(name, names_not2train):
            print(f"{name}:{ms} freeze.")
            ms.eval()
            _nr += 1
        else:
            print(f"{name}:{ms} unfreeze.")
            continue
    print(f"Total freeze {_nr} batch normal layers.")
    sys.stdout.flush()
