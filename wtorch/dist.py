import torch.distributed as dist
import os
import functools
import wml_utils as wmlu
import torch.nn as nn
import torch
from collections import OrderedDict
import pickle

ASYNC_NORM = (
    nn.BatchNorm1d,
    nn.BatchNorm2d,
    nn.BatchNorm3d,
    nn.InstanceNorm1d,
    nn.InstanceNorm2d,
    nn.InstanceNorm3d,
)

def get_world_size() -> int:
    if not dist.is_available():
        return 1
    if not dist.is_initialized():
        return 1
    return dist.get_world_size()


def get_rank() -> int:
    if not dist.is_available():
        return 0
    if not dist.is_initialized():
        return 0
    return dist.get_rank()

def setup_dist_group(rank,world_size,port="12355",host="localhost",backend='nccl'):
    os.environ['MASTER_ADDR'] = host
    os.environ['MASTER_PORT'] = port
    #backend: gloo, nccl
    dist.init_process_group(backend,rank=rank,world_size=world_size)

def cleanup_dist_train():
    dist.destroy_process_group()

@functools.lru_cache()
def _get_global_gloo_group():
    """
    Return a process group based on gloo backend, containing all the ranks
    The result is cached.
    """
    if dist.get_backend() == "nccl":
        return dist.new_group(backend="gloo")
    else:
        return dist.group.WORLD

def pyobj2tensor(pyobj, device="cuda"):
    """serialize picklable python object to tensor"""
    storage = torch.ByteStorage.from_buffer(pickle.dumps(pyobj))
    return torch.ByteTensor(storage).to(device=device)

def tensor2pyobj(tensor):
    """deserialize tensor to picklable python object"""
    return pickle.loads(tensor.cpu().numpy().tobytes())

def _get_reduce_op(op_name):
    return {
        "sum": dist.ReduceOp.SUM,
        "mean": dist.ReduceOp.SUM,
    }[op_name.lower()]

def all_reduce(py_dict, op="sum", group=None):
    """
    Apply all reduce function for python dict object.
    NOTE: make sure that every py_dict has the same keys and values are in the same shape.

    Args:
        py_dict (dict): dict to apply all reduce op.
        op (str): operator, could be "sum" or "mean".
    """
    world_size = get_world_size()
    if world_size == 1:
        return py_dict
    if group is None:
        group = _get_global_gloo_group()
    if dist.get_world_size(group) == 1:
        return py_dict

    # all reduce logic across different devices.
    py_key = list(py_dict.keys())
    py_key_tensor = pyobj2tensor(py_key)
    dist.broadcast(py_key_tensor, src=0)
    py_key = tensor2pyobj(py_key_tensor)

    tensor_shapes = [py_dict[k].shape for k in py_key]
    tensor_numels = [py_dict[k].numel() for k in py_key]

    flatten_tensor = torch.cat([py_dict[k].flatten() for k in py_key])
    dist.all_reduce(flatten_tensor, op=_get_reduce_op(op))
    if op == "mean":
        flatten_tensor /= world_size

    split_tensors = [
        x.reshape(shape)
        for x, shape in zip(torch.split(flatten_tensor, tensor_numels), tensor_shapes)
    ]
    return OrderedDict({k: v for k, v in zip(py_key, split_tensors)})

def get_async_norm_states(module):
    async_norm_states = OrderedDict()
    for name, child in module.named_modules():
        if isinstance(child, ASYNC_NORM):
            for k, v in child.state_dict().items():
                async_norm_states[".".join([name, k])] = v
    return async_norm_states

def all_reduce_norm(module):
    """
    All reduce norm statistics in different devices.
    """
    states = get_async_norm_states(module)
    print("Reduce keys:")
    wmlu.show_list(list(states.keys()))
    states = all_reduce(states, op="mean")
    module.load_state_dict(states, strict=False)

def _find_free_port():
    """
    Find an available port of current machine / node.
    """
    import socket

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # Binding to port 0 will cause the OS to find an available port for us
    sock.bind(("", 0))
    port = sock.getsockname()[1]
    sock.close()
    # NOTE: there is still a chance the port could be taken by other processes.
    return port

def convert_sync_batchnorm(module, process_group=None):
        r"""Helper function to convert all :attr:`BatchNorm*D` layers in the model to
        :class:`torch.nn.SyncBatchNorm` layers.

        Args:
            module (nn.Module): module containing one or more attr:`BatchNorm*D` layers
            process_group (optional): process group to scope synchronization,
                default is the whole world

        Returns:
            The original :attr:`module` with the converted :class:`torch.nn.SyncBatchNorm`
            layers. If the original :attr:`module` is a :attr:`BatchNorm*D` layer,
            a new :class:`torch.nn.SyncBatchNorm` layer object will be returned
            instead.

        Example::

            >>> # Network with nn.BatchNorm layer
            >>> module = torch.nn.Sequential(
            >>>            torch.nn.Linear(20, 100),
            >>>            torch.nn.BatchNorm1d(100),
            >>>          ).cuda()
            >>> # creating process group (optional)
            >>> # ranks is a list of int identifying rank ids.
            >>> ranks = list(range(8))
            >>> r1, r2 = ranks[:4], ranks[4:] 
            >>> # Note: every rank calls into new_group for every
            >>> # process group created, even if that rank is not
            >>> # part of the group.
            >>> process_groups = [torch.distributed.new_group(pids) for pids in [r1, r2]]
            >>> process_group = process_groups[0 if dist.get_rank() <= 3 else 1]
            >>> sync_bn_module = torch.nn.SyncBatchNorm.convert_sync_batchnorm(module, process_group)

        """
        module_output = module
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm) and module.training:
            module_output = torch.nn.SyncBatchNorm(module.num_features,
                                                   module.eps, module.momentum,
                                                   module.affine,
                                                   module.track_running_stats,
                                                   process_group)
            if module.affine:
                with torch.no_grad():
                    module_output.weight = module.weight
                    module_output.bias = module.bias
            module_output.running_mean = module.running_mean
            module_output.running_var = module.running_var
            module_output.num_batches_tracked = module.num_batches_tracked
            if hasattr(module, "qconfig"):
                module_output.qconfig = module.qconfig
        for name, child in module.named_children():
            module_output.add_module(name, convert_sync_batchnorm(child, process_group))
        del module
        return module_output
