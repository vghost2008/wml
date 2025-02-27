import cv2
import torch.distributed as dist
import os
import functools
import wml.wml_utils as wmlu
import torch.nn as nn
import torch
from collections import OrderedDict
import subprocess
import pickle
from typing import Callable, Optional, Tuple, Union
from torch.distributed import ProcessGroup
from torch import distributed as torch_dist
from collections.abc import Iterable, Mapping
from torch import Tensor
from torch.nn.parallel._functions import _get_stream




ASYNC_NORM = (
    nn.BatchNorm1d,
    nn.BatchNorm2d,
    nn.BatchNorm3d,
    nn.InstanceNorm1d,
    nn.InstanceNorm2d,
    nn.InstanceNorm3d,
)

def get_world_size(group: Optional[ProcessGroup] = None) -> int:
    """Return the number of the given process group.

    Note:
        Calling ``get_world_size`` in non-distributed environment will return
        1.

    Args:
        group (ProcessGroup, optional): The process group to work on. If None,
            the default process group will be used. Defaults to None.

    Returns:
        int: Return the number of processes of the given process group if in
        distributed environment, otherwise 1.
    """
    if is_distributed():
        # handle low versions of torch like 1.5.0 which does not support
        # passing in None for group argument
        if group is None:
            group = get_default_group()
        return torch_dist.get_world_size(group)
    else:
        return 1

def get_rank() -> int:
    if not dist.is_available():
        return 0
    if not dist.is_initialized():
        return 0
    return dist.get_rank()

def is_main_process() -> bool:
    return get_rank() == 0

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

def configure_nccl():
    """Configure multi-machine environment variables of NCCL."""
    os.environ["NCCL_LAUNCH_MODE"] = "PARALLEL"
    os.environ["NCCL_IB_HCA"] = subprocess.getoutput(
        "pushd /sys/class/infiniband/ > /dev/null; for i in mlx5_*; "
        "do cat $i/ports/1/gid_attrs/types/* 2>/dev/null "
        "| grep v >/dev/null && echo $i ; done; popd > /dev/null"
    )
    os.environ["NCCL_IB_GID_INDEX"] = "3"
    os.environ["NCCL_IB_TC"] = "106"


def configure_omp(num_threads=1):
    """
    If OMP_NUM_THREADS is not configured and world_size is greater than 1,
    Configure OMP_NUM_THREADS environment variables of NCCL to `num_thread`.

    Args:
        num_threads (int): value of `OMP_NUM_THREADS` to set.
    """
    # We set OMP_NUM_THREADS=1 by default, which achieves the best speed on our machines
    # feel free to change it for better performance.
    if "OMP_NUM_THREADS" not in os.environ and get_world_size() > 1:
        os.environ["OMP_NUM_THREADS"] = str(num_threads)
        if is_main_process():
            print(
                "\n***************************************************************\n"
                "We set `OMP_NUM_THREADS` for each process to {} to speed up.\n"
                "please further tune the variable for optimal performance.\n"
                "***************************************************************".format(
                    os.environ["OMP_NUM_THREADS"]
                )
            )


def configure_module(ulimit_value=8192):
    """
    Configure pytorch module environment. setting of ulimit and cv2 will be set.

    Args:
        ulimit_value(int): default open file number on linux. Default value: 8192.
    """
    # system setting
    try:
        import resource

        rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
        resource.setrlimit(resource.RLIMIT_NOFILE, (ulimit_value, rlimit[1]))
    except Exception:
        # Exception might be raised in Windows OS or rlimit reaches max limit number.
        # However, set rlimit value might not be necessary.
        pass

    # cv2
    # multiprocess might be harmful on performance of torch dataloader
    os.environ["OPENCV_OPENCL_RUNTIME"] = "disabled"
    try:
        cv2.setNumThreads(0)
        cv2.ocl.setUseOpenCL(False)
    except Exception:
        # cv2 version mismatch might rasie exceptions.
        pass

def reduce_mean(tensor):
    """"Obtain the mean of tensor on different GPUs."""
    if not (dist.is_available() and dist.is_initialized()):
        return tensor
    tensor = tensor.clone()
    dist.all_reduce(tensor.div_(dist.get_world_size()), op=dist.ReduceOp.SUM)
    return tensor

def is_distributed() -> bool:
    """Return True if distributed environment has been initialized."""
    return torch_dist.is_available() and torch_dist.is_initialized()

def get_default_group() -> Optional[ProcessGroup]:
    """Return default process group."""

    return torch_dist.distributed_c10d._get_default_group()

def barrier(group: Optional[ProcessGroup] = None) -> None:
    """Synchronize all processes from the given process group.

    This collective blocks processes until the whole group enters this
    function.

    Note:
        Calling ``barrier`` in non-distributed environment will do nothing.

    Args:
        group (ProcessGroup, optional): The process group to work on. If None,
            the default process group will be used. Defaults to None.
    """
    if is_distributed():
        # handle low versions of torch like 1.5.0 which does not support
        # passing in None for group argument
        if group is None:
            group = get_default_group()
        torch_dist.barrier(group)

def broadcast(data: Tensor,
              src: int = 0,
              group: Optional[ProcessGroup] = None) -> None:
    """Broadcast the data from ``src`` process to the whole group.

    ``data`` must have the same number of elements in all processes
    participating in the collective.

    Note:
        Calling ``broadcast`` in non-distributed environment does nothing.

    Args:
        data (Tensor): Data to be sent if ``src`` is the rank of current
            process, and data to be used to save received data otherwise.
        src (int): Source rank. Defaults to 0.
        group (ProcessGroup, optional): The process group to work on. If None,
            the default process group will be used. Defaults to None.

    Examples:
        >>> import torch
        >>> import mmengine.dist as dist

        >>> # non-distributed environment
        >>> data = torch.arange(2, dtype=torch.int64)
        >>> data
        tensor([0, 1])
        >>> dist.broadcast(data)
        >>> data
        tensor([0, 1])

        >>> # distributed environment
        >>> # We have 2 process groups, 2 ranks.
        >>> data = torch.arange(2, dtype=torch.int64) + 1 + 2 * rank
        >>> data
        tensor([1, 2]) # Rank 0
        tensor([3, 4]) # Rank 1
        >>> dist.broadcast(data)
        >>> data
        tensor([1, 2]) # Rank 0
        tensor([1, 2]) # Rank 1
    """
    if get_world_size(group) > 1:
        if group is None:
            group = get_default_group()

        input_device = get_data_device(data)
        backend_device = get_comm_device(group)
        data_on_device = cast_data_device(data, backend_device)
        # broadcast requires tensor is contiguous
        data_on_device = data_on_device.contiguous()  # type: ignore
        torch_dist.broadcast(data_on_device, src, group)

        if get_rank(group) != src:
            cast_data_device(data_on_device, input_device, data)


def get_data_device(data: Union[Tensor, Mapping, Iterable]) -> torch.device:
    """Return the device of ``data``.

    If ``data`` is a sequence of Tensor, all items in ``data`` should have a
    same device type.

    If ``data`` is a dict whose values are Tensor, all values should have a
    same device type.

    Args:
        data (Tensor or Sequence or dict): Inputs to be inferred the device.

    Returns:
        torch.device: The device of ``data``.

    Examples:
        >>> import torch
        >>> from mmengine.dist import cast_data_device
        >>> # data is a Tensor
        >>> data = torch.tensor([0, 1])
        >>> get_data_device(data)
        device(type='cpu')
        >>> # data is a list of Tensor
        >>> data = [torch.tensor([0, 1]), torch.tensor([2, 3])]
        >>> get_data_device(data)
        device(type='cpu')
        >>> # data is a dict
        >>> data = {'key1': torch.tensor([0, 1]), 'key2': torch.tensor([0, 1])}
        >>> get_data_device(data)
        device(type='cpu')
    """
    if isinstance(data, Tensor):
        return data.device
    elif isinstance(data, Mapping):
        pre = None
        for v in data.values():
            cur = get_data_device(v)
            if pre is None:
                pre = cur
            else:
                if cur != pre:
                    raise ValueError(
                        'device type in data should be consistent, but got '
                        f'{cur} and {pre}')
        if pre is None:
            raise ValueError('data should not be empty.')
        return pre
    elif isinstance(data, Iterable) and not isinstance(data, str):
        pre = None
        for item in data:
            cur = get_data_device(item)
            if pre is None:
                pre = cur
            else:
                if cur != pre:
                    raise ValueError(
                        'device type in data should be consistent, but got '
                        f'{cur} and {pre}')
        if pre is None:
            raise ValueError('data should not be empty.')
        return pre
    else:
        raise TypeError('data should be a Tensor, sequence of tensor or dict, '
                        f'but got {data}')


def get_comm_device(group: Optional[ProcessGroup] = None) -> torch.device:
    """Return the device for communication among groups.

    Args:
        group (ProcessGroup, optional): The process group to work on.

    Returns:
        torch.device: The device of backend.
    """
    backend = get_backend(group)
    if backend == 'hccl':
        import torch_npu  # noqa: F401
        return torch.device('npu', torch.npu.current_device())
    elif backend == torch_dist.Backend.NCCL:
        return torch.device('cuda', torch.cuda.current_device())
    elif backend == 'cncl':
        import torch_mlu  # noqa: F401
        return torch.device('mlu', torch.mlu.current_device())
    elif backend == 'smddp':
        return torch.device('cuda', torch.cuda.current_device())
    else:
        # GLOO and MPI backends use cpu device by default
        return torch.device('cpu')


def get_backend(group: Optional[ProcessGroup] = None) -> Optional[str]:
    """Return the backend of the given process group.

    Note:
        Calling ``get_backend`` in non-distributed environment will return
        None.

    Args:
        group (ProcessGroup, optional): The process group to work on. The
            default is the general main process group. If another specific
            group is specified, the calling process must be part of
            :attr:`group`. Defaults to None.

    Returns:
        str or None: Return the backend of the given process group as a lower
        case string if in distributed environment, otherwise None.
    """
    if is_distributed():
        # handle low versions of torch like 1.5.0 which does not support
        # passing in None for group argument
        if group is None:
            group = get_default_group()
        return torch_dist.get_backend(group)
    else:
        return None


def cast_data_device(
    data: Union[Tensor, Mapping, Iterable],
    device: torch.device,
    out: Optional[Union[Tensor, Mapping, Iterable]] = None
) -> Union[Tensor, Mapping, Iterable]:
    """Recursively convert Tensor in ``data`` to ``device``.

    If ``data`` has already on the ``device``, it will not be casted again.

    Args:
        data (Tensor or list or dict): Inputs to be casted.
        device (torch.device): Destination device type.
        out (Tensor or list or dict, optional): If ``out`` is specified, its
            value will be equal to ``data``. Defaults to None.

    Returns:
        Tensor or list or dict: ``data`` was casted to ``device``.
    """
    if out is not None:
        if type(data) != type(out):
            raise TypeError(
                'out should be the same type with data, but got data is '
                f'{type(data)} and out is {type(data)}')

        if isinstance(out, set):
            raise TypeError('out should not be a set')

    if isinstance(data, Tensor):
        if get_data_device(data) == device:
            data_on_device = data
        else:
            data_on_device = data.to(device)

        if out is not None:
            # modify the value of out inplace
            out.copy_(data_on_device)  # type: ignore

        return data_on_device
    elif isinstance(data, Mapping):
        data_on_device = {}
        if out is not None:
            data_len = len(data)
            out_len = len(out)  # type: ignore
            if data_len != out_len:
                raise ValueError('length of data and out should be same, '
                                 f'but got {data_len} and {out_len}')

            for k, v in data.items():
                data_on_device[k] = cast_data_device(v, device,
                                                     out[k])  # type: ignore
        else:
            for k, v in data.items():
                data_on_device[k] = cast_data_device(v, device)

        if len(data_on_device) == 0:
            raise ValueError('data should not be empty')

        # To ensure the type of output as same as input, we use `type(data)`
        # to wrap the output
        return type(data)(data_on_device)  # type: ignore
    elif isinstance(data, Iterable) and not isinstance(
            data, str) and not isinstance(data, np.ndarray):
        data_on_device = []
        if out is not None:
            for v1, v2 in zip(data, out):
                data_on_device.append(cast_data_device(v1, device, v2))
        else:
            for v in data:
                data_on_device.append(cast_data_device(v, device))

        if len(data_on_device) == 0:
            raise ValueError('data should not be empty')

        return type(data)(data_on_device)  # type: ignore
    else:
        raise TypeError('data should be a Tensor, list of tensor or dict, '
                        f'but got {data}')


def get_dist_info(group: Optional[ProcessGroup] = None) -> Tuple[int, int]:
    """Get distributed information of the given process group.

    Note:
        Calling ``get_dist_info`` in non-distributed environment will return
        (0, 1).

    Args:
        group (ProcessGroup, optional): The process group to work on. If None,
            the default process group will be used. Defaults to None.

    Returns:
        tuple[int, int]: Return a tuple containing the ``rank`` and
        ``world_size``.
    """
    world_size = get_world_size(group)
    rank = get_rank()
    return rank, world_size

def master_only(func: Callable) -> Callable:
    
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        rank, _ = get_dist_info()
        if rank == 0:
            return func(*args, **kwargs)

    return wrapper

def get_stream(device:int):
    if torch.__version__.startswith("1."):
        return _get_stream(device)
    else:
        if -1 == device:
            return None
        return _get_stream(torch.device(f"cuda:{device}"))