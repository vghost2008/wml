r""""Contains definitions of the methods used by the _BaseDataLoaderIter to put
fetched tensors into pinned memory.

These **needs** to be in global scope since Py2 doesn't support serializing
static methods.
"""

import time
import torch
import collections
if torch.__version__ < "1.9.0":
    from torch._six import queue, container_abcs, string_classes
else:
    import queue
    import collections as container_abcs
    string_classes = (str, bytes)

from . import MP_STATUS_CHECK_INTERVAL
import os
from torch._utils import ExceptionWrapper


def _pin_memory_loop(in_queue, out_queue, device_id, done_event):
    # This setting is thread local, and prevents the copy in pin_memory from
    # consuming all CPU cores.
    torch.set_num_threads(1)

    torch.cuda.set_device(device_id)

    # See NOTE [ Data Loader Multiprocessing Shutdown Logic ] for details on the
    # logic of this function.
    while not done_event.is_set():
        try:
            r = in_queue.get(timeout=MP_STATUS_CHECK_INTERVAL)
        except queue.Empty:
            continue
        idx, data = r
        if not done_event.is_set() and not isinstance(data, ExceptionWrapper):
            try:
                data = pin_memory(data)
            except Exception:
                data = ExceptionWrapper(
                    where="in pin memory thread for device {}".format(device_id))
            r = (idx, data)
        while not done_event.is_set():
            try:
                out_queue.put(r, timeout=MP_STATUS_CHECK_INTERVAL)
                time.sleep(1e-4)
                break
            except queue.Full:
                time.sleep(1)
                continue
        del r  # save memory


def pin_memory(data):
    if isinstance(data, torch.Tensor):
        return data.pin_memory().cuda()
    elif isinstance(data, string_classes):
        return data
    elif isinstance(data, collections.abc.Mapping):
        return {k: pin_memory(sample) for k, sample in data.items()}
    elif isinstance(data, tuple) and hasattr(data, '_fields'):  # namedtuple
        return type(data)(*(pin_memory(sample) for sample in data))
    elif isinstance(data, collections.abc.Sequence):
        return [pin_memory(sample) for sample in data]
    elif hasattr(data, "pin_memory"):
        return data.pin_memory()
    else:
        return data

def _pin_memory_loop_stream(in_queue, out_queue, device_id, done_event,stream):
    # This setting is thread local, and prevents the copy in pin_memory from
    # consuming all CPU cores.
    torch.set_num_threads(1)

    torch.cuda.set_device(device_id)

    # See NOTE [ Data Loader Multiprocessing Shutdown Logic ] for details on the
    # logic of this function.
    while not done_event.is_set():
        try:
            r = in_queue.get(timeout=MP_STATUS_CHECK_INTERVAL)
        except queue.Empty:
            continue
        idx, data = r
        if not done_event.is_set() and not isinstance(data, ExceptionWrapper):
            try:
                with torch.cuda.stream(stream):
                    data = pin_memory_stream(data)
            except Exception:
                data = ExceptionWrapper(
                    where="in pin memory thread for device {}".format(device_id))
            r = (idx, data)
        while not done_event.is_set():
            try:
                out_queue.put(r, timeout=MP_STATUS_CHECK_INTERVAL)
                time.sleep(1e-4)
                break
            except queue.Full:
                time.sleep(1)
                continue
        del r  # save memory


def pin_memory_stream(data):
    if isinstance(data, torch.Tensor):
        if data.dtype==torch.int16: #hack: 不处理int16
            return data
        return data.pin_memory().cuda(non_blocking=True)
    elif isinstance(data, string_classes):
        return data
    elif isinstance(data, collections.abc.Mapping):
        return {k: pin_memory_stream(sample) for k, sample in data.items()}
    elif isinstance(data, tuple) and hasattr(data, '_fields'):  # namedtuple
        return type(data)(*(pin_memory_stream(sample) for sample in data))
    elif isinstance(data, collections.abc.Sequence):
        return [pin_memory_stream(sample) for sample in data]
    elif hasattr(data, "pin_memory"):
        return data.pin_memory().cuda(non_blocking=True)
    else:
        return data
