import importlib
from typing import Any

_C = importlib.import_module("ktorch._C")

Tensor = _C.Tensor
Device = _C.Device
DType = _C.DType
Stream = _C.Stream
Event = _C.Event


def empty(shape, dtype=DType.F32, device=Device.CUDA):
    return _C.empty(shape, dtype, device)


def zeros(shape, dtype=DType.F32, device=Device.CUDA):
    return _C.zeros(shape, dtype, device)


def random_uniform(shape, low=-1.0, high=1.0, seed=0, dtype=DType.F32, device=Device.CUDA):
    return _C.random_uniform(shape, low, high, seed, dtype, device)


def tensor_from_list_float(shape, values, device=Device.CUDA):
    return _C.tensor_from_list_float(shape, values, device)


def tensor_from_list_int32(shape, values, device=Device.CUDA):
    return _C.tensor_from_list_int32(shape, values, device)


def current_stream() -> Stream:
    return _C.current_stream()


def set_current_stream(stream: Stream) -> None:
    _C.set_current_stream(stream)


def stream_from_pool(idx: int) -> Stream:
    return _C.stream_from_pool(idx)


def next_stream() -> Stream:
    return _C.next_stream()


def stream_pool_size() -> int:
    return _C.stream_pool_size()


def stream_guard(stream: Stream):
    return _C.stream_guard(stream)


def synchronize(stream: Stream | None = None) -> None:
    if stream is None:
        _C.synchronize()
    else:
        stream.synchronize()


def record_event(event: Event, stream: Stream) -> None:
    _C.record_event(event, stream)


def wait_event(stream: Stream, event: Event) -> None:
    _C.wait_event(stream, event)


def from_numpy(array: Any, device=Device.CUDA):
    try:
        import numpy as np
    except Exception as e:
        raise RuntimeError("ktorch.from_numpy requires numpy.") from e

    arr = np.asarray(array)
    if not arr.flags.c_contiguous:
        raise ValueError("ktorch.from_numpy: input array must be C-contiguous.")

    if arr.dtype == np.float32:
        t = Tensor(list(arr.shape), dtype=DType.F32, device=device)
        t.copy_from_buffer_float(arr)
        return t
    if arr.dtype == np.int32:
        t = Tensor(list(arr.shape), dtype=DType.I32, device=device)
        t.copy_from_buffer_int32(arr)
        return t
    raise ValueError("ktorch.from_numpy supports only float32 and int32 arrays.")


def from_torch(tensor: Any, device=Device.CUDA):
    try:
        import torch
    except Exception as e:
        raise RuntimeError("ktorch.from_torch requires torch.") from e

    if not isinstance(tensor, torch.Tensor):
        raise TypeError("ktorch.from_torch: input must be a torch.Tensor.")

    t = tensor.contiguous()
    if t.dtype == torch.float32:
        out = Tensor(list(t.shape), dtype=DType.F32, device=device)
        out.copy_from_torch_float(t)
        return out
    if t.dtype == torch.int32:
        out = Tensor(list(t.shape), dtype=DType.I32, device=device)
        out.copy_from_torch_int32(t)
        return out
    if t.dtype == torch.int64:
        t_i32 = t.to(dtype=torch.int32).contiguous()
        out = Tensor(list(t_i32.shape), dtype=DType.I32, device=device)
        out.copy_from_torch_int32(t_i32)
        return out
    raise ValueError("ktorch.from_torch supports only torch.float32, torch.int32, and torch.int64.")


def from_torch_borrow_cpu(tensor: Any, require_contiguous: bool = True, require_pinned: bool = True):
    return _C.from_torch_borrow_cpu(tensor, require_contiguous, require_pinned)


def to_numpy(tensor: Tensor):
    try:
        import numpy as np
    except Exception as e:
        raise RuntimeError("ktorch.to_numpy requires numpy.") from e

    if tensor.dtype == DType.F32:
        return tensor.to_numpy_float()
    if tensor.dtype == DType.I32:
        return tensor.to_numpy_int32()
    raise ValueError("ktorch.to_numpy supports only F32 and I32 tensors.")
