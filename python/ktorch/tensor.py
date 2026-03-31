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


def from_torch_borrow_cpu(tensor: Any, require_contiguous: bool = True, require_pinned: bool = True):
    return _C.from_torch_borrow_cpu(tensor, require_contiguous, require_pinned)
