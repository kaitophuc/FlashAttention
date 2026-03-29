import importlib
from typing import Any

_C = importlib.import_module("ktorch._C")

Tensor = _C.Tensor
Device = _C.Device
DType = _C.DType


def empty(shape, dtype=DType.F32, device=Device.CUDA):
    return _C.empty(shape, dtype, device)


def zeros(shape, dtype=DType.F32, device=Device.CUDA):
    return _C.zeros(shape, dtype, device)


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
