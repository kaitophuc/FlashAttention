import importlib

_C = importlib.import_module("ktorch._C")

Tensor = _C.Tensor
Device = _C.Device
DType = _C.DType


def empty(shape, dtype=DType.F32, device=Device.CUDA):
    return _C.empty(shape, dtype, device)


def zeros(shape, dtype=DType.F32, device=Device.CUDA):
    return _C.zeros(shape, dtype, device)
