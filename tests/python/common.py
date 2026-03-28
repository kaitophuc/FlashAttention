import math

import ktorch


def cuda_available():
    try:
        _ = ktorch.Tensor([1, 1], dtype=ktorch.DType.F32, device=ktorch.Device.CUDA)
        return True
    except Exception:
        return False


def assert_allclose(lhs, rhs, atol=1e-4, rtol=1e-4):
    if len(lhs) != len(rhs):
        raise AssertionError(f"length mismatch: {len(lhs)} vs {len(rhs)}")
    for i, (a, b) in enumerate(zip(lhs, rhs)):
        tol = atol + rtol * abs(b)
        if math.fabs(a - b) > tol:
            raise AssertionError(f"idx {i}: got {a}, expected {b}, tol {tol}")
