import unittest
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
PYTHON_SRC = REPO_ROOT / "python"
if str(PYTHON_SRC) not in sys.path:
    sys.path.insert(0, str(PYTHON_SRC))

import ktorch
from ktorch import ops

from common import cuda_available, assert_allclose


class OptimizerApiTest(unittest.TestCase):
    def test_sgd_update_in_place(self):
        if not cuda_available():
            self.skipTest("CUDA unavailable")

        param = ktorch.Tensor([2, 3], dtype=ktorch.DType.F32, device=ktorch.Device.CUDA)
        grad = ktorch.Tensor([2, 3], dtype=ktorch.DType.F32, device=ktorch.Device.CUDA)

        p0 = [1.0, -2.0, 3.5, 4.0, 0.0, -1.5]
        g0 = [0.2, -0.1, 1.0, -2.0, 3.0, 0.5]
        param.copy_from_list_float(p0)
        grad.copy_from_list_float(g0)

        ret = ops.sgd_update_(param, grad, 0.25)
        self.assertIsNone(ret)

        expected = [pv - 0.25 * gv for pv, gv in zip(p0, g0)]
        assert_allclose(param.to_list_float(), expected)

    def test_sgd_update_rejects_shape_mismatch(self):
        if not cuda_available():
            self.skipTest("CUDA unavailable")

        param = ktorch.Tensor([4], dtype=ktorch.DType.F32, device=ktorch.Device.CUDA)
        grad = ktorch.Tensor([5], dtype=ktorch.DType.F32, device=ktorch.Device.CUDA)
        with self.assertRaises(Exception):
            ops.sgd_update_(param, grad, 0.1)

    def test_sgd_update_rejects_dtype_mismatch(self):
        if not cuda_available():
            self.skipTest("CUDA unavailable")

        param = ktorch.Tensor([4], dtype=ktorch.DType.F32, device=ktorch.Device.CUDA)
        grad = ktorch.Tensor([4], dtype=ktorch.DType.I32, device=ktorch.Device.CUDA)
        grad.copy_from_list_int32([1, 2, 3, 4])
        with self.assertRaises(Exception):
            ops.sgd_update_(param, grad, 0.1)

    def test_sgd_update_rejects_non_finite_lr(self):
        if not cuda_available():
            self.skipTest("CUDA unavailable")

        param = ktorch.Tensor([4], dtype=ktorch.DType.F32, device=ktorch.Device.CUDA)
        grad = ktorch.Tensor([4], dtype=ktorch.DType.F32, device=ktorch.Device.CUDA)
        grad.copy_from_list_float([1.0, 1.0, 1.0, 1.0])

        with self.assertRaises(Exception):
            ops.sgd_update_(param, grad, float("nan"))
        with self.assertRaises(Exception):
            ops.sgd_update_(param, grad, float("inf"))


if __name__ == "__main__":
    unittest.main()
