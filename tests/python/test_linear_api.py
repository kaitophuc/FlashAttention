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


class LinearApiTest(unittest.TestCase):
    def test_forward_and_backward_shapes(self):
        if not cuda_available():
            self.skipTest("CUDA unavailable")

        x = ktorch.Tensor([2, 3], dtype=ktorch.DType.F32, device=ktorch.Device.CUDA)
        w = ktorch.Tensor([4, 3], dtype=ktorch.DType.F32, device=ktorch.Device.CUDA)
        b = ktorch.Tensor([4], dtype=ktorch.DType.F32, device=ktorch.Device.CUDA)

        x.copy_from_list_float([1.0, 2.0, 3.0, -1.0, 0.0, 1.0])
        w.copy_from_list_float([
            1.0, 0.0, 0.0,
            0.0, 1.0, 0.0,
            0.0, 0.0, 1.0,
            1.0, 1.0, 1.0,
        ])
        b.copy_from_list_float([0.0, 0.0, 0.0, 1.0])

        y, ctx = ops.linear_forward(x, w, b)
        self.assertEqual(y.shape, [2, 4])

        expected = [1.0, 2.0, 3.0, 7.0, -1.0, 0.0, 1.0, 1.0]
        assert_allclose(y.to_list_float(), expected)

        dy = ktorch.Tensor([2, 4], dtype=ktorch.DType.F32, device=ktorch.Device.CUDA)
        dy.copy_from_list_float([1.0] * 8)

        grads = ops.linear_backward(dy, ctx, True, True, True)
        self.assertTrue(grads.has_dX)
        self.assertTrue(grads.has_dW)
        self.assertTrue(grads.has_db)
        self.assertEqual(grads.dX.shape, [2, 3])
        self.assertEqual(grads.dW.shape, [4, 3])
        self.assertEqual(grads.db.shape, [4])


if __name__ == "__main__":
    unittest.main()
