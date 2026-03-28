import unittest
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
PYTHON_SRC = REPO_ROOT / "python"
if str(PYTHON_SRC) not in sys.path:
    sys.path.insert(0, str(PYTHON_SRC))

import ktorch
from ktorch import ops

from common import cuda_available


class LayerNormApiTest(unittest.TestCase):
    def test_forward_backward_shapes(self):
        if not cuda_available():
            self.skipTest("CUDA unavailable")

        x = ktorch.Tensor([2, 4], dtype=ktorch.DType.F32, device=ktorch.Device.CUDA)
        gamma = ktorch.Tensor([4], dtype=ktorch.DType.F32, device=ktorch.Device.CUDA)
        beta = ktorch.Tensor([4], dtype=ktorch.DType.F32, device=ktorch.Device.CUDA)

        x.copy_from_list_float([1.0, 2.0, 3.0, 4.0, -1.0, 0.0, 1.0, 2.0])
        gamma.copy_from_list_float([1.0, 1.0, 1.0, 1.0])
        beta.copy_from_list_float([0.0, 0.0, 0.0, 0.0])

        y, ctx = ops.layernorm_forward(x, gamma, beta, 1e-5)
        self.assertEqual(y.shape, [2, 4])

        dy = ktorch.Tensor([2, 4], dtype=ktorch.DType.F32, device=ktorch.Device.CUDA)
        dy.copy_from_list_float([1.0] * 8)

        grads = ops.layernorm_backward(dy, ctx, True, True, True)
        self.assertTrue(grads.has_dX)
        self.assertTrue(grads.has_dgamma)
        self.assertTrue(grads.has_dbeta)
        self.assertEqual(grads.dX.shape, [2, 4])
        self.assertEqual(grads.dgamma.shape, [4])
        self.assertEqual(grads.dbeta.shape, [4])


if __name__ == "__main__":
    unittest.main()
