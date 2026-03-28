import math
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


class SoftmaxApiTest(unittest.TestCase):
    def test_forward_row_sums(self):
        if not cuda_available():
            self.skipTest("CUDA unavailable")

        x = ktorch.Tensor([2, 3], dtype=ktorch.DType.F32, device=ktorch.Device.CUDA)
        x.copy_from_list_float([1.0, 2.0, 3.0, -1.0, 0.0, 1.0])

        y = ops.softmax_forward(x)
        vals = y.to_list_float()

        self.assertAlmostEqual(vals[0] + vals[1] + vals[2], 1.0, places=5)
        self.assertAlmostEqual(vals[3] + vals[4] + vals[5], 1.0, places=5)

        # First row expected via stable softmax.
        exps = [math.exp(-2.0), math.exp(-1.0), math.exp(0.0)]
        s = sum(exps)
        expected = [exps[0] / s, exps[1] / s, exps[2] / s]
        assert_allclose(vals[:3], expected, atol=1e-5, rtol=1e-5)

    def test_backward_shape(self):
        if not cuda_available():
            self.skipTest("CUDA unavailable")

        y = ktorch.Tensor([2, 3], dtype=ktorch.DType.F32, device=ktorch.Device.CUDA)
        y.copy_from_list_float([0.2, 0.3, 0.5, 0.1, 0.2, 0.7])

        dy = ktorch.Tensor([2, 3], dtype=ktorch.DType.F32, device=ktorch.Device.CUDA)
        dy.copy_from_list_float([1.0, -1.0, 0.5, -0.5, 2.0, -1.0])

        dx = ops.softmax_backward(dy, y)
        self.assertEqual(dx.shape, [2, 3])


if __name__ == "__main__":
    unittest.main()
