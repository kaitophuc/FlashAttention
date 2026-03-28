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


class ReluApiTest(unittest.TestCase):
    def test_forward_backward(self):
        if not cuda_available():
            self.skipTest("CUDA unavailable")

        x = ktorch.Tensor([2, 3], dtype=ktorch.DType.F32, device=ktorch.Device.CUDA)
        x.copy_from_list_float([-1.0, 0.0, 1.5, 2.0, -3.0, 4.0])

        y, ctx = ops.relu_forward(x)
        assert_allclose(y.to_list_float(), [0.0, 0.0, 1.5, 2.0, 0.0, 4.0])

        dy = ktorch.Tensor([2, 3], dtype=ktorch.DType.F32, device=ktorch.Device.CUDA)
        dy.copy_from_list_float([1.0, 1.0, 1.0, -2.0, 3.0, -4.0])

        dx = ops.relu_backward(dy, ctx)
        assert_allclose(dx.to_list_float(), [0.0, 0.0, 1.0, -2.0, 0.0, -4.0])


if __name__ == "__main__":
    unittest.main()
