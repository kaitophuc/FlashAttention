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

    def test_softmax_cross_entropy_forward_backward(self):
        if not cuda_available():
            self.skipTest("CUDA unavailable")

        logits = ktorch.Tensor([2, 3], dtype=ktorch.DType.F32, device=ktorch.Device.CUDA)
        logits.copy_from_list_float([2.0, 1.0, 0.0, -1.0, 0.5, 3.0])

        labels = ktorch.Tensor([2], dtype=ktorch.DType.I32, device=ktorch.Device.CUDA)
        labels.copy_from_list_int32([0, 2])

        loss, ctx = ops.softmax_cross_entropy_forward(logits, labels)
        self.assertEqual(loss.shape, [1])
        self.assertEqual(ctx.m, 2)
        self.assertEqual(ctx.n, 3)

        # Mean CE must be positive finite for this input.
        loss_val = loss.to_list_float()[0]
        self.assertTrue(math.isfinite(loss_val))
        self.assertGreater(loss_val, 0.0)

        dx = ops.softmax_cross_entropy_backward(ctx)
        self.assertEqual(dx.shape, [2, 3])

        # Reference: dX = (softmax(logits) - onehot(labels)) / batch.
        row0 = [2.0, 1.0, 0.0]
        m0 = max(row0)
        e0 = [math.exp(v - m0) for v in row0]
        s0 = sum(e0)
        p0 = [v / s0 for v in e0]

        row1 = [-1.0, 0.5, 3.0]
        m1 = max(row1)
        e1 = [math.exp(v - m1) for v in row1]
        s1 = sum(e1)
        p1 = [v / s1 for v in e1]

        expected = [
            (p0[0] - 1.0) / 2.0, p0[1] / 2.0, p0[2] / 2.0,
            p1[0] / 2.0, p1[1] / 2.0, (p1[2] - 1.0) / 2.0,
        ]
        assert_allclose(dx.to_list_float(), expected, atol=2e-5, rtol=2e-5)


if __name__ == "__main__":
    unittest.main()
