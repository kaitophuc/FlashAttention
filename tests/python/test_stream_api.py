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


class StreamApiTest(unittest.TestCase):
    def test_event_record_and_wait(self):
        if not cuda_available():
            self.skipTest("CUDA unavailable")

        copy_s = ktorch.stream_from_pool(0)
        compute_s = ktorch.stream_from_pool(1)
        ev = ktorch.Event()

        with ktorch.stream_guard(copy_s):
            x = ktorch.Tensor([2, 2], dtype=ktorch.DType.F32, device=ktorch.Device.CUDA)
            x.copy_from_list_float([1.0, 2.0, 3.0, 4.0])
            ktorch.record_event(ev, copy_s)

        with ktorch.stream_guard(compute_s):
            ktorch.wait_event(compute_s, ev)
            y, _ = ops.relu_forward(x)

        ktorch.synchronize(compute_s)
        assert_allclose(y.to_list_float(), [1.0, 2.0, 3.0, 4.0])

    def test_pool_and_next_stream_basics(self):
        if not cuda_available():
            self.skipTest("CUDA unavailable")

        n = ktorch.stream_pool_size()
        self.assertGreaterEqual(n, 2)

        s0 = ktorch.stream_from_pool(0)
        s1 = ktorch.stream_from_pool(1)
        self.assertIsInstance(s0, ktorch.Stream)
        self.assertIsInstance(s1, ktorch.Stream)

        nxt = ktorch.next_stream()
        self.assertIsInstance(nxt, ktorch.Stream)

        with self.assertRaises(Exception):
            _ = ktorch.stream_from_pool(-1)
        with self.assertRaises(Exception):
            _ = ktorch.stream_from_pool(n)

    def test_stream_guard_nested_and_exception_restore(self):
        if not cuda_available():
            self.skipTest("CUDA unavailable")

        a = ktorch.stream_from_pool(0)
        b = ktorch.stream_from_pool(1)

        with ktorch.stream_guard(a):
            with ktorch.stream_guard(b):
                x = ktorch.Tensor([2, 3], dtype=ktorch.DType.F32, device=ktorch.Device.CUDA)
                x.copy_from_list_float([1.0, -2.0, 3.0, -4.0, 5.0, -6.0])
                y, _ = ops.relu_forward(x)
                ktorch.synchronize(b)
                assert_allclose(y.to_list_float(), [1.0, 0.0, 3.0, 0.0, 5.0, 0.0])

        try:
            with ktorch.stream_guard(a):
                raise RuntimeError("intentional")
        except RuntimeError:
            pass

        with ktorch.stream_guard(b):
            t = ktorch.Tensor([2, 2], dtype=ktorch.DType.F32, device=ktorch.Device.CUDA)
            t.copy_from_list_float([1.0, 2.0, 3.0, 4.0])
            y = ops.softmax_forward(t)
            ktorch.synchronize(b)
            vals = y.to_list_float()
            self.assertAlmostEqual(vals[0] + vals[1], 1.0, places=5)
            self.assertAlmostEqual(vals[2] + vals[3], 1.0, places=5)

    def test_interleaved_two_stream_ops_with_explicit_sync(self):
        if not cuda_available():
            self.skipTest("CUDA unavailable")

        a = ktorch.stream_from_pool(2)
        b = ktorch.stream_from_pool(3)

        with ktorch.stream_guard(a):
            xa = ktorch.Tensor([2, 2], dtype=ktorch.DType.F32, device=ktorch.Device.CUDA)
            wa = ktorch.Tensor([2, 2], dtype=ktorch.DType.F32, device=ktorch.Device.CUDA)
            ba = ktorch.Tensor([2], dtype=ktorch.DType.F32, device=ktorch.Device.CUDA)
            xa.copy_from_list_float([1.0, 2.0, -1.0, 3.0])
            wa.copy_from_list_float([2.0, 0.0, 1.0, -1.0])
            ba.copy_from_list_float([0.5, -0.5])
            ya, ctxa = ops.linear_forward(xa, wa, ba)

        with ktorch.stream_guard(b):
            xb = ktorch.Tensor([2, 3], dtype=ktorch.DType.F32, device=ktorch.Device.CUDA)
            xb.copy_from_list_float([1.0, 2.0, 3.0, -1.0, 0.0, 1.0])
            yb = ops.softmax_forward(xb)
            dyb = ktorch.Tensor([2, 3], dtype=ktorch.DType.F32, device=ktorch.Device.CUDA)
            dyb.copy_from_list_float([1.0, -1.0, 0.5, -0.5, 2.0, -1.0])
            dxb = ops.softmax_backward(dyb, yb)

        with ktorch.stream_guard(a):
            dya = ktorch.Tensor([2, 2], dtype=ktorch.DType.F32, device=ktorch.Device.CUDA)
            dya.copy_from_list_float([1.0, 1.0, 1.0, 1.0])
            grada = ops.linear_backward(dya, ctxa, True, True, True)

        ktorch.synchronize(a)
        ktorch.synchronize(b)

        assert_allclose(ya.to_list_float(), [2.5, -1.5, -1.5, -4.5])

        yb_vals = yb.to_list_float()
        self.assertAlmostEqual(yb_vals[0] + yb_vals[1] + yb_vals[2], 1.0, places=5)
        self.assertAlmostEqual(yb_vals[3] + yb_vals[4] + yb_vals[5], 1.0, places=5)

        dxb_vals = dxb.to_list_float()
        self.assertAlmostEqual(dxb_vals[0] + dxb_vals[1] + dxb_vals[2], 0.0, places=4)
        self.assertAlmostEqual(dxb_vals[3] + dxb_vals[4] + dxb_vals[5], 0.0, places=4)

        self.assertTrue(grada.has_dX)
        self.assertTrue(grada.has_dW)
        self.assertTrue(grada.has_db)


if __name__ == "__main__":
    unittest.main()
