import unittest
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
PYTHON_SRC = REPO_ROOT / "python"
if str(PYTHON_SRC) not in sys.path:
    sys.path.insert(0, str(PYTHON_SRC))

import ktorch

from common import cuda_available, assert_allclose


class TensorApiTest(unittest.TestCase):
    def test_cpu_tensor_metadata_and_copy(self):
        t = ktorch.Tensor([2, 3], dtype=ktorch.DType.F32, device=ktorch.Device.CPU)
        self.assertEqual(t.shape, [2, 3])
        self.assertEqual(t.numel(), 6)
        self.assertEqual(t.nbytes(), 24)

        t.copy_from_list_float([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        assert_allclose(t.to_list_float(), [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])

    def test_cuda_roundtrip(self):
        if not cuda_available():
            self.skipTest("CUDA unavailable")

        t = ktorch.zeros([2, 2], dtype=ktorch.DType.F32, device=ktorch.Device.CUDA)
        t.copy_from_list_float([1.0, -2.0, 3.0, -4.0])
        assert_allclose(t.to_list_float(), [1.0, -2.0, 3.0, -4.0])

        c = t.clone(ktorch.Device.CPU)
        assert_allclose(c.to_list_float(), [1.0, -2.0, 3.0, -4.0])

    def test_from_torch_helper_cpu(self):
        try:
            import torch
        except Exception:
            self.skipTest("torch unavailable")

        x = torch.tensor([[1.0, -2.0], [3.5, 4.25]], dtype=torch.float32)
        y = torch.tensor([7, -3], dtype=torch.int64)

        tx = ktorch.from_torch(x, device=ktorch.Device.CPU)
        ty = ktorch.from_torch(y, device=ktorch.Device.CPU)

        assert_allclose(tx.to_list_float(), [1.0, -2.0, 3.5, 4.25])
        self.assertEqual(ty.to_list_int32(), [7, -3])

    def test_from_torch_borrow_cpu_pinned_default(self):
        try:
            import torch
        except Exception:
            self.skipTest("torch unavailable")

        x = torch.randn(2, 3, dtype=torch.float32)
        with self.assertRaises(Exception):
            _ = ktorch.from_torch_borrow_cpu(x)

        xp = torch.randn(2, 3, dtype=torch.float32).pin_memory()
        tx = ktorch.from_torch_borrow_cpu(xp)
        self.assertEqual(tx.device, ktorch.Device.CPU)
        self.assertTrue(tx.is_borrowed)
        self.assertTrue(tx.is_read_only)
        self.assertTrue(tx.validate_torch_borrow_version())

    def test_from_torch_borrow_cpu_rejects_cuda(self):
        if not cuda_available():
            self.skipTest("CUDA unavailable")
        try:
            import torch
        except Exception:
            self.skipTest("torch unavailable")

        x = torch.randn(2, 3, device="cuda", dtype=torch.float32)
        with self.assertRaises(Exception):
            _ = ktorch.from_torch_borrow_cpu(x, require_pinned=False)

    def test_copy_from_tensor_with_strict_immutability(self):
        if not cuda_available():
            self.skipTest("CUDA unavailable")
        try:
            import torch
        except Exception:
            self.skipTest("torch unavailable")

        src_torch = torch.tensor([[1.0, -2.0], [3.5, 4.25]], dtype=torch.float32).pin_memory()
        src = ktorch.from_torch_borrow_cpu(src_torch)
        dst = ktorch.empty([2, 2], dtype=ktorch.DType.F32, device=ktorch.Device.CUDA)
        s = ktorch.next_stream()

        with ktorch.stream_guard(s):
            dst.copy_from(src, s, True)
        ktorch.synchronize(s)
        assert_allclose(dst.to_list_float(), [1.0, -2.0, 3.5, 4.25])

        src_torch.add_(1.0)
        with self.assertRaises(Exception):
            with ktorch.stream_guard(s):
                dst.copy_from(src, s, True)

    def test_item_float_and_int32(self):
        tf = ktorch.Tensor([1], dtype=ktorch.DType.F32, device=ktorch.Device.CPU)
        tf.copy_from_list_float([3.25])
        self.assertAlmostEqual(tf.item_float(), 3.25, places=6)

        ti = ktorch.Tensor([1], dtype=ktorch.DType.I32, device=ktorch.Device.CPU)
        ti.copy_from_list_int32([42])
        self.assertEqual(ti.item_int32(), 42)

    def test_item_rejects_wrong_dtype_or_shape(self):
        tf = ktorch.Tensor([2], dtype=ktorch.DType.F32, device=ktorch.Device.CPU)
        tf.copy_from_list_float([1.0, 2.0])
        with self.assertRaises(Exception):
            tf.item_float()

        ti = ktorch.Tensor([1], dtype=ktorch.DType.I32, device=ktorch.Device.CPU)
        ti.copy_from_list_int32([7])
        with self.assertRaises(Exception):
            ti.item_float()

    def test_tensor_from_list_float(self):
        t = ktorch.tensor_from_list_float([2, 2], [1.5, -2.0, 3.25, 4.0], device=ktorch.Device.CPU)
        self.assertEqual(t.shape, [2, 2])
        self.assertEqual(t.dtype, ktorch.DType.F32)
        assert_allclose(t.to_list_float(), [1.5, -2.0, 3.25, 4.0])

    def test_tensor_from_list_int32(self):
        t = ktorch.tensor_from_list_int32([4], [7, -3, 11, 0], device=ktorch.Device.CPU)
        self.assertEqual(t.shape, [4])
        self.assertEqual(t.dtype, ktorch.DType.I32)
        self.assertEqual(t.to_list_int32(), [7, -3, 11, 0])

    def test_tensor_from_list_rejects_size_mismatch(self):
        with self.assertRaises(Exception):
            _ = ktorch.tensor_from_list_float([2, 2], [1.0, 2.0, 3.0], device=ktorch.Device.CPU)

        with self.assertRaises(Exception):
            _ = ktorch.tensor_from_list_int32([3], [1, 2], device=ktorch.Device.CPU)

    def test_random_uniform_seeded_cuda(self):
        if not cuda_available():
            self.skipTest("CUDA unavailable")

        a = ktorch.random_uniform([16], low=-0.05, high=0.05, seed=1234)
        b = ktorch.random_uniform([16], low=-0.05, high=0.05, seed=1234)
        c = ktorch.random_uniform([16], low=-0.05, high=0.05, seed=1235)

        av = a.to_list_float()
        bv = b.to_list_float()
        cv = c.to_list_float()

        assert_allclose(av, bv, atol=0.0, rtol=0.0)
        self.assertNotEqual(av, cv)
        for v in av:
            self.assertGreaterEqual(v, -0.05)
            self.assertLess(v, 0.05)

    def test_random_uniform_rejects_invalid_args(self):
        with self.assertRaises(Exception):
            _ = ktorch.random_uniform([4], low=1.0, high=1.0, seed=0, device=ktorch.Device.CUDA)

        with self.assertRaises(Exception):
            _ = ktorch.random_uniform([4], low=2.0, high=-1.0, seed=0, device=ktorch.Device.CUDA)


if __name__ == "__main__":
    unittest.main()
