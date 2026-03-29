import unittest
import sys
from pathlib import Path
import numpy as np

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

    def test_buffer_float_roundtrip(self):
        if not cuda_available():
            self.skipTest("CUDA unavailable")

        t = ktorch.Tensor([2, 3], dtype=ktorch.DType.F32, device=ktorch.Device.CUDA)
        arr = np.asarray([[1.0, -2.0, 3.0], [4.0, 5.5, -6.0]], dtype=np.float32, order="C")
        t.copy_from_buffer_float(arr)

        out = t.to_numpy_float()
        self.assertEqual(out.dtype, np.float32)
        self.assertEqual(tuple(out.shape), (2, 3))
        assert_allclose(out.reshape(-1).tolist(), arr.reshape(-1).tolist())

    def test_buffer_int32_roundtrip(self):
        if not cuda_available():
            self.skipTest("CUDA unavailable")

        t = ktorch.Tensor([4], dtype=ktorch.DType.I32, device=ktorch.Device.CUDA)
        arr = np.asarray([7, -3, 42, 0], dtype=np.int32)
        t.copy_from_buffer_int32(arr)

        out = t.to_numpy_int32()
        self.assertEqual(out.dtype, np.int32)
        self.assertEqual(tuple(out.shape), (4,))
        self.assertEqual(out.tolist(), arr.tolist())

    def test_buffer_rejects_wrong_dtype(self):
        t = ktorch.Tensor([2, 2], dtype=ktorch.DType.F32, device=ktorch.Device.CPU)
        arr = np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
        with self.assertRaises(Exception):
            t.copy_from_buffer_float(arr)

    def test_buffer_rejects_non_contiguous(self):
        t = ktorch.Tensor([2, 2], dtype=ktorch.DType.F32, device=ktorch.Device.CPU)
        base = np.asarray([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
        non_contig = base[:, ::2]  # shape [2,2], non-C-contiguous
        self.assertFalse(non_contig.flags.c_contiguous)
        with self.assertRaises(Exception):
            t.copy_from_buffer_float(non_contig)

    def test_buffer_rejects_size_mismatch(self):
        t = ktorch.Tensor([2, 2], dtype=ktorch.DType.F32, device=ktorch.Device.CPU)
        arr = np.asarray([1.0, 2.0, 3.0], dtype=np.float32)
        with self.assertRaises(Exception):
            t.copy_from_buffer_float(arr)

    def test_python_from_to_numpy_helpers(self):
        arr = np.asarray([[1.25, -2.5], [3.5, 4.75]], dtype=np.float32)
        t = ktorch.from_numpy(arr, device=ktorch.Device.CPU)
        out = ktorch.to_numpy(t)
        self.assertEqual(out.dtype, np.float32)
        self.assertEqual(tuple(out.shape), (2, 2))
        assert_allclose(out.reshape(-1).tolist(), arr.reshape(-1).tolist())


if __name__ == "__main__":
    unittest.main()
