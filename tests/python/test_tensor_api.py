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


if __name__ == "__main__":
    unittest.main()
