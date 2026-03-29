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


class ClassificationApiTest(unittest.TestCase):
    def test_correct_count_known_case(self):
        if not cuda_available():
            self.skipTest("CUDA unavailable")

        logits = ktorch.Tensor([4, 3], dtype=ktorch.DType.F32, device=ktorch.Device.CUDA)
        logits.copy_from_list_float([
            2.0, 1.0, 0.0,   # pred 0, label 0 -> correct
            0.1, 0.7, 0.2,   # pred 1, label 1 -> correct
            -1.0, 0.0, 1.0,  # pred 2, label 0 -> incorrect
            0.2, 0.9, 0.3,   # pred 1, label 1 -> correct
        ])
        labels = ktorch.Tensor([4], dtype=ktorch.DType.I32, device=ktorch.Device.CUDA)
        labels.copy_from_list_int32([0, 1, 0, 1])

        correct = ops.classification_correct_count(logits, labels)
        self.assertEqual(correct.shape, [1])
        self.assertEqual(correct.dtype, ktorch.DType.I32)
        self.assertEqual(correct.item_int32(), 3)

    def test_rejects_invalid_dtype_or_shape(self):
        if not cuda_available():
            self.skipTest("CUDA unavailable")

        logits_bad_dtype = ktorch.Tensor([2, 3], dtype=ktorch.DType.I32, device=ktorch.Device.CUDA)
        logits_bad_dtype.copy_from_list_int32([1, 2, 3, 4, 5, 6])
        labels = ktorch.Tensor([2], dtype=ktorch.DType.I32, device=ktorch.Device.CUDA)
        labels.copy_from_list_int32([0, 1])
        with self.assertRaises(Exception):
            ops.classification_correct_count(logits_bad_dtype, labels)

        logits = ktorch.Tensor([2, 3], dtype=ktorch.DType.F32, device=ktorch.Device.CUDA)
        logits.copy_from_list_float([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        labels_bad_shape = ktorch.Tensor([2, 1], dtype=ktorch.DType.I32, device=ktorch.Device.CUDA)
        labels_bad_shape.copy_from_list_int32([0, 1])
        with self.assertRaises(Exception):
            ops.classification_correct_count(logits, labels_bad_shape)

    def test_rejects_cpu_inputs(self):
        logits = ktorch.Tensor([2, 2], dtype=ktorch.DType.F32, device=ktorch.Device.CPU)
        logits.copy_from_list_float([0.0, 1.0, 2.0, 3.0])
        labels = ktorch.Tensor([2], dtype=ktorch.DType.I32, device=ktorch.Device.CPU)
        labels.copy_from_list_int32([1, 1])

        with self.assertRaises(Exception):
            ops.classification_correct_count(logits, labels)


if __name__ == "__main__":
    unittest.main()
