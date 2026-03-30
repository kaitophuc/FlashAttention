import importlib.util
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
PYTHON_SRC = REPO_ROOT / "python"
if str(PYTHON_SRC) not in sys.path:
    sys.path.insert(0, str(PYTHON_SRC))

import ktorch

from common import cuda_available, assert_allclose


class FashionMnistManualTrainApiTest(unittest.TestCase):
    @staticmethod
    def _load_script_module():
        script_path = REPO_ROOT / "tools" / "ktorch" / "fashion_mnist_manual_train.py"
        spec = importlib.util.spec_from_file_location("fashion_mnist_manual_train", script_path)
        if spec is None or spec.loader is None:
            raise RuntimeError("failed to load fashion_mnist_manual_train.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def test_init_model_params_uses_direct_ktorch_api(self):
        if not cuda_available():
            self.skipTest("CUDA unavailable")

        m = self._load_script_module()

        self.assertTrue(hasattr(m, "init_model_params"))
        self.assertFalse(hasattr(m, "random_init"))
        self.assertFalse(hasattr(m, "make_tensor"))
        self.assertFalse(hasattr(m, "make_labels_tensor"))

        w1a, b1a, w2a, b2a = m.init_model_params(seed=123, in_dim=8, hidden=4, out_dim=3)
        w1b, b1b, w2b, b2b = m.init_model_params(seed=123, in_dim=8, hidden=4, out_dim=3)

        self.assertEqual(w1a.shape, [4, 8])
        self.assertEqual(b1a.shape, [4])
        self.assertEqual(w2a.shape, [3, 4])
        self.assertEqual(b2a.shape, [3])

        self.assertEqual(w1a.dtype, ktorch.DType.F32)
        self.assertEqual(b1a.dtype, ktorch.DType.F32)
        self.assertEqual(w2a.dtype, ktorch.DType.F32)
        self.assertEqual(b2a.dtype, ktorch.DType.F32)

        assert_allclose(w1a.to_list_float(), w1b.to_list_float(), atol=0.0, rtol=0.0)
        assert_allclose(w2a.to_list_float(), w2b.to_list_float(), atol=0.0, rtol=0.0)
        assert_allclose(b1a.to_list_float(), b1b.to_list_float(), atol=0.0, rtol=0.0)
        assert_allclose(b2a.to_list_float(), b2b.to_list_float(), atol=0.0, rtol=0.0)


if __name__ == "__main__":
    unittest.main()
