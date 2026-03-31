import os
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
PYTHON_SRC = REPO_ROOT / "python"
if str(PYTHON_SRC) not in sys.path:
    sys.path.insert(0, str(PYTHON_SRC))

import ktorch
from ktorch import ops

from common import cuda_available, assert_allclose


def _stress_iters(env_name: str, default: int) -> int:
    raw = os.environ.get(env_name, str(default))
    try:
        value = int(raw)
    except Exception:
        value = default
    return max(1, value)


class BorrowStressApiTest(unittest.TestCase):
    def test_borrow_version_tracking_stress_cpu(self):
        try:
            import torch
        except Exception:
            self.skipTest("torch unavailable")

        iters = _stress_iters("KTORCH_BORROW_STRESS_ITERS_CPU", 256)
        for i in range(iters):
            src = torch.tensor([float(i), -1.0, 2.0, -3.0], dtype=torch.float32).pin_memory()
            borrowed = ktorch.from_torch_borrow_cpu(src)
            self.assertTrue(borrowed.is_borrowed)
            self.assertTrue(borrowed.validate_torch_borrow_version())

            if i % 7 == 0:
                src.add_(1.0)
                self.assertFalse(borrowed.validate_torch_borrow_version())
            else:
                self.assertTrue(borrowed.validate_torch_borrow_version())

    def test_strict_copy_from_borrow_stress_cuda(self):
        if not cuda_available():
            self.skipTest("CUDA unavailable")
        try:
            import torch
        except Exception:
            self.skipTest("torch unavailable")

        iters = _stress_iters("KTORCH_BORROW_STRESS_ITERS_CUDA", 128)
        s = ktorch.next_stream()
        dst = ktorch.empty([2, 3], dtype=ktorch.DType.F32, device=ktorch.Device.CUDA)

        failures = 0
        last_expected = None
        for i in range(iters):
            vals = [
                float(i),
                -float(i) - 0.5,
                1.25,
                -2.5,
                3.75,
                -4.0,
            ]
            src = torch.tensor(vals, dtype=torch.float32).view(2, 3).pin_memory()
            borrowed = ktorch.from_torch_borrow_cpu(src)

            mutate = (i % 11 == 0)
            if mutate:
                src.add_(0.25)
                with self.assertRaises(Exception):
                    with ktorch.stream_guard(s):
                        dst.copy_from(borrowed, s, True)
                failures += 1
                continue

            with ktorch.stream_guard(s):
                dst.copy_from(borrowed, s, True)
            last_expected = vals

            if i % 8 == 0:
                ktorch.synchronize(s)
                out = dst.clone(ktorch.Device.CPU).to_list_float()
                assert_allclose(out, last_expected)

        self.assertGreater(failures, 0)
        if last_expected is not None:
            ktorch.synchronize(s)
            out = dst.clone(ktorch.Device.CPU).to_list_float()
            assert_allclose(out, last_expected)

    def test_borrow_copy_event_pipeline_stress_cuda(self):
        if not cuda_available():
            self.skipTest("CUDA unavailable")
        try:
            import torch
        except Exception:
            self.skipTest("torch unavailable")

        iters = _stress_iters("KTORCH_BORROW_STRESS_ITERS_PIPELINE", 96)
        copy_s = ktorch.stream_from_pool(0)
        compute_s = ktorch.stream_from_pool(1)
        ready = ktorch.Event()

        x_gpu = ktorch.empty([2, 2], dtype=ktorch.DType.F32, device=ktorch.Device.CUDA)
        pending_borrows = []
        last_expected = None
        y = None

        for i in range(iters):
            vals = [float(i), -float(i) - 0.25, 2.0, -3.0]
            src = torch.tensor(vals, dtype=torch.float32).view(2, 2).pin_memory()
            borrowed = ktorch.from_torch_borrow_cpu(src)

            with ktorch.stream_guard(copy_s):
                x_gpu.copy_from(borrowed, copy_s, True)
                ktorch.record_event(ready, copy_s)

            with ktorch.stream_guard(compute_s):
                ktorch.wait_event(compute_s, ready)
                y, _ = ops.relu_forward(x_gpu)

            pending_borrows.append(borrowed)
            last_expected = [max(v, 0.0) for v in vals]

            if len(pending_borrows) >= 8:
                ktorch.synchronize(compute_s)
                out = y.clone(ktorch.Device.CPU).to_list_float()
                assert_allclose(out, last_expected)
                pending_borrows.clear()

        if last_expected is not None:
            ktorch.synchronize(compute_s)
            out = y.clone(ktorch.Device.CPU).to_list_float()
            assert_allclose(out, last_expected)


if __name__ == "__main__":
    unittest.main()
