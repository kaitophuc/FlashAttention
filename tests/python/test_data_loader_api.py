import unittest
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
PYTHON_SRC = REPO_ROOT / "python"
if str(PYTHON_SRC) not in sys.path:
    sys.path.insert(0, str(PYTHON_SRC))

import ktorch


class TinyDataset:
    def __init__(self, n: int):
        self._xs = [np.asarray([i, i + 1], dtype=np.float32) for i in range(n)]
        self._ys = [np.int32(i) for i in range(n)]

    def __len__(self):
        return len(self._xs)

    def __getitem__(self, idx):
        return self._xs[idx], self._ys[idx]


class BadDataset:
    def __len__(self):
        return 3

    def __getitem__(self, idx):
        return {"x": idx, "y": idx}


class DataLoaderApiTest(unittest.TestCase):
    def test_basic_batching_cpu(self):
        ds = TinyDataset(5)
        loader = ktorch.data.DataLoader(
            ds,
            batch_size=2,
            shuffle=False,
            drop_last=False,
            seed=123,
            device=ktorch.Device.CPU,
        )

        batches = list(loader)
        self.assertEqual(len(batches), 3)

        x0, y0 = batches[0]
        self.assertEqual(x0.dtype, ktorch.DType.F32)
        self.assertEqual(y0.dtype, ktorch.DType.I32)
        self.assertEqual(x0.device, ktorch.Device.CPU)
        self.assertEqual(y0.device, ktorch.Device.CPU)
        self.assertEqual(x0.shape, [2, 2])
        self.assertEqual(y0.shape, [2])

        x0_np = x0.to_numpy_float()
        y0_np = y0.to_numpy_int32()
        self.assertEqual(x0_np.tolist(), [[0.0, 1.0], [1.0, 2.0]])
        self.assertEqual(y0_np.tolist(), [0, 1])

    def test_drop_last(self):
        ds = TinyDataset(5)
        loader = ktorch.data.DataLoader(ds, batch_size=2, drop_last=True, device=ktorch.Device.CPU)
        self.assertEqual(len(loader), 2)
        batches = list(loader)
        self.assertEqual(len(batches), 2)

    def test_shuffle_is_deterministic(self):
        ds = TinyDataset(8)
        l1 = ktorch.data.DataLoader(ds, batch_size=2, shuffle=True, seed=42, device=ktorch.Device.CPU)
        l2 = ktorch.data.DataLoader(ds, batch_size=2, shuffle=True, seed=42, device=ktorch.Device.CPU)

        def collect_labels(loader):
            out = []
            for _, y in loader:
                out.extend(y.to_numpy_int32().reshape(-1).tolist())
            return out

        l1_epoch1 = collect_labels(l1)
        l2_epoch1 = collect_labels(l2)
        self.assertEqual(l1_epoch1, l2_epoch1)

        l1_epoch2 = collect_labels(l1)
        l2_epoch2 = collect_labels(l2)
        self.assertEqual(l1_epoch2, l2_epoch2)
        self.assertNotEqual(l1_epoch1, l1_epoch2)

    def test_invalid_item_shape_raises(self):
        loader = ktorch.data.DataLoader(BadDataset(), batch_size=2, device=ktorch.Device.CPU)
        with self.assertRaises(ValueError):
            _ = list(loader)

    def test_torch_adapter_optional(self):
        try:
            import torch
        except Exception:
            self.skipTest("torch unavailable")

        class TorchDS(torch.utils.data.Dataset):
            def __len__(self):
                return 4

            def __getitem__(self, idx):
                x = torch.tensor([float(idx), float(idx + 1)], dtype=torch.float32)
                y = torch.tensor(idx, dtype=torch.int64)
                return x, y

        ds = TorchDS()
        adapted = ktorch.data.adapters.from_torch_dataset(ds)
        loader = ktorch.data.DataLoader(adapted, batch_size=2, shuffle=False, device=ktorch.Device.CPU)

        x, y = next(iter(loader))
        self.assertEqual(x.to_numpy_float().tolist(), [[0.0, 1.0], [1.0, 2.0]])
        self.assertEqual(y.to_numpy_int32().tolist(), [0, 1])


if __name__ == "__main__":
    unittest.main()
