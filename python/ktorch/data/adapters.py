from __future__ import annotations

from typing import Any, Tuple

import numpy as np


class TorchDatasetAdapter:
    """Adapter that exposes a torch dataset through ktorch Dataset protocol."""

    def __init__(self, dataset: Any):
        self._dataset = dataset

    def __len__(self) -> int:
        return len(self._dataset)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        sample = self._dataset[idx]
        if not (isinstance(sample, tuple) and len(sample) == 2):
            raise ValueError("ktorch.data.adapters.TorchDatasetAdapter: item must be tuple (features, labels).")
        x, y = sample
        x_np = _to_numpy(x, np.float32)
        y_np = _to_numpy(y, np.int32)
        return x_np, y_np


def _to_numpy(value: Any, dtype) -> np.ndarray:
    # Optional torch support: only import torch if conversion is requested.
    try:
        import torch  # type: ignore
    except Exception:
        torch = None

    if torch is not None and isinstance(value, torch.Tensor):
        arr = value.detach().cpu().contiguous().numpy()
    else:
        arr = np.asarray(value)

    arr = np.asarray(arr, dtype=dtype)
    if not arr.flags.c_contiguous:
        arr = np.ascontiguousarray(arr)
    return arr


def from_torch_dataset(dataset: Any) -> TorchDatasetAdapter:
    return TorchDatasetAdapter(dataset)
