from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Iterator, Protocol, Sequence, Tuple, runtime_checkable

import numpy as np

from ..tensor import Device, Tensor, from_numpy


@runtime_checkable
class Dataset(Protocol):
    def __len__(self) -> int:
        ...

    def __getitem__(self, idx: int):
        ...


@dataclass(frozen=True)
class _BatchIndices:
    start: int
    end: int


class DataLoader:
    """Single-process deterministic data loader yielding (x_batch, y_batch) as ktorch tensors."""

    def __init__(
        self,
        dataset: Dataset,
        batch_size: int,
        shuffle: bool = False,
        drop_last: bool = False,
        seed: int = 0,
        device: Device = Device.CUDA,
    ) -> None:
        if not isinstance(dataset, Dataset):
            raise TypeError("ktorch.data.DataLoader: dataset must implement __len__ and __getitem__.")
        if batch_size <= 0:
            raise ValueError("ktorch.data.DataLoader: batch_size must be > 0.")

        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.drop_last = bool(drop_last)
        self.seed = int(seed)
        self.device = device
        self._epoch = 0

    def __len__(self) -> int:
        n = len(self.dataset)
        if self.drop_last:
            return n // self.batch_size
        return (n + self.batch_size - 1) // self.batch_size

    def __iter__(self) -> Iterator[Tuple[Tensor, Tensor]]:
        n = len(self.dataset)
        indices = list(range(n))
        if self.shuffle:
            rng = random.Random(self.seed + self._epoch)
            rng.shuffle(indices)
        self._epoch += 1

        for batch_indices in self._iter_batch_slices(n):
            batch_ids = indices[batch_indices.start:batch_indices.end]
            x_batch, y_batch = self._collate_batch(batch_ids)
            yield from_numpy(x_batch, device=self.device), from_numpy(y_batch, device=self.device)

    def _iter_batch_slices(self, n: int) -> Iterator[_BatchIndices]:
        start = 0
        while start < n:
            end = min(start + self.batch_size, n)
            if self.drop_last and (end - start) < self.batch_size:
                break
            yield _BatchIndices(start=start, end=end)
            start = end

    def _collate_batch(self, batch_ids: Sequence[int]) -> Tuple[np.ndarray, np.ndarray]:
        xs = []
        ys = []
        for idx in batch_ids:
            sample = self.dataset[idx]
            if not (isinstance(sample, tuple) and len(sample) == 2):
                raise ValueError("ktorch.data.DataLoader: dataset item must be a tuple (features, labels).")
            x, y = sample
            xs.append(np.asarray(x, dtype=np.float32))
            ys.append(np.asarray(y, dtype=np.int32))

        try:
            x_batch = np.stack(xs, axis=0)
        except Exception as e:
            raise ValueError("ktorch.data.DataLoader: could not stack features into a batch.") from e

        try:
            y_batch = np.stack(ys, axis=0)
        except Exception as e:
            raise ValueError("ktorch.data.DataLoader: could not stack labels into a batch.") from e

        x_batch = np.ascontiguousarray(x_batch, dtype=np.float32)
        y_batch = np.ascontiguousarray(y_batch, dtype=np.int32)
        return x_batch, y_batch
