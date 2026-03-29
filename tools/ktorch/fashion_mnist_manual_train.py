#!/usr/bin/env python3
import argparse
import random
import statistics
import time
from typing import List, Sequence, Tuple

import numpy as np

import ktorch
from ktorch import ops
from ktorch.data import DataLoader
from ktorch.data.adapters import from_torch_dataset


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train a tiny FashionMNIST MLP with manual ktorch backprop.")
    p.add_argument("--data-root", default="./data", help="Dataset root directory.")
    p.add_argument("--batch-size", type=int, default=64, help="Mini-batch size.")
    p.add_argument("--epochs", type=int, default=3, help="Number of training epochs.")
    p.add_argument("--hidden-dim", type=int, default=128, help="Hidden size for 2-layer MLP.")
    p.add_argument("--lr", type=float, default=0.05, help="SGD learning rate.")
    p.add_argument("--max-train-batches", type=int, default=0, help="Limit train batches per epoch (0 = full epoch).")
    p.add_argument("--max-test-batches", type=int, default=0, help="Limit test batches (0 = full test set).")
    p.add_argument("--seed", type=int, default=123, help="Random seed.")
    p.add_argument(
        "--benchmark-compare",
        action="store_true",
        help="Run deterministic train-step benchmark (native ktorch path) and exit.",
    )
    p.add_argument("--benchmark-warmup", type=int, default=20, help="Warmup steps for benchmark mode.")
    p.add_argument("--benchmark-steps", type=int, default=100, help="Measured steps for benchmark mode.")
    return p.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)


def random_init(rows: int, cols: int, scale: float) -> List[float]:
    vals = []
    for _ in range(rows * cols):
        vals.append(random.uniform(-1.0, 1.0) * scale)
    return vals


def zeros(n: int) -> List[float]:
    return [0.0] * n


def make_tensor(shape: Sequence[int], values: Sequence[float]) -> ktorch.Tensor:
    t = ktorch.Tensor(list(shape), dtype=ktorch.DType.F32, device=ktorch.Device.CUDA)
    t.copy_from_list_float(list(values))
    return t


def make_labels_tensor(labels: Sequence[int]) -> ktorch.Tensor:
    t = ktorch.Tensor([len(labels)], dtype=ktorch.DType.I32, device=ktorch.Device.CUDA)
    t.copy_from_list_int32([int(v) for v in labels])
    return t


def compute_correct_fast(logits_np: np.ndarray, labels_np: np.ndarray) -> int:
    preds = np.argmax(logits_np, axis=1)
    return int(np.sum(preds == labels_np))


def next_batch(it, loader):
    try:
        return next(it), it
    except StopIteration:
        it = iter(loader)
        return next(it), it


def train_step(batch, w1, b1, w2, b2, in_dim: int, out_dim: int, lr: float, compute_metrics: bool):
    x, labels_t = batch
    bsz = int(x.shape[0])
    x.view([bsz, in_dim])

    z1, ctx1 = ops.linear_forward(x, w1, b1)
    a1, relu_ctx = ops.relu_forward(z1)
    z2, ctx2 = ops.linear_forward(a1, w2, b2)

    loss_t, ce_ctx = ops.softmax_cross_entropy_forward(z2, labels_t)
    dz2 = ops.softmax_cross_entropy_backward(ce_ctx)

    g2 = ops.linear_backward(dz2, ctx2, True, True, True)
    dz1 = ops.relu_backward(g2.dX, relu_ctx)
    g1 = ops.linear_backward(dz1, ctx1, False, True, True)

    ops.sgd_update_(w2, g2.dW, lr)
    ops.sgd_update_(b2, g2.db, lr)
    ops.sgd_update_(w1, g1.dW, lr)
    ops.sgd_update_(b1, g1.db, lr)

    if not compute_metrics:
        loss = float(loss_t.to_numpy_float().reshape(-1)[0])
        return loss, 0, bsz

    loss = float(loss_t.to_numpy_float().reshape(-1)[0])
    logits_np = z2.to_numpy_float().reshape(bsz, out_dim)
    labels_np = labels_t.to_numpy_int32().reshape(-1)
    correct = compute_correct_fast(logits_np, labels_np)

    return loss, correct, bsz


def evaluate(test_loader, w1, b1, w2, b2, in_dim: int, out_dim: int, max_batches: int) -> Tuple[float, float]:
    total_loss = 0.0
    total_correct = 0
    total_seen = 0

    for batch_idx, batch in enumerate(test_loader):
        if max_batches > 0 and batch_idx >= max_batches:
            break

        x, labels_t = batch
        bsz = int(x.shape[0])
        x.view([bsz, in_dim])

        z1, _ = ops.linear_forward(x, w1, b1)
        a1, _ = ops.relu_forward(z1)
        z2, _ = ops.linear_forward(a1, w2, b2)

        loss_t, _ = ops.softmax_cross_entropy_forward(z2, labels_t)

        loss = float(loss_t.to_numpy_float().reshape(-1)[0])
        logits_np = z2.to_numpy_float().reshape(bsz, out_dim)
        labels_np = labels_t.to_numpy_int32().reshape(-1)
        correct = compute_correct_fast(logits_np, labels_np)

        total_loss += loss * bsz
        total_correct += correct
        total_seen += bsz

    avg_loss = total_loss / max(total_seen, 1)
    acc = total_correct / max(total_seen, 1)
    return avg_loss, acc


def init_params(in_dim: int, hidden: int, out_dim: int):
    w1_vals = random_init(hidden, in_dim, scale=0.05)
    b1_vals = zeros(hidden)
    w2_vals = random_init(out_dim, hidden, scale=0.05)
    b2_vals = zeros(out_dim)
    return w1_vals, b1_vals, w2_vals, b2_vals


def build_params(in_dim: int, hidden: int, out_dim: int, init_vals):
    w1_vals, b1_vals, w2_vals, b2_vals = init_vals
    w1 = make_tensor([hidden, in_dim], w1_vals)
    b1 = make_tensor([hidden], b1_vals)
    w2 = make_tensor([out_dim, hidden], w2_vals)
    b2 = make_tensor([out_dim], b2_vals)
    return w1, b1, w2, b2


def make_ktorch_loader(dataset, batch_size: int, shuffle: bool, seed: int):
    ds = from_torch_dataset(dataset)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=False,
        seed=seed,
        device=ktorch.Device.CUDA,
    )


def run_step_benchmark(train_loader, args, in_dim: int, hidden: int, out_dim: int, init_vals) -> float:
    w1, b1, w2, b2 = build_params(in_dim, hidden, out_dim, init_vals)
    loader_it = iter(train_loader)
    measured: List[float] = []

    total_steps = args.benchmark_warmup + args.benchmark_steps
    for step in range(total_steps):
        batch, loader_it = next_batch(loader_it, train_loader)
        t0 = time.perf_counter()
        train_step(batch, w1, b1, w2, b2, in_dim, out_dim, args.lr, compute_metrics=False)
        dt = time.perf_counter() - t0
        if step >= args.benchmark_warmup:
            measured.append(dt)

    return statistics.median(measured)


def run_compare_benchmark(train_ds, args, in_dim: int, hidden: int, out_dim: int) -> None:
    init_vals = init_params(in_dim, hidden, out_dim)
    loader = make_ktorch_loader(train_ds, batch_size=args.batch_size, shuffle=False, seed=args.seed)
    median = run_step_benchmark(loader, args, in_dim, hidden, out_dim, init_vals)
    print("[benchmark] train-step median (batch_size=%d, warmup=%d, steps=%d)" % (
        args.batch_size,
        args.benchmark_warmup,
        args.benchmark_steps,
    ))
    print(f"[benchmark] native_ktorch={median * 1000.0:.3f} ms")


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    try:
        import torch
        from torchvision import datasets, transforms
    except Exception as e:
        raise RuntimeError("This script requires torch and torchvision installed.") from e

    torch.manual_seed(args.seed)

    transform = transforms.ToTensor()
    train_ds = datasets.FashionMNIST(root=args.data_root, train=True, download=True, transform=transform)
    test_ds = datasets.FashionMNIST(root=args.data_root, train=False, download=True, transform=transform)

    in_dim = 784
    hidden = args.hidden_dim
    out_dim = 10

    if args.benchmark_compare:
        run_compare_benchmark(train_ds, args, in_dim, hidden, out_dim)
        return

    train_loader = make_ktorch_loader(train_ds, batch_size=args.batch_size, shuffle=True, seed=args.seed)
    test_loader = make_ktorch_loader(test_ds, batch_size=args.batch_size, shuffle=False, seed=args.seed)

    init_vals = init_params(in_dim, hidden, out_dim)
    w1, b1, w2, b2 = build_params(in_dim, hidden, out_dim, init_vals)

    train_start = time.perf_counter()

    for epoch in range(args.epochs):
        total_loss = 0.0
        total_correct = 0
        total_seen = 0

        for batch_idx, batch in enumerate(train_loader):
            if args.max_train_batches > 0 and batch_idx >= args.max_train_batches:
                break

            loss, correct, bsz = train_step(
                batch,
                w1,
                b1,
                w2,
                b2,
                in_dim,
                out_dim,
                args.lr,
                compute_metrics=True,
            )

            total_loss += loss * bsz
            total_correct += correct
            total_seen += bsz

        train_loss = total_loss / max(total_seen, 1)
        train_acc = total_correct / max(total_seen, 1)
        test_loss, test_acc = evaluate(test_loader, w1, b1, w2, b2, in_dim, out_dim, args.max_test_batches)

        print(
            f"[epoch {epoch + 1}/{args.epochs}] "
            "path=native_ktorch "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"test_loss={test_loss:.4f} test_acc={test_acc:.4f}"
        )

    train_end = time.perf_counter()
    print(f"Training completed in {train_end - train_start:.2f} seconds.")


if __name__ == "__main__":
    main()
