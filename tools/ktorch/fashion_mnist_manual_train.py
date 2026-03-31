#!/usr/bin/env python3
import argparse
import statistics
import time
from typing import List, Tuple

import ktorch
from ktorch import ops


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


def init_model_params(seed: int, in_dim: int, hidden: int, out_dim: int):
    w1 = ktorch.random_uniform(
        [hidden, in_dim],
        low=-0.05,
        high=0.05,
        seed=seed + 11,
        dtype=ktorch.DType.F32,
        device=ktorch.Device.CUDA,
    )
    b1 = ktorch.zeros([hidden], dtype=ktorch.DType.F32, device=ktorch.Device.CUDA)
    w2 = ktorch.random_uniform(
        [out_dim, hidden],
        low=-0.05,
        high=0.05,
        seed=seed + 23,
        dtype=ktorch.DType.F32,
        device=ktorch.Device.CUDA,
    )
    b2 = ktorch.zeros([out_dim], dtype=ktorch.DType.F32, device=ktorch.Device.CUDA)
    return w1, b1, w2, b2


def next_batch(it, loader):
    try:
        return next(it), it
    except StopIteration:
        it = iter(loader)
        return next(it), it


def torch_batch_to_ktorch(batch, use_cuda: bool, torch):
    images, labels = batch
    x = ktorch.from_torch(images, device=ktorch.Device.CUDA)
    y = ktorch.from_torch(labels, device=ktorch.Device.CUDA)
    return x, y


def train_step(batch, w1, b1, w2, b2, in_dim: int, lr: float):
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

    correct_t = ops.classification_correct_count(z2, labels_t)

    return loss_t, correct_t


def evaluate(test_loader, w1, b1, w2, b2, in_dim: int, max_batches: int, use_cuda: bool, torch) -> Tuple[float, float]:
    total_loss = 0.0
    total_correct = 0
    total_seen = 0

    for batch_idx, batch in enumerate(test_loader):
        if max_batches > 0 and batch_idx >= max_batches:
            break

        x, labels_t = torch_batch_to_ktorch(batch, use_cuda, torch)
        bsz = int(x.shape[0])
        x.view([bsz, in_dim])

        z1, _ = ops.linear_forward(x, w1, b1)
        a1, _ = ops.relu_forward(z1)
        z2, _ = ops.linear_forward(a1, w2, b2)

        loss_t, _ = ops.softmax_cross_entropy_forward(z2, labels_t)

        loss = float(loss_t.item_float())
        correct = int(ops.classification_correct_count(z2, labels_t).item_int32())

        total_loss += loss * bsz
        total_correct += correct
        total_seen += bsz

    avg_loss = total_loss / max(total_seen, 1)
    acc = total_correct / max(total_seen, 1)
    return avg_loss, acc


def make_torch_loader(dataset, batch_size: int, shuffle: bool, seed: int, use_cuda: bool, torch):
    generator = torch.Generator()
    generator.manual_seed(seed)
    num_workers = 4 if use_cuda else 0
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=use_cuda,
        persistent_workers=(num_workers > 0),
        generator=generator,
    )


def run_step_benchmark(train_loader, args, in_dim: int, hidden: int, out_dim: int, use_cuda: bool, torch) -> float:
    w1, b1, w2, b2 = init_model_params(args.seed, in_dim, hidden, out_dim)
    loader_it = iter(train_loader)
    measured: List[float] = []

    total_steps = args.benchmark_warmup + args.benchmark_steps
    for step in range(total_steps):
        batch, loader_it = next_batch(loader_it, train_loader)
        kt_batch = torch_batch_to_ktorch(batch, use_cuda, torch)
        t0 = time.perf_counter()
        train_step(kt_batch, w1, b1, w2, b2, in_dim, args.lr)
        dt = time.perf_counter() - t0
        if step >= args.benchmark_warmup:
            measured.append(dt)

    return statistics.median(measured)


def run_compare_benchmark(train_ds, args, in_dim: int, hidden: int, out_dim: int) -> None:
    try:
        import torch
    except Exception as e:
        raise RuntimeError("Benchmark mode requires torch installed.") from e
    use_cuda = torch.cuda.is_available()
    loader = make_torch_loader(train_ds, batch_size=args.batch_size, shuffle=False, seed=args.seed, use_cuda=use_cuda, torch=torch)
    median = run_step_benchmark(loader, args, in_dim, hidden, out_dim, use_cuda, torch)
    print("[benchmark] train-step median (batch_size=%d, warmup=%d, steps=%d)" % (
        args.batch_size,
        args.benchmark_warmup,
        args.benchmark_steps,
    ))
    print(f"[benchmark] native_ktorch={median * 1000.0:.3f} ms")


def main() -> None:
    args = parse_args()

    try:
        import torch
        from torchvision import datasets, transforms
    except Exception as e:
        raise RuntimeError("This script requires torch and torchvision installed.") from e

    torch.manual_seed(args.seed)
    use_cuda = torch.cuda.is_available()

    transform = transforms.ToTensor()
    train_ds = datasets.FashionMNIST(root=args.data_root, train=True, download=True, transform=transform)
    test_ds = datasets.FashionMNIST(root=args.data_root, train=False, download=True, transform=transform)

    in_dim = 784
    hidden = args.hidden_dim
    out_dim = 10

    if args.benchmark_compare:
        run_compare_benchmark(train_ds, args, in_dim, hidden, out_dim)
        return

    train_loader = make_torch_loader(train_ds, batch_size=args.batch_size, shuffle=True, seed=args.seed, use_cuda=use_cuda, torch=torch)
    test_loader = make_torch_loader(test_ds, batch_size=args.batch_size, shuffle=False, seed=args.seed, use_cuda=use_cuda, torch=torch)

    w1, b1, w2, b2 = init_model_params(args.seed, in_dim, hidden, out_dim)

    train_start = time.perf_counter()

    for epoch in range(args.epochs):
        for batch_idx, batch in enumerate(train_loader):
            if args.max_train_batches > 0 and batch_idx >= args.max_train_batches:
                break
            kt_batch = torch_batch_to_ktorch(batch, use_cuda, torch)

            loss_t, correct_t = train_step(
                kt_batch,
                w1,
                b1,
                w2,
                b2,
                in_dim,
                args.lr,
            )
            if correct_t is None:
                raise RuntimeError("internal error: compute_metrics=True must return correct_t tensor")
            _ = loss_t

    ktorch.synchronize()  # Ensure all GPU work is done before stopping the timer.

    train_end = time.perf_counter()
    print(f"Training completed in {train_end - train_start:.2f} seconds.")

    test_loss, test_acc = evaluate(test_loader, w1, b1, w2, b2, in_dim, args.max_test_batches, use_cuda, torch)
    print(f"Final test loss: {test_loss:.4f}, test accuracy: {test_acc:.4f}")


if __name__ == "__main__":
    main()
