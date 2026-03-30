#!/usr/bin/env python3
import argparse
import random
import statistics
import time
from typing import List, Tuple


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train a tiny FashionMNIST MLP with PyTorch.")
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
        help="Run deterministic train-step benchmark (PyTorch path) and exit.",
    )
    p.add_argument("--benchmark-warmup", type=int, default=20, help="Warmup steps for benchmark mode.")
    p.add_argument("--benchmark-steps", type=int, default=100, help="Measured steps for benchmark mode.")
    return p.parse_args()


def set_seed(seed: int, torch) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def configure_torch_cuda(torch) -> None:
    if not torch.cuda.is_available():
        return

    # Reasonable performance defaults for CUDA baseline runs.
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")


def init_params(in_dim: int, hidden: int, out_dim: int, device, torch):
    w1 = torch.empty(hidden, in_dim, device=device).uniform_(-0.05, 0.05).requires_grad_(True)
    b1 = torch.zeros(hidden, device=device, requires_grad=True)
    w2 = torch.empty(out_dim, hidden, device=device).uniform_(-0.05, 0.05).requires_grad_(True)
    b2 = torch.zeros(out_dim, device=device, requires_grad=True)
    return w1, b1, w2, b2


def make_loader(dataset, batch_size: int, shuffle: bool, seed: int, use_cuda: bool, torch):
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


def move_batch(images, labels, device, use_cuda: bool):
    if use_cuda:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
    else:
        images = images.to(device)
        labels = labels.to(device)
    return images, labels


def forward_logits(images, w1, b1, w2, b2, F):
    x = images.view(images.size(0), -1)
    z1 = F.linear(x, w1, b1)
    a1 = F.relu(z1)
    logits = F.linear(a1, w2, b2)
    return logits


def train_step(batch, w1, b1, w2, b2, optimizer, device, use_cuda: bool, F):
    images, labels = batch
    images, labels = move_batch(images, labels, device, use_cuda)

    optimizer.zero_grad(set_to_none=True)
    logits = forward_logits(images, w1, b1, w2, b2, F)
    loss = F.cross_entropy(logits, labels)
    loss.backward()
    optimizer.step()


def evaluate(test_loader, w1, b1, w2, b2, device, max_batches: int, use_cuda: bool, torch, F) -> Tuple[float, float]:
    total_loss = 0.0
    total_correct = 0
    total_seen = 0

    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(test_loader):
            if max_batches > 0 and batch_idx >= max_batches:
                break

            images, labels = move_batch(images, labels, device, use_cuda)
            logits = forward_logits(images, w1, b1, w2, b2, F)

            loss = F.cross_entropy(logits, labels)
            preds = logits.argmax(dim=1)

            bsz = images.size(0)
            total_loss += loss.item() * bsz
            total_correct += (preds == labels).sum().item()
            total_seen += bsz

    avg_loss = total_loss / max(total_seen, 1)
    acc = total_correct / max(total_seen, 1)
    return avg_loss, acc


def next_batch(it, loader):
    try:
        return next(it), it
    except StopIteration:
        it = iter(loader)
        return next(it), it


def run_step_benchmark(train_loader, args, in_dim: int, hidden: int, out_dim: int, device, use_cuda: bool, torch, F) -> float:
    w1, b1, w2, b2 = init_params(in_dim, hidden, out_dim, device, torch)
    optimizer = torch.optim.SGD([w1, b1, w2, b2], lr=args.lr)

    loader_it = iter(train_loader)
    measured: List[float] = []

    total_steps = args.benchmark_warmup + args.benchmark_steps
    for step in range(total_steps):
        batch, loader_it = next_batch(loader_it, train_loader)

        if use_cuda:
            torch.cuda.synchronize(device)
        t0 = time.perf_counter()

        train_step(batch, w1, b1, w2, b2, optimizer, device, use_cuda, F)

        if use_cuda:
            torch.cuda.synchronize(device)
        dt = time.perf_counter() - t0

        if step >= args.benchmark_warmup:
            measured.append(dt)

    return statistics.median(measured)


def run_compare_benchmark(train_ds, args, in_dim: int, hidden: int, out_dim: int, device, use_cuda: bool, torch, F) -> None:
    loader = make_loader(train_ds, batch_size=args.batch_size, shuffle=False, seed=args.seed, use_cuda=use_cuda, torch=torch)
    median = run_step_benchmark(loader, args, in_dim, hidden, out_dim, device, use_cuda, torch, F)
    print("[benchmark] train-step median (batch_size=%d, warmup=%d, steps=%d)" % (
        args.batch_size,
        args.benchmark_warmup,
        args.benchmark_steps,
    ))
    print(f"[benchmark] pytorch={median * 1000.0:.3f} ms")


def main() -> None:
    args = parse_args()

    try:
        import torch
        import torch.nn.functional as F
        from torchvision import datasets, transforms
    except Exception as e:
        raise RuntimeError("This script requires torch and torchvision installed.") from e

    set_seed(args.seed, torch)
    configure_torch_cuda(torch)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    transform = transforms.ToTensor()
    train_ds = datasets.FashionMNIST(root=args.data_root, train=True, download=True, transform=transform)
    test_ds = datasets.FashionMNIST(root=args.data_root, train=False, download=True, transform=transform)

    in_dim = 784
    hidden = args.hidden_dim
    out_dim = 10

    if args.benchmark_compare:
        run_compare_benchmark(train_ds, args, in_dim, hidden, out_dim, device, use_cuda, torch, F)
        return

    train_loader = make_loader(train_ds, batch_size=args.batch_size, shuffle=True, seed=args.seed, use_cuda=use_cuda, torch=torch)
    test_loader = make_loader(test_ds, batch_size=args.batch_size, shuffle=False, seed=args.seed, use_cuda=use_cuda, torch=torch)

    w1, b1, w2, b2 = init_params(in_dim, hidden, out_dim, device, torch)
    optimizer = torch.optim.SGD([w1, b1, w2, b2], lr=args.lr)

    train_start = time.perf_counter()

    for epoch in range(args.epochs):
        for batch_idx, batch in enumerate(train_loader):
            if args.max_train_batches > 0 and batch_idx >= args.max_train_batches:
                break

            train_step(batch, w1, b1, w2, b2, optimizer, device, use_cuda, F)

    if use_cuda:
        torch.cuda.synchronize(device)

    train_end = time.perf_counter()
    print(f"Training completed in {train_end - train_start:.2f} seconds.")

    test_loss, test_acc = evaluate(test_loader, w1, b1, w2, b2, device, args.max_test_batches, use_cuda, torch, F)
    print(f"Final test loss: {test_loss:.4f}, test accuracy: {test_acc:.4f}")


if __name__ == "__main__":
    main()
