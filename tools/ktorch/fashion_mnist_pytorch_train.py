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
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")


def build_model(in_dim: int, hidden: int, out_dim: int, device, torch):
    import torch.nn as nn

    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(in_dim, hidden),
        nn.ReLU(),
        nn.Linear(hidden, out_dim),
    ).to(device=device)

    with torch.no_grad():
        model[1].weight.uniform_(-0.05, 0.05)
        model[1].bias.zero_()
        model[3].weight.uniform_(-0.05, 0.05)
        model[3].bias.zero_()

    return model


def train_step_device_tensors(images, labels, model, optimizer, F):
    optimizer.zero_grad(set_to_none=True)
    logits = model(images)
    loss = F.cross_entropy(logits, labels)
    loss.backward()
    optimizer.step()


def evaluate_device_tensors(test_images, test_labels, model, batch_size: int, max_batches: int, torch, F) -> Tuple[float, float]:
    total_loss = 0.0
    total_correct = 0
    total_seen = 0
    n = int(test_images.size(0))
    batches_seen = 0

    model.eval()
    with torch.no_grad():
        for start in range(0, n, batch_size):
            if max_batches > 0 and batches_seen >= max_batches:
                break
            end = min(start + batch_size, n)
            images = test_images[start:end]
            labels = test_labels[start:end]
            logits = model(images)
            loss = F.cross_entropy(logits, labels)
            preds = logits.argmax(dim=1)
            bsz = images.size(0)
            total_loss += loss.item() * bsz
            total_correct += (preds == labels).sum().item()
            total_seen += bsz
            batches_seen += 1

    avg_loss = total_loss / max(total_seen, 1)
    acc = total_correct / max(total_seen, 1)
    return avg_loss, acc


def next_index_window(offset: int, total: int, batch_size: int):
    start = offset
    end = min(offset + batch_size, total)
    nxt = 0 if end >= total else end
    return start, end, nxt


def run_step_benchmark(train_images, train_labels, args, in_dim: int, hidden: int, out_dim: int, device, torch, F) -> float:
    model = build_model(in_dim, hidden, out_dim, device, torch)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)

    total = int(train_images.size(0))
    cursor = 0
    measured: List[float] = []

    total_steps = args.benchmark_warmup + args.benchmark_steps
    for step in range(total_steps):
        start, end, cursor = next_index_window(cursor, total, args.batch_size)
        images = train_images[start:end]
        labels = train_labels[start:end]

        torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        train_step_device_tensors(images, labels, model, optimizer, F)
        torch.cuda.synchronize(device)
        dt = time.perf_counter() - t0

        if step >= args.benchmark_warmup:
            measured.append(dt)

    return statistics.median(measured)


def run_compare_benchmark(train_images, train_labels, args, in_dim: int, hidden: int, out_dim: int, device, torch, F) -> None:
    median = run_step_benchmark(train_images, train_labels, args, in_dim, hidden, out_dim, device, torch, F)
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

    if not torch.cuda.is_available():
        raise RuntimeError("This script is CUDA-only. No CUDA device detected.")
    device = torch.device("cuda")

    transform = transforms.ToTensor()
    train_ds = datasets.FashionMNIST(root=args.data_root, train=True, download=True, transform=transform)
    test_ds = datasets.FashionMNIST(root=args.data_root, train=False, download=True, transform=transform)

    in_dim = 784
    hidden = args.hidden_dim
    out_dim = 10

    train_images = train_ds.data.to(device=device, dtype=torch.float32).unsqueeze(1).div_(255.0)
    train_labels = train_ds.targets.to(device=device, dtype=torch.long)
    test_images = test_ds.data.to(device=device, dtype=torch.float32).unsqueeze(1).div_(255.0)
    test_labels = test_ds.targets.to(device=device, dtype=torch.long)

    if args.benchmark_compare:
        run_compare_benchmark(train_images, train_labels, args, in_dim, hidden, out_dim, device, torch, F)
        return

    model = build_model(in_dim, hidden, out_dim, device, torch)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)

    n_train = int(train_images.size(0))
    train_start = time.perf_counter()

    for _epoch in range(args.epochs):
        model.train()
        order = torch.randperm(n_train, device=device)
        batch_idx = 0
        for start in range(0, n_train, args.batch_size):
            if args.max_train_batches > 0 and batch_idx >= args.max_train_batches:
                break
            end = min(start + args.batch_size, n_train)
            idx = order[start:end]
            images = train_images.index_select(0, idx)
            labels = train_labels.index_select(0, idx)
            train_step_device_tensors(images, labels, model, optimizer, F)
            batch_idx += 1

    torch.cuda.synchronize(device)

    train_end = time.perf_counter()
    print(f"Training completed in {train_end - train_start:.2f} seconds.")

    test_loss, test_acc = evaluate_device_tensors(
        test_images, test_labels, model, args.batch_size, args.max_test_batches, torch, F
    )
    print(f"Final test loss: {test_loss:.4f}, test accuracy: {test_acc:.4f}")


if __name__ == "__main__":
    main()
