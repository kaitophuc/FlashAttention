#!/usr/bin/env python3
import argparse
import random
import time
from typing import Tuple


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
        "--device",
        default="cuda",
        choices=["cuda", "cpu"],
        help="Execution device preference (falls back to CPU if CUDA unavailable).",
    )
    return p.parse_args()


def set_seed(seed: int, torch) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def evaluate(test_loader, w1, b1, w2, b2, device, max_batches: int, torch, F) -> Tuple[float, float]:
    total_loss = 0.0
    total_correct = 0
    total_seen = 0

    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(test_loader):
            if max_batches > 0 and batch_idx >= max_batches:
                break

            x = images.to(device).view(images.size(0), -1)
            y = labels.to(device)

            z1 = F.linear(x, w1, b1)
            a1 = F.relu(z1)
            logits = F.linear(a1, w2, b2)

            loss = F.cross_entropy(logits, y)
            preds = logits.argmax(dim=1)

            bsz = x.size(0)
            total_loss += loss.item() * bsz
            total_correct += (preds == y).sum().item()
            total_seen += bsz

    avg_loss = total_loss / max(total_seen, 1)
    acc = total_correct / max(total_seen, 1)
    return avg_loss, acc


def main() -> None:
    args = parse_args()

    try:
        import torch
        import torch.nn.functional as F
        from torchvision import datasets, transforms
    except Exception as e:
        raise RuntimeError("This script requires torch and torchvision installed.") from e

    set_seed(args.seed, torch)

    device = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    transform = transforms.ToTensor()
    train_ds = datasets.FashionMNIST(root=args.data_root, train=True, download=True, transform=transform)
    test_ds = datasets.FashionMNIST(root=args.data_root, train=False, download=True, transform=transform)

    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=False)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, drop_last=False)

    in_dim = 784
    hidden = args.hidden_dim
    out_dim = 10

    # Match the ktorch script style: explicit parameter tensors (not nn.Module).
    w1 = (torch.empty(hidden, in_dim, device=device).uniform_(-0.05, 0.05)).requires_grad_(True)
    b1 = torch.zeros(hidden, device=device, requires_grad=True)
    w2 = (torch.empty(out_dim, hidden, device=device).uniform_(-0.05, 0.05)).requires_grad_(True)
    b2 = torch.zeros(out_dim, device=device, requires_grad=True)
    params = [w1, b1, w2, b2]
    optimizer = torch.optim.SGD(params, lr=args.lr)

    main_loop_start = time.perf_counter()

    for epoch in range(args.epochs):
        epoch_start = time.perf_counter()
        total_loss = 0.0
        total_correct = 0
        total_seen = 0

        for batch_idx, (images, labels) in enumerate(train_loader):
            if args.max_train_batches > 0 and batch_idx >= args.max_train_batches:
                break

            x = images.to(device).view(images.size(0), -1)
            y = labels.to(device)

            optimizer.zero_grad(set_to_none=True)

            z1 = F.linear(x, w1, b1)
            a1 = F.relu(z1)
            logits = F.linear(a1, w2, b2)
            loss = F.cross_entropy(logits, y)
            loss.backward()
            optimizer.step()

            preds = logits.argmax(dim=1)
            bsz = x.size(0)
            total_loss += loss.item() * bsz
            total_correct += (preds == y).sum().item()
            total_seen += bsz

        train_loss = total_loss / max(total_seen, 1)
        train_acc = total_correct / max(total_seen, 1)
        test_loss, test_acc = evaluate(test_loader, w1, b1, w2, b2, device, args.max_test_batches, torch, F)

        epoch_end = time.perf_counter()
        epoch_seconds = epoch_end - epoch_start
        samples_per_sec = total_seen / max(epoch_seconds, 1e-9)

        print(
            f"[epoch {epoch + 1}/{args.epochs}] "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"test_loss={test_loss:.4f} test_acc={test_acc:.4f} "
            f"epoch_time={epoch_seconds:.2f}s samples/s={samples_per_sec:.1f}"
        )

    main_loop_end = time.perf_counter()
    print(f"Main training loop time: {main_loop_end - main_loop_start:.2f} seconds")


if __name__ == "__main__":
    main()
