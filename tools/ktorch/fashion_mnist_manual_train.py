#!/usr/bin/env python3
import argparse
import math
import random
import time
from typing import List, Sequence, Tuple

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
    return p.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)


def make_tensor(shape: Sequence[int], values: Sequence[float]) -> ktorch.Tensor:
    t = ktorch.Tensor(list(shape), dtype=ktorch.DType.F32, device=ktorch.Device.CUDA)
    t.copy_from_list_float(list(values))
    return t


def flatten_images_torch_to_list(images) -> List[float]:
    # images: [B, 1, 28, 28] torch tensor
    # Normalize to [0, 1] float and flatten row-major to [B * 784].
    flat = images.reshape(images.shape[0], -1).float()
    return flat.tolist()  # list[list[float]]


def random_init(rows: int, cols: int, scale: float) -> List[float]:
    vals = []
    for _ in range(rows * cols):
        vals.append(random.uniform(-1.0, 1.0) * scale)
    return vals


def zeros(n: int) -> List[float]:
    return [0.0] * n


def rowwise_softmax(logits: List[float], batch_size: int, num_classes: int) -> List[float]:
    probs = [0.0] * (batch_size * num_classes)
    for i in range(batch_size):
        base = i * num_classes
        row = logits[base:base + num_classes]
        m = max(row)
        exps = [math.exp(v - m) for v in row]
        s = sum(exps)
        for j in range(num_classes):
            probs[base + j] = exps[j] / s
    return probs


def cross_entropy_and_grad(
    logits: List[float],
    labels: Sequence[int],
    batch_size: int,
    num_classes: int,
) -> Tuple[float, List[float], int]:
    probs = rowwise_softmax(logits, batch_size, num_classes)
    grad = [0.0] * (batch_size * num_classes)

    eps = 1e-12
    loss = 0.0
    correct = 0

    for i in range(batch_size):
        base = i * num_classes
        y = int(labels[i])
        p_y = max(probs[base + y], eps)
        loss -= math.log(p_y)

        pred = 0
        best = probs[base]
        for j in range(1, num_classes):
            pj = probs[base + j]
            if pj > best:
                best = pj
                pred = j
        if pred == y:
            correct += 1

        for j in range(num_classes):
            g = probs[base + j]
            if j == y:
                g -= 1.0
            grad[base + j] = g / batch_size

    return loss / batch_size, grad, correct


def sgd_update(param: ktorch.Tensor, grad: ktorch.Tensor, lr: float) -> None:
    p = param.to_list_float()
    g = grad.to_list_float()
    updated = [pv - lr * gv for pv, gv in zip(p, g)]
    param.copy_from_list_float(updated)


def evaluate(test_loader, w1, b1, w2, b2, max_batches: int) -> Tuple[float, float]:
    total_loss = 0.0
    total_correct = 0
    total_seen = 0

    for batch_idx, (images, labels) in enumerate(test_loader):
        if max_batches > 0 and batch_idx >= max_batches:
            break

        x_rows = flatten_images_torch_to_list(images)
        bsz = len(x_rows)
        x_flat = [v for row in x_rows for v in row]

        x = make_tensor([bsz, 784], x_flat)
        z1, _ = ops.linear_forward(x, w1, b1)
        a1, _ = ops.relu_forward(z1)
        z2, _ = ops.linear_forward(a1, w2, b2)

        logits = z2.to_list_float()
        loss, _, correct = cross_entropy_and_grad(logits, labels.tolist(), bsz, 10)

        total_loss += loss * bsz
        total_correct += correct
        total_seen += bsz

    avg_loss = total_loss / max(total_seen, 1)
    acc = total_correct / max(total_seen, 1)
    return avg_loss, acc


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    try:
        import torch
        from torchvision import datasets, transforms
    except Exception as e:
        raise RuntimeError("This script requires torch and torchvision installed.") from e

    transform = transforms.ToTensor()
    train_ds = datasets.FashionMNIST(root=args.data_root, train=True, download=True, transform=transform)
    test_ds = datasets.FashionMNIST(root=args.data_root, train=False, download=True, transform=transform)

    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=False)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, drop_last=False)

    in_dim = 784
    hidden = args.hidden_dim
    out_dim = 10

    w1 = make_tensor([hidden, in_dim], random_init(hidden, in_dim, scale=0.05))
    b1 = make_tensor([hidden], zeros(hidden))
    w2 = make_tensor([out_dim, hidden], random_init(out_dim, hidden, scale=0.05))
    b2 = make_tensor([out_dim], zeros(out_dim))

    train_start = time.perf_counter()

    for epoch in range(args.epochs):
        total_loss = 0.0
        total_correct = 0
        total_seen = 0

        for batch_idx, (images, labels) in enumerate(train_loader):
            if args.max_train_batches > 0 and batch_idx >= args.max_train_batches:
                break

            x_rows = flatten_images_torch_to_list(images)
            bsz = len(x_rows)
            x_flat = [v for row in x_rows for v in row]
            y = labels.tolist()

            x = make_tensor([bsz, in_dim], x_flat)

            z1, ctx1 = ops.linear_forward(x, w1, b1)
            a1, relu_ctx = ops.relu_forward(z1)
            z2, ctx2 = ops.linear_forward(a1, w2, b2)

            logits = z2.to_list_float()
            loss, dz2_vals, correct = cross_entropy_and_grad(logits, y, bsz, out_dim)
            dz2 = make_tensor([bsz, out_dim], dz2_vals)

            g2 = ops.linear_backward(dz2, ctx2, True, True, True)
            dz1 = ops.relu_backward(g2.dX, relu_ctx)
            g1 = ops.linear_backward(dz1, ctx1, False, True, True)

            sgd_update(w2, g2.dW, args.lr)
            sgd_update(b2, g2.db, args.lr)
            sgd_update(w1, g1.dW, args.lr)
            sgd_update(b1, g1.db, args.lr)

            total_loss += loss * bsz
            total_correct += correct
            total_seen += bsz

        train_loss = total_loss / max(total_seen, 1)
        train_acc = total_correct / max(total_seen, 1)
        test_loss, test_acc = evaluate(test_loader, w1, b1, w2, b2, args.max_test_batches)

        print(
            f"[epoch {epoch + 1}/{args.epochs}] "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"test_loss={test_loss:.4f} test_acc={test_acc:.4f}"
        )

    train_end = time.perf_counter()
    print(f"Training completed in {train_end - train_start:.2f} seconds.")


if __name__ == "__main__":
    main()
