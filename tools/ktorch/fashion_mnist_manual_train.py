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
    p.add_argument(
        "--strict-borrow-immutability",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Validate borrowed torch tensors were not mutated while owned by ktorch copy boundary.",
    )
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


def _iter_limited_batches(loader, max_batches: int):
    for idx, batch in enumerate(loader):
        if max_batches > 0 and idx >= max_batches:
            break
        yield batch


def _slot_tensor_matches(t, shape, dtype) -> bool:
    return t is not None and list(t.shape) == list(shape) and t.dtype == dtype


def _schedule_batch_to_slot(slot, batch, copy_stream, strict_borrow_immutability: bool):
    images, labels = batch
    x_cpu = ktorch.from_torch_borrow_cpu(images, require_contiguous=True, require_pinned=True)
    y_cpu = ktorch.from_torch_borrow_cpu(labels, require_contiguous=True, require_pinned=True)

    with ktorch.stream_guard(copy_stream):
        # If this slot was used by prior compute, do not overwrite its GPU buffers
        # until that compute stream work is fully complete.
        if slot.get("compute_done_recorded", False):
            ktorch.wait_event(copy_stream, slot["compute_done_event"])

        if not _slot_tensor_matches(slot.get("x_gpu"), x_cpu.shape, x_cpu.dtype):
            slot["x_gpu"] = ktorch.empty(list(x_cpu.shape), dtype=x_cpu.dtype, device=ktorch.Device.CUDA)
        if not _slot_tensor_matches(slot.get("y_gpu"), y_cpu.shape, y_cpu.dtype):
            slot["y_gpu"] = ktorch.empty(list(y_cpu.shape), dtype=y_cpu.dtype, device=ktorch.Device.CUDA)

        # Keep borrowed CPU sources alive across async copy completion.
        slot["x_cpu_borrow"] = x_cpu
        slot["y_cpu_borrow"] = y_cpu
        slot["x_gpu"].copy_from(x_cpu, copy_stream, strict_borrow_immutability)
        slot["y_gpu"].copy_from(y_cpu, copy_stream, strict_borrow_immutability)

        if slot.get("ready_event") is None:
            slot["ready_event"] = ktorch.Event()
        ktorch.record_event(slot["ready_event"], copy_stream)


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


def evaluate(
    test_loader,
    w1,
    b1,
    w2,
    b2,
    in_dim: int,
    max_batches: int,
    copy_stream,
    compute_stream,
    strict_borrow_immutability: bool,
) -> Tuple[float, float]:
    total_loss = 0.0
    total_correct = 0
    total_seen = 0

    for batch_idx, batch in enumerate(test_loader):
        if max_batches > 0 and batch_idx >= max_batches:
            break

        slot = {}
        _schedule_batch_to_slot(slot, batch, copy_stream, strict_borrow_immutability)
        ktorch.wait_event(compute_stream, slot["ready_event"])
        x = slot["x_gpu"]
        labels_t = slot["y_gpu"]
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
    def collate_fn(samples):
        images, labels = zip(*samples)
        images_t = torch.stack(images, dim=0)
        labels_t = torch.as_tensor(labels, dtype=torch.int32)
        return images_t, labels_t

    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=use_cuda,
        persistent_workers=(num_workers > 0),
        generator=generator,
        collate_fn=collate_fn,
    )


def run_step_benchmark(train_loader, args, in_dim: int, hidden: int, out_dim: int, use_cuda: bool, torch) -> float:
    w1, b1, w2, b2 = init_model_params(args.seed, in_dim, hidden, out_dim)
    loader_it = iter(train_loader)
    measured: List[float] = []
    copy_stream = ktorch.next_stream()
    compute_stream = ktorch.current_stream()
    slots = [{}, {}, {}]
    cur = 0

    total_steps = args.benchmark_warmup + args.benchmark_steps
    first_batch, loader_it = next_batch(loader_it, train_loader)
    _schedule_batch_to_slot(slots[cur], first_batch, copy_stream, args.strict_borrow_immutability)

    for step in range(total_steps):
        next_batch_data, loader_it = next_batch(loader_it, train_loader)
        nxt = (cur + 1) % len(slots)
        _schedule_batch_to_slot(slots[nxt], next_batch_data, copy_stream, args.strict_borrow_immutability)

        ktorch.wait_event(compute_stream, slots[cur]["ready_event"])
        t0 = time.perf_counter()
        train_step((slots[cur]["x_gpu"], slots[cur]["y_gpu"]), w1, b1, w2, b2, in_dim, args.lr)
        if slots[cur].get("compute_done_event") is None:
            slots[cur]["compute_done_event"] = ktorch.Event()
        ktorch.record_event(slots[cur]["compute_done_event"], compute_stream)
        slots[cur]["compute_done_recorded"] = True
        slots[cur]["x_cpu_borrow"] = None
        slots[cur]["y_cpu_borrow"] = None
        dt = time.perf_counter() - t0
        if step >= args.benchmark_warmup:
            measured.append(dt)
        cur = nxt

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
    copy_stream = ktorch.next_stream()
    compute_stream = ktorch.current_stream()

    train_start = time.perf_counter()

    for epoch in range(args.epochs):
        slots = [{}, {}, {}]
        cur = 0
        batch_iter = iter(_iter_limited_batches(train_loader, args.max_train_batches))

        first_batch = next(batch_iter, None)
        if first_batch is None:
            continue
        # Pipeline warmup: prefetch the first batch before timing hot steps.
        _schedule_batch_to_slot(slots[cur], first_batch, copy_stream, args.strict_borrow_immutability)

        while True:
            nxt = (cur + 1) % len(slots)
            next_batch_data = next(batch_iter, None)
            has_next = next_batch_data is not None
            if has_next:
                _schedule_batch_to_slot(slots[nxt], next_batch_data, copy_stream, args.strict_borrow_immutability)

            ktorch.wait_event(compute_stream, slots[cur]["ready_event"])

            loss_t, correct_t = train_step(
                (slots[cur]["x_gpu"], slots[cur]["y_gpu"]),
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
            if slots[cur].get("compute_done_event") is None:
                slots[cur]["compute_done_event"] = ktorch.Event()
            ktorch.record_event(slots[cur]["compute_done_event"], compute_stream)
            slots[cur]["compute_done_recorded"] = True
            slots[cur]["x_cpu_borrow"] = None
            slots[cur]["y_cpu_borrow"] = None
            if not has_next:
                break
            cur = nxt

    ktorch.synchronize()  # Ensure all GPU work is done before stopping the timer.

    train_end = time.perf_counter()
    print(f"Training completed in {train_end - train_start:.2f} seconds.")

    test_loss, test_acc = evaluate(
        test_loader,
        w1,
        b1,
        w2,
        b2,
        in_dim,
        args.max_test_batches,
        copy_stream,
        compute_stream,
        args.strict_borrow_immutability,
    )
    print(f"Final test loss: {test_loss:.4f}, test accuracy: {test_acc:.4f}")


if __name__ == "__main__":
    main()
