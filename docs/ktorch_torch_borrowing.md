# Ktorch-First PyTorch Borrowing (CPU Borrow, Strict Control)

## 1. Goals and Non-Goals

### Goals
- Use PyTorch only as a CPU data ingestion layer (`DataLoader` workers + batching + optional pinned memory staging).
- Keep all GPU transfer and compute control inside `ktorch`.
- Borrow CPU tensor memory from PyTorch without copy at the borrow boundary.
- Enforce strict immutability checks so borrowed sources are not modified while `ktorch` is using them.

### Non-Goals (v1)
- No CUDA tensor borrowing from PyTorch.
- No PyTorch-driven CUDA stream/event orchestration for model compute.
- No shared autograd graph between frameworks.

## 2. Ownership and Lifetime Model

`Tensor` now supports two storage modes:
- `Owned`: memory allocated/freed by `ktorch`.
- `BorrowedExternal`: memory pointer owned externally (PyTorch in this design).

Borrowed tensors are configured as read-only in v1. They keep a strong owner token to the source PyTorch tensor so host memory cannot be released while `ktorch` still references it.

### Destruction behavior
- Owned tensor: normal `ktorch` free path.
- Borrowed tensor: never frees external data pointer.

## 3. Public API Contracts

### `ktorch.from_torch_borrow_cpu(tensor, require_contiguous=True, require_pinned=True)`
Creates a CPU `ktorch.Tensor` view over a PyTorch tensor without copying.

Requirements:
- Input must be `torch.Tensor` on CPU.
- Supported dtypes: `torch.float32`, `torch.int32`.
- Contiguous required by default.
- Pinned required by default.

Failure cases:
- CUDA tensor input.
- Non-contiguous tensor when `require_contiguous=True`.
- Non-pinned tensor when `require_pinned=True`.
- Unsupported dtype.

### `ktorch.copy_cpu_to_cuda_async(src_cpu, dst_cuda, stream, strict_immutability=True)`
Submits async H2D copy on the provided `ktorch` stream.

Requirements:
- `src_cpu.device == CPU`, `dst_cuda.device == CUDA`.
- Shape/dtype match.
- Contiguous source and destination.

Strict immutability behavior:
- For torch-borrowed source tensors, validates source `_version` before and immediately after copy submission.
- Throws if source mutation is detected.

### Stream event APIs
- `ktorch.Event()`
- `ktorch.record_event(event, stream)`
- `ktorch.wait_event(stream, event)`

These APIs are used to gate compute stream execution on copy completion without global synchronization.

## 4. Stream and Synchronization Rules

### Rule set
- Data loading: PyTorch CPU workers only.
- H2D transfer: `ktorch` copy stream only.
- Compute: `ktorch` compute stream only.
- Inter-stream order: explicit event handoff (`record_event` on copy stream, `wait_event` on compute stream).
- Avoid per-batch `synchronize()` in hot path.

### Typical timeline
1. PyTorch DataLoader yields pinned CPU tensors.
2. `ktorch.from_torch_borrow_cpu` borrows CPU memory.
3. `ktorch.copy_cpu_to_cuda_async` submits copies on copy stream.
4. `ktorch.record_event(copy_done, copy_stream)`.
5. `ktorch.wait_event(compute_stream, copy_done)`.
6. Model ops run on compute stream.

## 5. PyTorch Role Policy

Allowed PyTorch responsibilities:
- Dataset IO and parsing.
- CPU transforms/collation.
- Worker multiprocessing (`num_workers`).
- Producing pinned CPU batches.

Forbidden PyTorch responsibilities for training core:
- Model forward/backward/optimizer.
- GPU stream ordering for `ktorch` kernels.
- Mutating borrowed tensors after borrow boundary and before copy completion.

## 6. Usage Example (Manual FashionMNIST Trainer)

High-level flow:
- DataLoader configured with `pin_memory=True` and int32 label collation.
- For each batch:
  - Borrow CPU tensors with `from_torch_borrow_cpu`.
  - Allocate CUDA destination tensors in `ktorch`.
  - Submit async H2D copies on `copy_stream`.
  - Record event on `copy_stream`.
  - Wait on `compute_stream`.
  - Run `ktorch` forward/backward/update.

## 7. Troubleshooting

- Error: tensor must be pinned
  - Ensure DataLoader uses `pin_memory=True`.
- Error: dtype unsupported
  - Use `float32` for images and `int32` for labels before borrowing.
- Error: borrowed tensor mutated
  - Remove in-place ops on the source torch tensor after borrowing.
- Throughput not improving
  - Check that per-batch synchronizations are removed from hot path.
  - Verify copy/compute streams are distinct and event handoff is used.
