# FlashAttention From Scratch (Milestone 0)

This milestone sets up build + runtime plumbing so CUDA ops can be iterated quickly and validated against PyTorch.

## Decisions captured

- **Dtype:** fp16 / bf16 inputs and outputs, fp32 accumulation for score and softmax stats.
- **Layout:** `Q, K, V` are contiguous `[B, H, N, D]` in row-major order (`D` contiguous).
- **Initial constraints:** `N <= 2048`, `D in {64, 128}`.

## Build (CMake + CUDA)

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

Artifacts:

- `build/libflashattn_ops.so` - shared library exposing `fa_sdpa_forward`
- `build/flashattn_smoke` - tiny smoke executable

## Python harness

The harness generates random CUDA tensors, calls the C++/CUDA op, and compares with PyTorch SDPA.

```bash
python tools/harness.py --dtype fp16 --B 1 --H 4 --N 128 --D 64
python tools/harness.py --dtype bf16 --B 1 --H 4 --N 128 --D 64
```

Notes:

- Requires CUDA-capable PyTorch.
- Current kernel is intentionally naive for correctness scaffolding, not performance.
