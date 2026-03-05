# FlashAttention Scratch (CUDA/C++)

## Project goal

Build a minimal-but-real Transformer stack where you own:

1. the model (forward + backward),
2. the attention kernels (naive + FlashAttention-style),
3. correctness + performance evaluation.

Target: CUDA + C++ (optionally a small Python wrapper only for data loading + plotting).

---

## Repo structure

- `src/`
  - `tensor/`: minimal tensor runtime + allocator (GPU + CPU)
  - `ops/`: matmul, layernorm, softmax, dropout, gelu, attention
  - `kernels/`: CUDA kernels (naive + flash)
  - `nn/`: modules: Linear, LayerNorm, MHA, MLP, TransformerBlock
  - `train/`: optimizer, dataloader, training loop
- `tests/`: unit tests + gradient checks
- `bench/`: microbench + throughput benchmarks
- `tools/`: reference implementation helpers (PyTorch validation, plotting)

Notes:

1. Current repository is still early-stage and does not yet contain all folders listed above.
2. The structure above is the target architecture for upcoming milestones.

---

## Build and run

### Configure and build

```bash
cmake -S . -B build
cmake --build build -j
```

### Run correctness tests (GoogleTest)

```bash
tests/run.sh
```

Filter a single gtest case:

```bash
tests/run.sh -- --gtest_filter=TensorCorrectness.H2DAndD2HRoundTrip
```

### Run benchmarks (Google Benchmark)

```bash
bench/run.sh
```

Filter a benchmark:

```bash
bench/run.sh -- --benchmark_filter=BM_CopyD2D
```

If Google Benchmark is missing, CMake will try auto-fetch when `FA_AUTO_FETCH_BENCHMARK=ON`.

---

## Milestone status

- ✅ Milestone 0 — Environment + constraints
- ✅ Milestone 1 — Minimal tensor runtime
- ⬜ Milestone 2 — Core ops for Transformer
- ⬜ Milestone 3 — Working Transformer (naive attention)
- ⬜ Milestone 4 — FlashAttention forward
- ⬜ Milestone 5 — FlashAttention backward
- ⬜ Milestone 6 — Integrate into training
- ⬜ Milestone 7 — Production features

---

## Milestone 0 — Environment + constraints ✅

### Deliverables

- Build system: CMake with CUDA enabled.
- Optional tiny Python harness that can:
  - generate random tensors,
  - call C++/CUDA ops,
  - compare against PyTorch reference.

### Decisions

- Dtype baseline: `fp16/bf16` compute + `fp32` accum for softmax stats.
- Layout: `Q, K, V` in `[B, H, N, D]`, contiguous in `D`.
- Initial operating range: `N <= 2048`, `D in {64, 128}`.

### Exit criteria

- Clean configure/build on dev machine.
- Reproducible command path for test/bench invocation.

---

## Milestone 1 — Minimal tensor runtime ✅

### Deliverables

- `Tensor` class with:
  - shape/strides,
  - dtype/device metadata,
  - contiguous allocation,
  - view/reshape support,
  - device/host copy path.
- Memory:
  - GPU allocator (`cudaMallocAsync` when available, fallback supported),
  - pinned host buffers for H2D/D2H.
- Stream/event wrappers + CUDA error-check macros.

### Tests

- Allocation/free stress test.
- H2D/D2H correctness check.
- Bandwidth benchmark (H2D/D2H/D2D/H2H).

### Exit criteria

- Basic runtime correctness validated by GoogleTest.
- Bench harness runs and reports bandwidth counters.

---

## Milestone 2 — Core ops for Transformer

Implement forward + backward for each op (start naive, optimize later).

### 2.1 GEMM

Option A (recommended): use cuBLAS first.

#### Deliverables

- `linear_forward(X, W, b)` via cuBLAS + bias kernel.
- `linear_backward(dY, X, W)` returning `dX, dW, db`.

### 2.2 LayerNorm

#### Deliverables

- Forward: reduction for mean/var + normalize + affine.
- Backward: numerically correct gradients.

### 2.3 GELU / SwiGLU

#### Deliverables

- Forward and backward kernels.

### 2.4 Dropout

Optional early, required for training milestone.

- RNG strategy: Philox or curand states.
- Deterministic behavior under fixed seed.

### Tests

- Forward compare against PyTorch with dtype-aware tolerances.
- Gradient checks:
  - finite differences on small tensors,
  - compare against PyTorch autograd.

---

## Milestone 3 — Working Transformer (no FlashAttention yet)

### Model

- Token embedding + positional embedding (or RoPE later).
- 1–2 Transformer blocks:
  - MHA (naive attention),
  - MLP,
  - residual + LayerNorm.

### Deliverables

- Inference path matches PyTorch on random tokens.
- Tiny training run overfits small synthetic task.

### Baseline attention (naive)

- Scores: `S = Q @ K^T`, shape `[B, H, N, N]`.
- Softmax on last dim.
- Output: `P @ V`.
- Backward via formula decomposition (GEMMs + softmax backward).

### Bench

- Time + peak memory vs sequence length `N`.

---

## Milestone 4 — FlashAttention forward

Implement streaming attention forward without materializing `N x N`.

### 4.1 Algorithm understanding (write pseudocode)

Core state per row:

- `m_i`: running max,
- `l_i`: running exp sum,
- `o_i`: running output accumulator.

Tiling:

- Tile K/V by `Bc`, process Q by `Br`.
- For each K/V tile:
  - `s = q_tile * k_tile^T`,
  - update `m_i`, rescale `l_i` and `o_i`,
  - accumulate `o_i += exp(s - m_i) @ v_tile`.

### 4.2 CUDA kernel v1 (correctness-first)

#### Kernel design

- One block handles one `(B, H, q_block)` region.
- Shared memory for `K_tile` and `V_tile` (optional `Q_tile`).
- Register accumulators for partial output.

#### Deliverables

- FlashAttention forward for fp16/bf16 inputs.
- Non-causal first; causal mask optional initially.
- Unit test vs naive attention.

#### Bench

- Compare latency vs naive at `N = 256, 512, 1024, 2048`.

### 4.3 Kernel v2 optimization

- Tune `Br/Bc` for occupancy + shared memory.
- Vectorized loads/stores (`half2`, packed types).
- Warp-level reductions (`__shfl_sync`).
- Reduce shared-memory bank conflicts.
- Fuse scale (`1/sqrt(D)`) and masking logic.

---

## Milestone 5 — FlashAttention backward

Backward is substantially harder than forward.

### 5.1 Math strategy

Need `dQ, dK, dV` from `Q, K, V, O, dO` and softmax stats.

Strategies:

1. Store forward stats (`m, l`) for reuse.
2. Recompute in backward (less memory, more compute).

Start with strategy (1).

### 5.2 Backward kernel decomposition

- `dV` via tiled `P^T @ dO`.
- `dP = dO @ V^T`.
- Softmax backward: `dS = P * (dP - sum(dP * P))`.
- `dQ = dS @ K`.
- `dK = dS^T @ Q`.

All tiled, no full `P`/`S` materialization.

### Deliverables

- Correct non-causal backward.
- Add causal support.
- Gradient checks against PyTorch for small shapes (`B=1,H=1,N<=128`).

### Bench

- End-to-end attention forward+backward throughput.

---

## Milestone 6 — Integrate into Transformer training

Replace naive attention path with FlashAttention path.

### Deliverables

- Toy training converges similarly to baseline.
- Peak memory reduction measured.
- Tokens/s improvement demonstrated.

---

## Milestone 7 — Production features (optional, resume-strong)

Pick 3–5 features.

### Feature set A (practical)

- Causal + padding masks with variable lengths (`cu_seqlens`).
- Dropout inside attention with reproducible RNG.
- FP8/int8 experiments (optional advanced).

### Feature set B (kernel craft)

- Persistent-kernel occupancy strategy.
- Split-K for long sequence lengths.
- Prefetch pipelining (`cp.async` where supported).

### Feature set C (usability)

- PyTorch extension wrapper.
- Optional ONNX export for inference path.

---

## Testing strategy (non-negotiable)

### Correctness

- Forward numerics close to PyTorch.
- Backward gradients close to PyTorch.
- Stability tests: large magnitudes, long sequence lengths.

### Determinism

- Fixed seed + dropout should be reproducible.

### Performance

- Microbench: attention kernels.
- Macrobench: full block/model throughput.
- Track:
  - wall time,
  - achieved bandwidth,
  - achieved TFLOPs,
  - peak memory.

---

## Suggested resume-grade final demo

Include a report section in `README.md` with:

- diagrams for tiling + streaming softmax,
- latency vs `N` charts,
- memory vs `N` charts,
- Nsight Compute screenshots showing HBM transaction changes,
- ablations: naive vs flash v1 vs flash v2.

Code quality bar:

- clean launch configurations,
- reproducible benchmark commands,
- CI-runnable correctness tests,
- benchmark CLI like:

```bash
bench_attention --B 4 --H 16 --N 2048 --D 128
```

---

## Next planned implementation order

1. `ops/` core primitives (GEMM + LayerNorm + GELU/SwiGLU)
2. naive attention forward/backward
3. minimal block-level Transformer forward/backward
4. FlashAttention forward v1 then v2
5. FlashAttention backward
6. training integration + report artifacts
