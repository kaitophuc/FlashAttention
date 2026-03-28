# ktorch Implementation Notes

## Confirmed Assumptions
- **Python API is the primary surface for v1.**
  Impact: binding ergonomics and module structure prioritize Python-first usage over a polished C++ consumer SDK.
  Recommended next action: add a dedicated C++ examples folder if external C++ users become a target.
- **v1 supports only currently implemented ops (`linear`, `layernorm`, `relu`, `softmax`).**
  Impact: no dropout in bindings yet, and MLP smoke avoids dropout.
  Recommended next action: bind dropout once `src/ops_dropout.cu` is implemented.
- **v1 host debug I/O is list-based (`copy_from_list_float` / `to_list_float`) and F32-only.**
  Impact: easy correctness debugging but limited dtype coverage and potentially slow transfers for large tensors.
  Recommended next action: add typed host buffers and optional fast-path I/O APIs.
- **Default-stream-only behavior is preserved intentionally.**
  Impact: Python bindings inherit current stream restriction from core runtime and kernels.
  Recommended next action: introduce explicit stream object support after stream-safe audits.

## Low-Confidence Areas
- **pybind11 packaging discoverability across environments.**
  Why uncertain: some systems provide pybind11 only as Python package, others via CMake package paths.
  Validation test: `pip install -e .` on target machine and confirm `_C` module builds and imports.
  Recommended next action: add fallback logic for `pybind11` CMake discovery if portability issues appear.
- **C++ context ownership safety across all future API refactors.**
  Why uncertain: core contexts use raw tensor pointers; binding wrappers patch and own referenced tensors.
  Validation test: stress test repeated forward/backward while deleting Python references to intermediates.
  Recommended next action: migrate core context structs from raw pointers to safer ownership/handles.
- **Manual training smoke script stability on diverse CUDA drivers/architectures.**
  Why uncertain: script relies on iterative host read/write parameter updates.
  Validation test: run `tools/ktorch/mlp_manual_train.py` on each supported GPU generation.
  Recommended next action: add in-device optimizer ops and reduce host round-trips.

## Known Risks / Concerns
- **Ownership/lifetime risk:** core `LinearCtx`, `LayerNormCtx`, `ReluCtx` use raw pointers.
  Impact: dangling-pointer risk if wrappers ever fail to keep originating tensors alive.
  Recommended next action: replace pointer fields with stable tensor handles or intrusive refs in core.
- **Stream model risk:** runtime enforces default stream in many code paths.
  Impact: users cannot yet integrate safely with multi-stream execution models.
  Recommended next action: design stream-aware APIs and comprehensive stream-order tests before enabling.
- **DType limitation risk:** most bound ops are currently F32-only.
  Impact: lower performance than mixed precision paths and no BF16/F16 Python-level parity.
  Recommended next action: extend kernels and tests for BF16/F16, then update binding validators.
- **Performance caveat:** list-based host I/O copies frequently and synchronizes.
  Impact: high overhead in training loops, currently acceptable only for correctness/smoke workflows.
  Recommended next action: add binary buffer I/O and optional pinned host staging utilities.
- **ABI/toolchain coupling:** extension links to CUDA/cuBLAS and local compiler toolchain details.
  Impact: wheel portability is limited; editable source build is currently the supported path.
  Recommended next action: document supported toolchain matrix and consider manylinux packaging later.

## Improvement Opportunities
- Implement autograd engine for dynamic graphs and automatic gradient propagation.
- Add `dropout` CUDA implementation and bind it in Python once correctness tests pass.
- Support non-default stream objects in runtime and Python bindings.
- Expand tensor host I/O beyond float list APIs (typed arrays, memoryview/buffer protocol).
- Add F16/BF16 end-to-end tests and dtype-dispatch in bindings.
- Add a minimal optimizer module in `ktorch` to avoid host-side parameter updates in scripts.

## Decision Log
- **2026-03-28:** Chose direct `pybind11` binding of existing C++ `Tensor` as the v1 data boundary.
  Tradeoff: fastest integration with current runtime vs more ABI-sensitive extension build.
- **2026-03-28:** Chose explicit/manual backward APIs over autograd in v1.
  Tradeoff: lower framework complexity now vs less ergonomic model training code.
- **2026-03-28:** Chose to omit NumPy/PyTorch bridges in v1.
  Tradeoff: clean framework independence vs limited external interoperability initially.
- **2026-03-28:** Chose `scikit-build-core` editable workflow.
  Tradeoff: modern packaging and CMake integration vs extra build dependency setup.

## Open Questions For You
- Do you want CPU execution support for ops in Python soon, or keep Python API CUDA-only for now?
- Should we prioritize autograd engine next, or broaden op/dtype support first?
- For host I/O evolution, do you prefer a Python buffer-protocol path or a lightweight custom binary tensor format?
- Do you want `ktorch` package naming/versioning to stay in this repo root, or move to a dedicated subproject directory next?
