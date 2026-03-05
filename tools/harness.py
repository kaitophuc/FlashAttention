#!/usr/bin/env python3
import argparse
import ctypes
from pathlib import Path

import torch
import torch.nn.functional as F


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Milestone 0 harness: compare CUDA op vs PyTorch")
    p.add_argument("--lib", type=Path, default=Path("build/libflashattn_ops.so"))
    p.add_argument("--B", type=int, default=1)
    p.add_argument("--H", type=int, default=4)
    p.add_argument("--N", type=int, default=128)
    p.add_argument("--D", type=int, default=64, choices=[64, 128])
    p.add_argument("--dtype", type=str, default="fp16", choices=["fp16", "bf16"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--atol", type=float, default=5e-2)
    p.add_argument("--rtol", type=float, default=5e-2)
    return p.parse_args()


def load_api(lib_path: Path):
    if not lib_path.exists():
      raise FileNotFoundError(f"Shared library not found: {lib_path}")

    lib = ctypes.CDLL(str(lib_path.resolve()))
    fn = lib.fa_sdpa_forward
    fn.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_void_p,
    ]
    fn.restype = ctypes.c_int
    return fn


def main() -> None:
    args = parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this harness.")

    torch.manual_seed(args.seed)

    dtype = torch.float16 if args.dtype == "fp16" else torch.bfloat16
    dtype_code = 0 if args.dtype == "fp16" else 1

    q = torch.randn(args.B, args.H, args.N, args.D, device="cuda", dtype=dtype)
    k = torch.randn(args.B, args.H, args.N, args.D, device="cuda", dtype=dtype)
    v = torch.randn(args.B, args.H, args.N, args.D, device="cuda", dtype=dtype)

    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()

    out = torch.empty_like(q)

    ref = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=False)

    fa_sdpa_forward = load_api(args.lib)
    stream = torch.cuda.current_stream().cuda_stream

    err = fa_sdpa_forward(
        ctypes.c_void_p(q.data_ptr()),
        ctypes.c_void_p(k.data_ptr()),
        ctypes.c_void_p(v.data_ptr()),
        ctypes.c_void_p(out.data_ptr()),
        ctypes.c_int(args.B),
        ctypes.c_int(args.H),
        ctypes.c_int(args.N),
        ctypes.c_int(args.D),
        ctypes.c_int(dtype_code),
        ctypes.c_void_p(stream),
    )
    if err != 0:
        raise RuntimeError(f"fa_sdpa_forward returned CUDA error code {err}")

    torch.cuda.synchronize()

    max_abs = (out.float() - ref.float()).abs().max().item()
    ok = torch.allclose(out.float(), ref.float(), rtol=args.rtol, atol=args.atol)

    print(f"shape=[{args.B},{args.H},{args.N},{args.D}] dtype={args.dtype}")
    print(f"max_abs_error={max_abs:.6f}")
    print(f"allclose={ok} (rtol={args.rtol}, atol={args.atol})")

    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
