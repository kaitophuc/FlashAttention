#!/usr/bin/env python3
import argparse
import csv
import os
import sys
from dataclasses import dataclass


@dataclass
class LinearCase:
    name: str
    m: int
    k: int
    n: int
    bias: int


def load_cases(path: str):
    cases = []
    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            cases.append(
                LinearCase(
                    name=row["name"],
                    m=int(row["m"]),
                    k=int(row["k"]),
                    n=int(row["n"]),
                    bias=int(row["bias"]),
                )
            )
    return cases


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--shapes", default="bench/linear_shapes.csv")
    p.add_argument("--output", default="bench/results/torch_linear_backward.csv")
    p.add_argument("--warmup", type=int, default=int(os.getenv("FA_LINEAR_BWD_WARMUP", "50")))
    p.add_argument("--iters", type=int, default=int(os.getenv("FA_LINEAR_BWD_ITERS", "200")))
    return p.parse_args()


def backward_flops(m: int, n: int, k: int, has_bias: bool) -> float:
    mm_flops = 4.0 * float(m) * float(n) * float(k)
    bias_flops = float(m) * float(n) if has_bias else 0.0
    return mm_flops + bias_flops


def main():
    args = parse_args()
    try:
        import torch
    except Exception as exc:
        print(f"error: failed to import torch: {exc}", file=sys.stderr)
        return 1

    if not torch.cuda.is_available():
        print("error: CUDA is unavailable in PyTorch", file=sys.stderr)
        return 1

    cases = load_cases(args.shapes)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    device = torch.device("cuda")

    fieldnames = [
        "name",
        "m",
        "k",
        "n",
        "bias",
        "mode",
        "dtype",
        "gpu_name",
        "warmup",
        "iters",
        "latency_ms",
        "tflops",
        "status",
        "error",
    ]
    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        gpu_name = torch.cuda.get_device_name(device)
        for c in cases:
            row = {
                "name": c.name,
                "m": c.m,
                "k": c.k,
                "n": c.n,
                "bias": c.bias,
                "mode": "full",
                "dtype": "float32",
                "gpu_name": gpu_name,
                "warmup": args.warmup,
                "iters": args.iters,
                "latency_ms": "",
                "tflops": "",
                "status": "ok",
                "error": "",
            }
            try:
                x = torch.randn((c.m, c.k), device=device, dtype=torch.float32)
                w = torch.randn((c.n, c.k), device=device, dtype=torch.float32)
                dy = torch.randn((c.m, c.n), device=device, dtype=torch.float32)

                for _ in range(args.warmup):
                    dx = torch.matmul(dy, w)
                    dw = torch.matmul(dy.transpose(0, 1), x)
                    if c.bias:
                        db = dy.sum(dim=0)
                        del db
                    del dx, dw
                torch.cuda.synchronize()

                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                total_ms = 0.0
                for _ in range(args.iters):
                    start.record()
                    dx = torch.matmul(dy, w)
                    dw = torch.matmul(dy.transpose(0, 1), x)
                    if c.bias:
                        db = dy.sum(dim=0)
                        del db
                    end.record()
                    torch.cuda.synchronize()
                    total_ms += float(start.elapsed_time(end))
                    del dx, dw

                avg_ms = total_ms / float(args.iters)
                row["latency_ms"] = f"{avg_ms:.6f}"
                flops = backward_flops(c.m, c.n, c.k, c.bias == 1)
                tflops = (flops / (avg_ms * 1e-3)) / 1e12
                row["tflops"] = f"{tflops:.6f}"
            except Exception as exc:
                row["status"] = "error"
                row["error"] = str(exc).replace("\n", " ")
            writer.writerow(row)

    print(f"wrote: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
