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
    p.add_argument("--output", default="bench/results/torch_linear.csv")
    p.add_argument("--warmup", type=int, default=int(os.getenv("FA_LINEAR_WARMUP", "50")))
    p.add_argument("--iters", type=int, default=int(os.getenv("FA_LINEAR_ITERS", "200")))
    return p.parse_args()


def main():
    args = parse_args()
    try:
        import torch
        import torch.nn.functional as F
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
                with torch.no_grad():
                    x = torch.randn((c.m, c.k), device=device, dtype=torch.float32)
                    w = torch.randn((c.n, c.k), device=device, dtype=torch.float32)
                    b = torch.randn((c.n,), device=device, dtype=torch.float32) if c.bias else None

                    for _ in range(args.warmup):
                        _ = F.linear(x, w, b)
                    torch.cuda.synchronize()

                    start = torch.cuda.Event(enable_timing=True)
                    end = torch.cuda.Event(enable_timing=True)
                    total_ms = 0.0
                    for _ in range(args.iters):
                        start.record()
                        _ = F.linear(x, w, b)
                        end.record()
                        torch.cuda.synchronize()
                        total_ms += float(start.elapsed_time(end))

                    avg_ms = total_ms / float(args.iters)
                    row["latency_ms"] = f"{avg_ms:.6f}"
                    flops = 2.0 * float(c.m) * float(c.k) * float(c.n)
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
