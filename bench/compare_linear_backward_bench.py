#!/usr/bin/env python3
import argparse
import csv
import os
from typing import Dict, Tuple


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--native", default="bench/results/native_linear_backward.csv")
    p.add_argument("--torch", default="bench/results/torch_linear_backward.csv")
    p.add_argument("--output", default="bench/results/linear_backward_compare.csv")
    return p.parse_args()


def time_to_ms(value: str, unit: str) -> float:
    v = float(value)
    u = unit.strip().lower()
    if u == "ms":
        return v
    if u == "us":
        return v * 1e-3
    if u == "ns":
        return v * 1e-6
    if u == "s":
        return v * 1e3
    raise ValueError(f"unsupported time unit: {unit}")


def parse_case_from_name(name: str):
    # Example:
    # LinearBackward/name=mlp_s/m=32/k=256/n=256/bias=0/mode=full
    parts = name.split("/")
    out = {}
    for p in parts:
        if "=" in p:
            k, v = p.split("=", 1)
            out[k] = v
    return out


def backward_flops(m: int, n: int, k: int, has_bias: bool) -> float:
    mm_flops = 4.0 * float(m) * float(n) * float(k)
    bias_flops = float(m) * float(n) if has_bias else 0.0
    return mm_flops + bias_flops


def load_native(path: str) -> Dict[Tuple[str, int, int, int, int, str], dict]:
    out = {}
    with open(path, "r", newline="") as f:
        lines = f.readlines()

    header_idx = None
    for i, line in enumerate(lines):
        if line.lstrip().startswith("name,"):
            header_idx = i
            break

    if header_idx is None:
        return out

    reader = csv.DictReader(lines[header_idx:])
    for row in reader:
        name = row.get("name", "")
        if not name.startswith("LinearBackward/"):
            continue
        if row.get("error_occurred", "").lower() in ("true", "1"):
            continue

        case = parse_case_from_name(name)
        case_name = case.get("name", "")
        m = int(case.get("m", "0"))
        k = int(case.get("k", "0"))
        n = int(case.get("n", "0"))
        bias = int(case.get("bias", "0"))
        mode = case.get("mode", "full")
        latency_ms = time_to_ms(row["real_time"], row["time_unit"])
        flops = backward_flops(m, n, k, bias == 1)

        key = (case_name, m, k, n, bias, mode)
        out[key] = {
            "native_latency_ms": latency_ms,
            "native_tflops": (flops / (latency_ms * 1e-3)) / 1e12,
        }
    return out


def load_torch(path: str) -> Dict[Tuple[str, int, int, int, int, str], dict]:
    out = {}
    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("status", "ok") != "ok":
                continue
            key = (
                row["name"],
                int(row["m"]),
                int(row["k"]),
                int(row["n"]),
                int(row["bias"]),
                row.get("mode", "full"),
            )
            out[key] = {
                "torch_latency_ms": float(row["latency_ms"]),
                "torch_tflops": float(row["tflops"]),
            }
    return out


def main():
    args = parse_args()
    native = load_native(args.native)
    torch = load_torch(args.torch)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    fields = [
        "name",
        "m",
        "k",
        "n",
        "bias",
        "mode",
        "native_latency_ms",
        "torch_latency_ms",
        "speedup_native_vs_torch",
        "native_tflops",
        "torch_tflops",
    ]
    rows = []
    for key, nrow in native.items():
        if key not in torch:
            continue
        trow = torch[key]
        speedup = trow["torch_latency_ms"] / nrow["native_latency_ms"]
        rows.append(
            {
                "name": key[0],
                "m": key[1],
                "k": key[2],
                "n": key[3],
                "bias": key[4],
                "mode": key[5],
                "native_latency_ms": f"{nrow['native_latency_ms']:.6f}",
                "torch_latency_ms": f"{trow['torch_latency_ms']:.6f}",
                "speedup_native_vs_torch": f"{speedup:.6f}",
                "native_tflops": f"{nrow['native_tflops']:.6f}",
                "torch_tflops": f"{trow['torch_tflops']:.6f}",
            }
        )

    rows.sort(key=lambda r: (r["name"], int(r["bias"]), r["mode"]))
    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    print(f"wrote: {args.output}")
    print(f"native rows: {len(native)}, torch rows: {len(torch)}, matched rows: {len(rows)}")
    if rows:
        print("name,bias,mode,native_ms,torch_ms,speedup(torch/native)")
        for row in rows:
            print(
                f"{row['name']},{row['bias']},{row['mode']},{row['native_latency_ms']},"
                f"{row['torch_latency_ms']},{row['speedup_native_vs_torch']}"
            )
    else:
        print("no matched rows found; check shape keys and benchmark outputs")


if __name__ == "__main__":
    main()
