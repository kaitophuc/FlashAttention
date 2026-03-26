#!/usr/bin/env python3
import argparse
import csv
import glob
import os
import sys
from typing import List, Dict


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--compare", default="bench/results/linear_compare.csv")
    p.add_argument("--native", default="bench/results/native_linear.csv")
    p.add_argument("--torch", default="bench/results/torch_linear.csv")
    p.add_argument("--outdir", default="bench/results")
    p.add_argument("--prefix", default="linear_compare")
    return p.parse_args()


def find_latest_compare_csv(outdir: str) -> str:
    candidates = sorted(glob.glob(os.path.join(outdir, "linear_compare_*.csv")))
    if not candidates:
        return ""
    return candidates[-1]


def read_compare_csv(path: str) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            label = f"{row['name']}{'_b' if row['bias'] == '1' else ''}"
            rows.append(
                {
                    "label": label,
                    "m": int(row["m"]),
                    "k": int(row["k"]),
                    "n": int(row["n"]),
                    "bias": int(row["bias"]),
                    "native_latency_ms": float(row["native_latency_ms"]),
                    "torch_latency_ms": float(row["torch_latency_ms"]),
                    "native_tflops": float(row["native_tflops"]),
                    "torch_tflops": float(row["torch_tflops"]),
                    "speedup": float(row["speedup_native_vs_torch"]),
                }
            )
    return rows


def main():
    args = parse_args()

    compare_path = args.compare
    if not os.path.exists(compare_path):
        latest = find_latest_compare_csv(args.outdir)
        if latest:
            compare_path = latest
        else:
            print(
                f"error: compare file not found: {args.compare}\n"
                "hint: run bench/run_linear_compare.sh first",
                file=sys.stderr,
            )
            return 1

    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"error: failed to import matplotlib: {exc}", file=sys.stderr)
        return 1

    rows = read_compare_csv(compare_path)
    if not rows:
        print("error: compare file has no rows", file=sys.stderr)
        return 1

    os.makedirs(args.outdir, exist_ok=True)

    # If prefix not explicitly provided, derive from the compare CSV filename.
    prefix = args.prefix
    if prefix == "linear_compare":
        stem = os.path.splitext(os.path.basename(compare_path))[0]
        if stem.startswith("linear_compare_"):
            prefix = stem

    labels = [r["label"] for r in rows]
    native_lat = [r["native_latency_ms"] for r in rows]
    torch_lat = [r["torch_latency_ms"] for r in rows]
    native_tflops = [r["native_tflops"] for r in rows]
    torch_tflops = [r["torch_tflops"] for r in rows]
    speedup = [r["speedup"] for r in rows]
    bytes_per_case = []
    for r in rows:
        # Approx forward bytes touched (fp32): read X + read W + optional read b + write Y.
        x_bytes = r["m"] * r["k"] * 4
        w_bytes = r["n"] * r["k"] * 4
        b_bytes = r["n"] * 4 if r["bias"] == 1 else 0
        y_bytes = r["m"] * r["n"] * 4
        bytes_per_case.append(float(x_bytes + w_bytes + b_bytes + y_bytes))
    native_bw = [b / (ms * 1e-3) / 1e9 for b, ms in zip(bytes_per_case, native_lat)]
    torch_bw = [b / (ms * 1e-3) / 1e9 for b, ms in zip(bytes_per_case, torch_lat)]

    x = list(range(len(labels)))
    width = 0.38
    colors = {"native": "#4285f4", "torch": "#db4437", "speedup": "#f4b400"}

    # Figure 1: grouped bars per case (similar style to requested sample).
    fig, axes = plt.subplots(4, 1, figsize=(15, 15), constrained_layout=True)

    ax = axes[0]
    ax.bar([i - width / 2 for i in x], native_lat, width, label="Native", color=colors["native"])
    ax.bar([i + width / 2 for i in x], torch_lat, width, label="PyTorch", color=colors["torch"])
    ax.set_title("Linear Forward Latency by Case (FP32)")
    ax.set_ylabel("Latency (ms)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.legend()

    ax = axes[1]
    ax.bar([i - width / 2 for i in x], native_tflops, width, label="Native", color=colors["native"])
    ax.bar([i + width / 2 for i in x], torch_tflops, width, label="PyTorch", color=colors["torch"])
    ax.set_title("Linear Forward Throughput by Case (FP32)")
    ax.set_ylabel("TFLOP/s")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.legend()

    ax = axes[2]
    ax.bar([i - width / 2 for i in x], native_bw, width, label="Native", color=colors["native"])
    ax.bar([i + width / 2 for i in x], torch_bw, width, label="PyTorch", color=colors["torch"])
    ax.set_title("Linear Forward Effective Runtime Bandwidth by Case (FP32)")
    ax.set_ylabel("GB/s")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.legend()

    ax = axes[3]
    ax.bar(x, speedup, width=0.55, label="Torch/Native", color=colors["speedup"])
    ax.axhline(1.0, color="black", linestyle="--", linewidth=1)
    ax.set_title("Speedup by Case (Torch/Native)")
    ax.set_ylabel("x")
    ax.set_xlabel("Case")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.legend()

    dashboard_png = os.path.join(args.outdir, f"{prefix}_bar_dashboard.png")
    fig.savefig(dashboard_png, dpi=160)
    plt.close(fig)

    # Figure 2: latency-only grouped bars (clean single chart).
    fig2, ax2 = plt.subplots(figsize=(14, 5), constrained_layout=True)
    ax2.bar([i - width / 2 for i in x], native_lat, width, label="Native", color=colors["native"])
    ax2.bar([i + width / 2 for i in x], torch_lat, width, label="PyTorch", color=colors["torch"])
    ax2.set_title("Linear Forward Latency Comparison (Grouped Bars)")
    ax2.set_ylabel("Latency (ms)")
    ax2.set_xlabel("Case")
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=30, ha="right")
    ax2.grid(axis="y", linestyle="--", alpha=0.3)
    ax2.legend()
    latency_png = os.path.join(args.outdir, f"{prefix}_latency_grouped_bar.png")
    fig2.savefig(latency_png, dpi=160)
    plt.close(fig2)

    print(f"wrote: {dashboard_png}")
    print(f"wrote: {latency_png}")
    print(f"input compare file: {compare_path}")
    print(f"input compare rows: {len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
