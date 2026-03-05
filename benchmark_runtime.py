#!/usr/bin/env python3
"""Runtime benchmark for all Kalman filter models.

Measures inference time (ms/step) and parameter count for each model.
Produces a CSV and prints a LaTeX-ready table.

Usage:
    python benchmark_runtime.py --device cuda:0
    python benchmark_runtime.py --device cpu
"""

import argparse
import csv
import time
from pathlib import Path

import torch
import torch.nn as nn

from models import (
    AutoRegKF,
    BayesianKalmanNet,
    ClassicalKalmanFilter,
    DeepKalmanFilter,
    KalmanNet,
    RecurrentKalmanNetwork,
    RecursiveKalmanNet,
)

MODELS = {
    "FW-NKF (ours)": ("deep_kf", DeepKalmanFilter),
    "KalmanNet": ("kalman_net", KalmanNet),
    "BayesKNet": ("bayes_knet", BayesianKalmanNet),
    "Recursive KNet": ("recursive_knet", RecursiveKalmanNet),
    "Recurrent KNet": ("recurrent_kalman_network", RecurrentKalmanNetwork),
    "Classical KF": ("classical_kf", ClassicalKalmanFilter),
    "Autoreg KF": ("autoreg_kf", AutoRegKF),
}

# Lorenz dimensions (state=3, obs=2, control=0) — matches paper
STATE_DIM = 3
OBS_DIM = 2
CONTROL_DIM = 0


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def count_trainable_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def benchmark_model(
    model: nn.Module,
    device: torch.device,
    batch_size: int = 1,
    seq_len: int = 100,
    n_warmup: int = 50,
    n_runs: int = 200,
) -> float:
    """Returns average inference time in ms per timestep."""
    model.eval()
    model.to(device)

    obs = torch.randn(batch_size, seq_len, OBS_DIM, device=device)

    # Warmup
    with torch.no_grad():
        for _ in range(n_warmup):
            model.reset_state(batch_size, device)
            model(obs)

    if device.type == "cuda":
        torch.cuda.synchronize(device)

    # Timed runs
    times = []
    with torch.no_grad():
        for _ in range(n_runs):
            model.reset_state(batch_size, device)

            if device.type == "cuda":
                torch.cuda.synchronize(device)
            t0 = time.perf_counter()

            model(obs)

            if device.type == "cuda":
                torch.cuda.synchronize(device)
            t1 = time.perf_counter()

            times.append(t1 - t0)

    avg_total_ms = (sum(times) / len(times)) * 1000.0
    ms_per_step = avg_total_ms / seq_len
    return ms_per_step


def main():
    parser = argparse.ArgumentParser(description="Runtime benchmark")
    parser.add_argument("--device", default="cuda:0", help="Device to benchmark on")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seq-len", type=int, default=100)
    parser.add_argument("--n-warmup", type=int, default=50)
    parser.add_argument("--n-runs", type=int, default=200)
    parser.add_argument("--save-dir", default="./checkpoints", help="Output directory")
    args = parser.parse_args()

    device = torch.device(args.device)
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(device)}")
    print(f"Batch size: {args.batch_size}, Seq len: {args.seq_len}")
    print(f"Warmup: {args.n_warmup}, Runs: {args.n_runs}")
    print(f"Dims: state={STATE_DIM}, obs={OBS_DIM}, control={CONTROL_DIM}")
    print()

    results = []

    for display_name, (key, cls) in MODELS.items():
        # Build model with default args
        try:
            if key in ("autoreg_kf", "recursive_knet", "recurrent_kalman_network"):
                model = cls(state_dim=STATE_DIM, obs_dim=OBS_DIM)
            else:
                model = cls(
                    state_dim=STATE_DIM,
                    obs_dim=OBS_DIM,
                    control_dim=CONTROL_DIM,
                )
        except Exception as e:
            print(f"[SKIP] {display_name}: {e}")
            continue

        n_params = count_params(model)
        n_trainable = count_trainable_params(model)

        ms_per_step = benchmark_model(
            model,
            device,
            batch_size=args.batch_size,
            seq_len=args.seq_len,
            n_warmup=args.n_warmup,
            n_runs=args.n_runs,
        )

        results.append({
            "method": display_name,
            "key": key,
            "params": n_params,
            "trainable_params": n_trainable,
            "ms_per_step": ms_per_step,
        })

        print(
            f"{display_name:25s}  "
            f"params={n_params:>8,}  "
            f"trainable={n_trainable:>8,}  "
            f"{ms_per_step:.4f} ms/step"
        )

    # Save CSV
    csv_path = save_dir / "runtime_benchmark.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["method", "key", "params", "trainable_params", "ms_per_step"])
        w.writeheader()
        w.writerows(results)
    print(f"\nSaved to {csv_path}")

    # Print LaTeX table
    print("\n% --- LaTeX table ---")
    print(r"\begin{table}[t]")
    print(r"\caption{Runtime and model size comparison (batch=1, seq=100, " + args.device + r")}")
    print(r"\centering")
    print(r"\begin{tabular}{@{}lrr@{}}")
    print(r"\toprule")
    print(r"Method & \#Params & ms/step \\")
    print(r"\midrule")
    for r in results:
        params_str = f"{r['params']:,}"
        ms_str = f"{r['ms_per_step']:.3f}"
        print(f"{r['method']} & {params_str} & {ms_str} \\\\")
    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\end{table}")


if __name__ == "__main__":
    main()
