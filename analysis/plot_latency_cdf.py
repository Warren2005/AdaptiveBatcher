#!/usr/bin/env python3
"""
Plot CDF curves for adaptive batcher vs fixed-window baselines.
Input: CSV with columns: timestamp_ns, batch_id, batch_size, latency_ns, action
Usage: python3 analysis/plot_latency_cdf.py data/bench_results.csv
"""

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ACTION_NAMES = {0: "Immediate (0μs)", 1: "Fast (50μs)", 2: "Normal (100μs)",
                3: "Slow (200μs)",    4: "Lazy (500μs)"}

def plot_cdf(latencies_dict, title="Latency CDF", output=None):
    fig, ax = plt.subplots(figsize=(10, 6))

    for label, lat in latencies_dict.items():
        lat_us = np.sort(np.array(lat, dtype=float)) / 1000.0  # ns → μs
        cdf    = np.arange(1, len(lat_us) + 1) / len(lat_us)
        ax.plot(lat_us, cdf, label=label, linewidth=1.5)

    ax.set_xlabel("Latency (μs)", fontsize=12)
    ax.set_ylabel("CDF", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(left=0)
    ax.set_ylim(0, 1.05)

    # Mark p99
    for label, lat in latencies_dict.items():
        lat_us = np.array(lat, dtype=float) / 1000.0
        p99    = np.percentile(lat_us, 99)
        ax.axvline(p99, linestyle='--', alpha=0.5)

    plt.tight_layout()
    if output:
        plt.savefig(output, dpi=150)
        print(f"Saved: {output}")
    else:
        plt.show()


def main():
    if len(sys.argv) < 2:
        print("Usage: plot_latency_cdf.py <bench_results.csv>")
        sys.exit(1)

    df = pd.read_csv(sys.argv[1])

    if "action" in df.columns:
        # Per-action breakdown
        latencies = {}
        for action_id, name in ACTION_NAMES.items():
            subset = df[df["action"] == action_id]["latency_ns"]
            if len(subset) > 0:
                latencies[name] = subset.values
        plot_cdf(latencies, "Latency CDF by Action",
                 output="data/cdf_by_action.png")
    else:
        # Single policy
        plot_cdf({"All": df["latency_ns"].values}, "Latency CDF",
                 output="data/cdf_all.png")

    # Summary stats
    lat_us = df["latency_ns"].values / 1000.0
    print(f"p50={np.percentile(lat_us, 50):.1f}μs  "
          f"p99={np.percentile(lat_us, 99):.1f}μs  "
          f"p99.9={np.percentile(lat_us, 99.9):.1f}μs")


if __name__ == "__main__":
    main()
