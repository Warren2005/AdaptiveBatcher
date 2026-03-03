#!/usr/bin/env python3
"""
Plot p99 vs sample count after each regime transition.
Input CSV: timestamp_ns, batch_id, batch_size, latency_ns, action, regime (optional)
Usage: python3 analysis/plot_regime_adaptation.py data/bench_results.csv
"""

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

REGIME_NAMES = ["OPEN", "MORNING", "MIDDAY", "AFTERNOON", "CLOSE"]
REGIME_COLORS = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00"]

# Regime transition times in minutes since open (matching regime_generator.cpp)
REGIME_TRANSITIONS = [0, 30, 120, 270, 360]


def rolling_p99(values, window=100):
    result = []
    for i in range(len(values)):
        start = max(0, i - window + 1)
        result.append(np.percentile(values[start:i+1], 99))
    return np.array(result)


def main():
    if len(sys.argv) < 2:
        print("Usage: plot_regime_adaptation.py <bench_results.csv>")
        sys.exit(1)

    df = pd.read_csv(sys.argv[1])
    df["latency_us"] = df["latency_ns"] / 1000.0

    MARKET_OPEN_NS = 34200 * 1_000_000_000

    fig, ax = plt.subplots(figsize=(14, 6))

    lat  = df["latency_us"].values
    p99s = rolling_p99(lat, window=100)
    x    = np.arange(len(p99s))

    ax.plot(x, p99s, color="steelblue", linewidth=1.2, label="Adaptive p99 (rolling 100)")

    # Mark regime transitions
    if "timestamp_ns" in df.columns:
        open_ns = df["timestamp_ns"].min()
        for i, t_min in enumerate(REGIME_TRANSITIONS[1:], 1):
            t_ns    = open_ns + t_min * 60 * 1_000_000_000
            idx     = df[df["timestamp_ns"] >= t_ns].index
            if len(idx) > 0:
                ax.axvline(idx[0], linestyle='--', color=REGIME_COLORS[i],
                           label=f"→ {REGIME_NAMES[i]}", alpha=0.7)

    ax.set_xlabel("Sample index")
    ax.set_ylabel("p99 latency (μs)")
    ax.set_title("p99 Adaptation After Regime Transitions")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    output = "data/regime_adaptation.png"
    plt.tight_layout()
    plt.savefig(output, dpi=150)
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
