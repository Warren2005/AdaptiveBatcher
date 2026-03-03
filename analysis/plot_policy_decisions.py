#!/usr/bin/env python3
"""
Plot action selection distribution over time-of-day.
Input CSV: timestamp_ns, batch_id, batch_size, latency_ns, action
Usage: python3 analysis/plot_policy_decisions.py data/bench_results.csv
"""

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ACTION_NAMES = ["Immediate", "Fast (50μs)", "Normal (100μs)", "Slow (200μs)", "Lazy (500μs)"]
ACTION_COLORS = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00"]

MARKET_OPEN_NS = 34200 * 1_000_000_000  # 9:30am in ns since midnight


def main():
    if len(sys.argv) < 2:
        print("Usage: plot_policy_decisions.py <bench_results.csv>")
        sys.exit(1)

    df = pd.read_csv(sys.argv[1])
    if "timestamp_ns" not in df or "action" not in df:
        print("CSV must have 'timestamp_ns' and 'action' columns")
        sys.exit(1)

    # Convert timestamp to minutes since open
    df["minutes"] = (df["timestamp_ns"] - MARKET_OPEN_NS) / 60e9

    # Bin into 5-minute windows
    bins = np.arange(0, 395, 5)  # 0..390 minutes (6.5h)
    df["bin"] = pd.cut(df["minutes"], bins=bins, labels=bins[:-1])

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    # Top: action count per bin (stacked bar)
    ax = axes[0]
    for a in range(5):
        counts = df[df["action"] == a].groupby("bin", observed=True).size()
        ax.bar(counts.index.astype(float), counts.values, width=4.5,
               label=ACTION_NAMES[a], color=ACTION_COLORS[a], alpha=0.8)
    ax.set_ylabel("Batch count")
    ax.set_title("Policy Decisions Over Time of Day")
    ax.legend(loc="upper right", fontsize=8)

    # Bottom: action fraction per bin (stacked area)
    ax = axes[1]
    totals  = df.groupby("bin", observed=True).size()
    bottoms = np.zeros(len(totals))
    for a in range(5):
        counts  = df[df["action"] == a].groupby("bin", observed=True).size().reindex(
            totals.index, fill_value=0)
        fracs   = (counts / totals.replace(0, np.nan)).fillna(0).values
        ax.bar(totals.index.astype(float), fracs, width=4.5, bottom=bottoms,
               label=ACTION_NAMES[a], color=ACTION_COLORS[a], alpha=0.8)
        bottoms += fracs

    ax.set_xlabel("Minutes since market open")
    ax.set_ylabel("Fraction")
    ax.set_ylim(0, 1)
    ax.legend(loc="upper right", fontsize=8)

    plt.tight_layout()
    output = "data/policy_decisions.png"
    plt.savefig(output, dpi=150)
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
