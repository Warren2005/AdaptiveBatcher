#!/usr/bin/env python3
"""
Plot cumulative reward + weight norms over time.
Input CSV: step, reward, weight_norm_0..4 (one row per SGD update)
Usage: python3 analysis/plot_reward_convergence.py data/training_log.csv
"""

import sys
import pandas as pd
import matplotlib.pyplot as plt

ACTION_NAMES = ["Immediate", "Fast", "Normal", "Slow", "Lazy"]
ACTION_COLORS = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00"]


def main():
    if len(sys.argv) < 2:
        print("Usage: plot_reward_convergence.py <training_log.csv>")
        sys.exit(1)

    df = pd.read_csv(sys.argv[1])

    _, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    # Top: cumulative reward
    ax = axes[0]
    if "reward" in df.columns:
        cum_reward = df["reward"].cumsum()
        ax.plot(cum_reward.values, color="steelblue", linewidth=1.2)
        ax.set_ylabel("Cumulative Reward")
        ax.set_title("SGD Convergence: Reward and Weight Norms")
        ax.grid(True, alpha=0.3)

    # Bottom: weight norms per action
    ax = axes[1]
    for a in range(5):
        col = f"weight_norm_{a}"
        if col in df.columns:
            ax.plot(df[col].values, label=ACTION_NAMES[a],
                    color=ACTION_COLORS[a], linewidth=1.2)

    ax.set_xlabel("SGD step")
    ax.set_ylabel("Weight L2 norm")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output = "data/reward_convergence.png"
    plt.savefig(output, dpi=150)
    print(f"Saved: {output}")

    # Summary stats
    if "reward" in df.columns:
        print(f"Final mean reward (last 100 steps): "
              f"{df['reward'].tail(100).mean():.2f}")


if __name__ == "__main__":
    main()
