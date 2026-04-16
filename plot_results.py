"""
Generates all result plots:
  1. Training curves: reward vs episode, wait vs episode
  2. Bar charts: DQN vs baselines (avg reward, avg wait)
  3. Queue evolution: DQN vs Fixed-Time over decision steps
"""
import csv
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np


def load_csv(path):
    """Load a CSV file into a list of ``OrderedDict`` rows via ``DictReader``.

    This is a thin generic helper — all type conversion is left to the
    caller since different CSVs have different column types.
    """
    with open(path, "r") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    return rows


def smooth(values, window=20):
    """Compute a moving average and return aligned x-indices.

    Returns ``(x, smoothed)`` where *x* starts at ``window - 1`` so the
    smoothed curve lines up with the correct episode numbers on the
    x-axis.  Falls back to the identity when the array is shorter than
    *window*.
    """
    if len(values) < window:
        return np.arange(len(values)), values
    # Uniform kernel — each output point is the mean of `window` neighbours
    kernel = np.ones(window) / window
    smoothed = np.convolve(values, kernel, mode="valid")
    x = np.arange(window - 1, len(values))
    return x, smoothed


def plot_training_curves(metrics_path="training_metrics.csv"):
    """Generate a 2-panel figure: reward vs episode and wait vs episode.

    Each panel shows the raw per-episode values at low opacity with a
    bold 20-episode moving-average overlay to highlight the trend.
    """
    rows = load_csv(metrics_path)
    episodes = np.array([float(r["episode"]) for r in rows])
    rewards = np.array([float(r["episode_reward"]) for r in rows])
    waits = np.array([float(r["average_wait"]) for r in rows])

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("Training Curves", fontsize=14)

    ax = axes[0]
    ax.plot(episodes, rewards, alpha=0.3, color="blue")
    sx, sy = smooth(rewards)
    ax.plot(sx, sy, color="blue", label="Smoothed (20)")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Reward")
    ax.set_title("Reward vs Episode")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(episodes, waits, alpha=0.3, color="red")
    sx, sy = smooth(waits)
    ax.plot(sx, sy, color="red", label="Smoothed (20)")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Average Wait")
    ax.set_title("Wait vs Episode")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("training_curves.png", dpi=150)
    print("Saved training_curves.png")


def plot_bar_comparison(summary_path="eval_summary.csv"):
    """Generate a side-by-side bar chart comparing all evaluated policies.

    Left panel: average episode reward.  Right panel: average wait time.
    DQN bars are highlighted in colour; all other policies use grey so
    the learned policy stands out visually.
    """
    rows = load_csv(summary_path)
    policies = [r["policy"] for r in rows]
    avg_rewards = [float(r["avg_reward"]) for r in rows]
    avg_waits = [float(r["avg_wait"]) for r in rows]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Policy Comparison", fontsize=14)

    x = np.arange(len(policies))

    ax = axes[0]
    bars = ax.bar(x, avg_rewards, color=["#2196F3" if "DQN" in p else "#9E9E9E" for p in policies])
    ax.set_xticks(x)
    ax.set_xticklabels(policies, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Avg Episode Reward")
    ax.set_title("Average Reward by Policy")
    ax.grid(True, alpha=0.3, axis="y")

    ax = axes[1]
    bars = ax.bar(x, avg_waits, color=["#F44336" if "DQN" in p else "#9E9E9E" for p in policies])
    ax.set_xticks(x)
    ax.set_xticklabels(policies, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Avg Wait")
    ax.set_title("Average Wait by Policy")
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig("policy_comparison.png", dpi=150)
    print("Saved policy_comparison.png")


def plot_queue_evolution(queue_path="eval_queue_evolution.csv"):
    """Plot mean ± std shaded-area queue evolution across episodes.

    For each policy, all per-episode queue traces are zero-padded to the
    same length, then the mean and standard deviation are computed at
    every decision step.  The mean is drawn as a solid line and the
    ±1 std band is shown as a translucent fill, giving a clear picture
    of both the central tendency and the variability.
    """
    rows = load_csv(queue_path)

    # Group rows by policy → episode → list of queue values
    by_policy = defaultdict(lambda: defaultdict(list))
    for r in rows:
        by_policy[r["policy"]][int(r["episode"])].append(float(r["queue_length"]))

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = {"DQN Policy": "#2196F3", "Fixed-Time (Keep)": "#FF9800"}

    for policy_name, episodes in by_policy.items():
        all_traces = list(episodes.values())
        max_len = max(len(t) for t in all_traces)
        # Pad shorter traces with NaN so nanmean/nanstd ignore missing tails
        padded = np.full((len(all_traces), max_len), np.nan)
        for i, trace in enumerate(all_traces):
            padded[i, :len(trace)] = trace

        mean_q = np.nanmean(padded, axis=0)
        std_q = np.nanstd(padded, axis=0)
        steps = np.arange(max_len)

        color = colors.get(policy_name, "#666666")
        ax.plot(steps, mean_q, label=policy_name, color=color)
        ax.fill_between(steps, mean_q - std_q, mean_q + std_q, alpha=0.15, color=color)

    ax.set_xlabel("Decision Step")
    ax.set_ylabel("Total Queue Length")
    ax.set_title("Queue Evolution: DQN vs Fixed-Time")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("queue_evolution.png", dpi=150)
    print("Saved queue_evolution.png")


if __name__ == "__main__":
    print("Generating training curves...")
    plot_training_curves()

    print("Generating policy comparison...")
    plot_bar_comparison()

    print("Generating queue evolution...")
    plot_queue_evolution()

    if plt.get_backend().lower() != "agg":
        plt.show()
