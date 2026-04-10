import csv

import matplotlib.pyplot as plt
import numpy as np


def load_metrics(path="training_metrics.csv"):
    data = {}
    with open(path, "r") as f:
        reader = csv.DictReader(f)
        for key in reader.fieldnames:
            data[key] = []
        for row in reader:
            for key in reader.fieldnames:
                data[key].append(float(row[key]))
    return {k: np.array(v) for k, v in data.items()}


def smooth(values, window=20):
    if len(values) < window:
        return values
    kernel = np.ones(window) / window
    return np.convolve(values, kernel, mode="valid")


def plot_with_smoothing(ax, episodes, values, ylabel, title, window=20):
    ax.plot(episodes, values, alpha=0.3, label="Raw")
    smoothed = smooth(values, window=window)

    if len(values) >= window:
        smoothed_x = episodes[window - 1 :]
    else:
        smoothed_x = episodes

    ax.plot(smoothed_x, smoothed, label=f"Smoothed ({window})")
    ax.set_xlabel("Episode")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()


def get_metric(metrics, key, fallback_key=None, default_value=None):
    if key in metrics:
        return metrics[key], key
    if fallback_key and fallback_key in metrics:
        return metrics[fallback_key], fallback_key
    if default_value is not None:
        return np.full_like(metrics["episode"], default_value, dtype=np.float32), "default"
    raise KeyError(f"Missing metric '{key}' and fallback '{fallback_key}'")


def plot_curves(metrics):
    episodes = metrics["episode"]

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle("DQN Training Curves (Paper-like Single Intersection)", fontsize=14)

    plot_with_smoothing(
        axes[0, 0],
        episodes,
        metrics["episode_reward"],
        ylabel="Episode Reward",
        title="Episode Reward",
    )

    ax = axes[0, 1]
    ax.plot(episodes, metrics["epsilon"])
    ax.set_xlabel("Episode")
    ax.set_ylabel("Epsilon")
    ax.set_title("Exploration Rate")
    ax.grid(True, alpha=0.3)

    plot_with_smoothing(
        axes[0, 2],
        episodes,
        metrics["average_queue"],
        ylabel="Average Queue",
        title="Average Queue",
    )

    plot_with_smoothing(
        axes[1, 0],
        episodes,
        metrics["average_wait"],
        ylabel="Average Wait",
        title="Average Wait",
    )

    imbalance_values, imbalance_key = get_metric(
        metrics,
        "average_imbalance",
        fallback_key="average_queue",
        default_value=0.0,
    )
    plot_with_smoothing(
        axes[1, 1],
        episodes,
        imbalance_values,
        ylabel="Average Imbalance",
        title=f"Average Imbalance ({imbalance_key})",
    )

    throughput_values, throughput_key = get_metric(
        metrics,
        "avg_departed_per_sec",
        fallback_key="total_departed",
    )
    throughput_title = "Throughput (Departed/sec)" if throughput_key == "avg_departed_per_sec" else "Throughput (Total Departed)"
    throughput_ylabel = "Departed per sec" if throughput_key == "avg_departed_per_sec" else "Vehicles Departed"
    plot_with_smoothing(
        axes[1, 2],
        episodes,
        throughput_values,
        ylabel=throughput_ylabel,
        title=throughput_title,
    )

    plt.tight_layout()
    plt.savefig("training_curves.png", dpi=150)
    print("Saved training_curves.png")
    if plt.get_backend().lower() != "agg":
        plt.show()


if __name__ == "__main__":
    metrics = load_metrics()
    plot_curves(metrics)
