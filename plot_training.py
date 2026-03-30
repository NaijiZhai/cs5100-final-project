import csv
import numpy as np
import matplotlib.pyplot as plt


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


def plot_curves(metrics):
    episodes = metrics["episode"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("DQN Training Curves", fontsize=14)

    # Reward
    ax = axes[0, 0]
    ax.plot(episodes, metrics["reward"], alpha=0.3, color="blue", label="Raw")
    smoothed = smooth(metrics["reward"])
    ax.plot(range(len(smoothed)), smoothed, color="blue", label="Smoothed (20)")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Reward")
    ax.set_title("Episode Reward")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Epsilon
    ax = axes[0, 1]
    ax.plot(episodes, metrics["epsilon"], color="orange")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Epsilon")
    ax.set_title("Exploration Rate")
    ax.grid(True, alpha=0.3)

    # Total Queue
    ax = axes[1, 0]
    ax.plot(episodes, metrics["total_queue"], alpha=0.3, color="red", label="Raw")
    smoothed_q = smooth(metrics["total_queue"])
    ax.plot(range(len(smoothed_q)), smoothed_q, color="red", label="Smoothed (20)")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Final Queue Length")
    ax.set_title("Queue Length at Episode End")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Departed
    ax = axes[1, 1]
    ax.plot(episodes, metrics["departed"], alpha=0.3, color="green", label="Raw")
    smoothed_d = smooth(metrics["departed"])
    ax.plot(range(len(smoothed_d)), smoothed_d, color="green", label="Smoothed (20)")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Vehicles Departed")
    ax.set_title("Throughput (Departed Vehicles)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("training_curves.png", dpi=150)
    print("Saved training_curves.png")
    plt.show()


if __name__ == "__main__":
    metrics = load_metrics()
    plot_curves(metrics)
