import argparse
import csv
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from agent import Agent
from traffic_signal_env import TrafficSignalEnv


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_arrival_prob(text):
    parts = [p.strip() for p in text.split(",") if p.strip()]
    if len(parts) != 4:
        raise argparse.ArgumentTypeError("arrival_prob must contain 4 comma-separated values")

    values = tuple(float(p) for p in parts)
    for v in values:
        if v < 0.0 or v > 1.0:
            raise argparse.ArgumentTypeError("arrival_prob values must be in [0, 1]")
    return values


def build_parser():
    parser = argparse.ArgumentParser(description="Train DQN for paper-like single-intersection control")

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--episodes", type=int, default=600)
    parser.add_argument("--max-decisions", type=int, default=300, help="Decision steps per episode")
    parser.add_argument("--max-steps", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument(
        "--control-interval",
        type=int,
        default=None,
        help="Deprecated in cycle-level stepping and ignored by default.",
    )
    parser.add_argument("--arrival-prob", type=parse_arrival_prob, default=(0.30, 0.30, 0.30, 0.30))
    parser.add_argument("--demand-variation", type=float, default=0.25)
    parser.add_argument("--demand-period", type=int, default=240)
    parser.add_argument("--history-windows", type=int, default=3)
    parser.add_argument("--green-delta", type=int, default=2)
    parser.add_argument("--min-green", type=int, default=6)

    parser.add_argument("--epsilon-start", type=float, default=1.0)
    parser.add_argument("--epsilon-end", type=float, default=0.05)
    parser.add_argument("--epsilon-decay", type=int, default=180000)

    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--memory-size", type=int, default=100000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--target-update", type=int, default=20)

    parser.add_argument("--reward-wait-weight", type=float, default=1.0)
    parser.add_argument("--reward-queue-weight", type=float, default=0.2)
    parser.add_argument("--reward-imbalance-weight", type=float, default=0.2)
    parser.add_argument("--reward-throughput-weight", type=float, default=0.5)
    parser.add_argument("--reward-switch-weight", type=float, default=0.3)

    parser.add_argument("--log-path", type=str, default="training_metrics.csv")
    parser.add_argument("--model-path", type=str, default="dqn_model.pth")
    parser.add_argument("--checkpoint-path", type=str, default="dqn_checkpoint.pth")
    parser.add_argument("--print-every", type=int, default=20)

    return parser


def build_env_kwargs(args):
    max_decisions = args.max_decisions if args.max_steps is None else args.max_steps
    return {
        "max_decisions": max_decisions,
        "control_interval": args.control_interval,
        "arrival_prob": tuple(args.arrival_prob),
        "demand_variation": args.demand_variation,
        "demand_period": args.demand_period,
        "history_windows": args.history_windows,
        "green_delta": args.green_delta,
        "min_green": args.min_green,
        "reward_wait_weight": args.reward_wait_weight,
        "reward_queue_weight": args.reward_queue_weight,
        "reward_imbalance_weight": args.reward_imbalance_weight,
        "reward_throughput_weight": args.reward_throughput_weight,
        "reward_switch_weight": args.reward_switch_weight,
    }


def build_agent_kwargs(args):
    return {
        "gamma": args.gamma,
        "learning_rate": args.learning_rate,
        "hidden_dim": args.hidden_dim,
        "memory_size": args.memory_size,
        "batch_size": args.batch_size,
    }


def train_dqn(args):
    set_seed(args.seed)

    env_kwargs = build_env_kwargs(args)
    env = TrafficSignalEnv(**env_kwargs)

    n_episode = args.episodes
    n_steps = env.max_decisions

    agent_kwargs = build_agent_kwargs(args)
    agent = Agent(
        env.observation_space.shape[0],
        env.action_space.n,
        **agent_kwargs,
    )

    global_step = 0
    best_episode_reward = float("-inf")
    best_episode = -1

    for out_path in [args.log_path, args.model_path, args.checkpoint_path]:
        out_dir = Path(out_path).parent
        if str(out_dir) not in ("", "."):
            out_dir.mkdir(parents=True, exist_ok=True)

    with open(args.log_path, "w", newline="") as log_file:
        writer = csv.writer(log_file)
        writer.writerow(
            [
                "episode",
                "episode_reward",
                "avg_reward",
                "epsilon",
                "sim_seconds",
                "total_departed",
                "avg_departed_per_sec",
                "switch_count",
                "average_queue",
                "average_wait",
                "average_imbalance",
                "cumulative_queue",
                "cumulative_wait",
                "final_total_queue",
                "final_total_wait",
            ]
        )

        for episode in range(n_episode):
            s, info = env.reset(seed=args.seed + episode)
            steps_this_episode = 0
            epsilon = args.epsilon_start

            for _ in range(n_steps):
                steps_this_episode += 1
                global_step += 1

                epsilon = np.interp(
                    global_step,
                    [0, args.epsilon_decay],
                    [args.epsilon_start, args.epsilon_end],
                )

                if random.random() <= epsilon:
                    action = env.action_space.sample()
                else:
                    action = agent.select_action(s)

                s_new, r, done, trunc, info = env.step(action)
                agent.memo.add_memo(s, action, r, done, trunc, s_new)
                s = s_new

                if agent.memo.can_sample():
                    batch_s, batch_a, batch_r, batch_done, batch_trunc, batch_s_new = agent.memo.sample(
                        device=agent.device
                    )

                    with torch.no_grad():
                        target_q = agent.target_network(batch_s_new)
                        max_target_q = target_q.max(dim=1, keepdim=True)[0]
                        batch_terminal = torch.maximum(batch_done, batch_trunc)
                        targets = batch_r + agent.gamma * (1 - batch_terminal) * max_target_q

                    q_values = agent.online_network(batch_s)
                    a_q = torch.gather(q_values, 1, index=batch_a)
                    loss = nn.functional.smooth_l1_loss(a_q, targets)

                    agent.optimizer.zero_grad()
                    loss.backward()
                    agent.optimizer.step()

                if done or trunc:
                    break

            episode_summary = info["episode"]
            episode_reward = episode_summary["episode_reward"]
            if episode_reward > best_episode_reward:
                best_episode_reward = episode_reward
                best_episode = episode

            writer.writerow(
                [
                    episode,
                    f"{episode_reward:.2f}",
                    f"{episode_summary['average_reward']:.4f}",
                    f"{epsilon:.4f}",
                    episode_summary["sim_seconds"],
                    episode_summary["total_departed"],
                    f"{episode_summary['avg_departed_per_sec']:.4f}",
                    episode_summary["switch_count"],
                    f"{episode_summary['average_queue']:.4f}",
                    f"{episode_summary['average_wait']:.4f}",
                    f"{episode_summary['average_imbalance']:.4f}",
                    f"{episode_summary['cumulative_queue']:.2f}",
                    f"{episode_summary['cumulative_wait']:.2f}",
                    episode_summary["final_total_queue"],
                    episode_summary["final_total_wait"],
                ]
            )

            if (episode + 1) % args.target_update == 0:
                agent.update_target_network()

            if args.print_every > 0 and episode % args.print_every == 0:
                print(
                    f"Episode {episode}: "
                    f"decisions={steps_this_episode}, "
                    f"sim_seconds={episode_summary['sim_seconds']}, "
                    f"reward={episode_reward:.2f}, "
                    f"avg_wait={episode_summary['average_wait']:.2f}, "
                    f"avg_queue={episode_summary['average_queue']:.2f}, "
                    f"avg_imbalance={episode_summary['average_imbalance']:.2f}, "
                    f"departed={episode_summary['total_departed']}, "
                    f"switches={episode_summary['switch_count']}, "
                    f"epsilon={epsilon:.3f}"
                )

    print(f"Training metrics saved to {args.log_path}")

    torch.save(agent.online_network.state_dict(), args.model_path)
    print(f"Model state_dict saved to {args.model_path}")

    checkpoint = {
        "state_dict": agent.online_network.state_dict(),
        "agent_kwargs": agent_kwargs,
        "env_kwargs": env_kwargs,
        "train_kwargs": {
            "seed": args.seed,
            "episodes": args.episodes,
            "epsilon_start": args.epsilon_start,
            "epsilon_end": args.epsilon_end,
            "epsilon_decay": args.epsilon_decay,
            "target_update": args.target_update,
        },
        "best_episode_reward": best_episode_reward,
        "best_episode": best_episode,
    }
    torch.save(checkpoint, args.checkpoint_path)
    print(f"Model checkpoint saved to {args.checkpoint_path}")

    return {
        "agent": agent,
        "env_kwargs": env_kwargs,
        "agent_kwargs": agent_kwargs,
        "best_episode_reward": best_episode_reward,
        "best_episode": best_episode,
    }


def main(cli_args=None):
    parser = build_parser()
    args = parser.parse_args(cli_args)
    train_dqn(args)


if __name__ == "__main__":
    main()
