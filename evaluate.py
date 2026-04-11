import argparse
import csv
import json
import os

import numpy as np
import torch

from agent import Agent
from baseline import (
    DQNPolicy,
    DemandAwareFixedTimePolicy,
    FixedTimePolicy,
    QueueBasedPolicy,
    RandomPolicy,
    StaticActionPolicy,
)
from traffic_signal_env import TrafficSignalEnv


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
    parser = argparse.ArgumentParser(description="Evaluate fixed/random/queue/DQN policies")

    parser.add_argument("--model-path", type=str, default="", help="Checkpoint path (.pth)")
    parser.add_argument("--n-episodes", type=int, default=50)
    parser.add_argument("--seed-offset", type=int, default=0)

    parser.add_argument("--hidden-dim", type=int, default=64, help="Used only for plain state_dict models")
    parser.add_argument("--ignore-checkpoint-env", action="store_true")

    parser.add_argument("--max-decisions", type=int, default=300)
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

    parser.add_argument("--output-json", type=str, default="")

    return parser


def build_env_kwargs_from_args(args):
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
    }


def evaluate_policy(policy, env, n_episodes=50, seed_offset=0, track_queue=False):
    results = []
    queue_traces = []

    for episode in range(n_episodes):
        obs, info = env.reset(seed=seed_offset + episode)
        policy.reset()

        step_queues = []
        done = False
        step = 0
        while not done:
            action = policy.act(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            if track_queue:
                step_queues.append(info.get("total_queue", 0))
            step += 1

        results.append(info["episode"])
        if track_queue:
            queue_traces.append(step_queues)

    summary = {
        "avg_episode_reward": float(np.mean([r["episode_reward"] for r in results])),
        "std_episode_reward": float(np.std([r["episode_reward"] for r in results])),
        "avg_reward_per_step": float(np.mean([r["average_reward"] for r in results])),
        "avg_queue": float(np.mean([r["average_queue"] for r in results])),
        "std_queue": float(np.std([r["average_queue"] for r in results])),
        "avg_wait": float(np.mean([r["average_wait"] for r in results])),
        "std_wait": float(np.std([r["average_wait"] for r in results])),
        "avg_imbalance": float(np.mean([r["average_imbalance"] for r in results])),
        "avg_cumulative_queue": float(np.mean([r["cumulative_queue"] for r in results])),
        "avg_cumulative_wait": float(np.mean([r["cumulative_wait"] for r in results])),
        "avg_departed": float(np.mean([r["total_departed"] for r in results])),
        "avg_departed_per_sec": float(np.mean([r["avg_departed_per_sec"] for r in results])),
        "avg_switch_count": float(np.mean([r["switch_count"] for r in results])),
        "avg_final_queue": float(np.mean([r["final_total_queue"] for r in results])),
        "avg_final_wait": float(np.mean([r["final_total_wait"] for r in results])),
    }

    return summary, results, queue_traces


def evaluate_static_actions(env, n_episodes=50, seed_offset=0, candidate_actions=(0, 1, 2)):
    action_summaries = {}
    best_action = None
    best_summary = None
    best_reward = float("-inf")

    for action in candidate_actions:
        policy = StaticActionPolicy(action=action)
        summary, _, _ = evaluate_policy(policy, env, n_episodes=n_episodes, seed_offset=seed_offset)
        action_summaries[action] = summary

        if summary["avg_episode_reward"] > best_reward:
            best_reward = summary["avg_episode_reward"]
            best_action = action
            best_summary = summary

    return best_action, best_summary, action_summaries


def auto_model_path(requested_path):
    if requested_path:
        return requested_path

    if os.path.exists("dqn_checkpoint.pth"):
        return "dqn_checkpoint.pth"
    if os.path.exists("dqn_model.pth"):
        return "dqn_model.pth"
    return ""


def safe_torch_load(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def load_agent_and_env(model_path, override_env_kwargs=None, ignore_checkpoint_env=False, fallback_hidden_dim=64):
    loaded = safe_torch_load(model_path)

    checkpoint_env_kwargs = None
    checkpoint_agent_kwargs = {}

    if isinstance(loaded, dict) and "state_dict" in loaded:
        state_dict = loaded["state_dict"]
        checkpoint_env_kwargs = loaded.get("env_kwargs")
        checkpoint_agent_kwargs = loaded.get("agent_kwargs", {})
    else:
        state_dict = loaded

    env_kwargs = override_env_kwargs
    if checkpoint_env_kwargs and not ignore_checkpoint_env:
        env_kwargs = checkpoint_env_kwargs

    if env_kwargs is None:
        raise ValueError("env kwargs are not available; provide overrides or use checkpoint with env metadata")

    env = TrafficSignalEnv(**env_kwargs)

    hidden_dim = int(checkpoint_agent_kwargs.get("hidden_dim", fallback_hidden_dim))
    gamma = float(checkpoint_agent_kwargs.get("gamma", 0.99))
    learning_rate = float(checkpoint_agent_kwargs.get("learning_rate", 1e-3))

    agent = Agent(
        env.observation_space.shape[0],
        env.action_space.n,
        hidden_dim=hidden_dim,
        gamma=gamma,
        learning_rate=learning_rate,
    )
    agent.online_network.load_state_dict(state_dict)
    agent.online_network.eval()

    return agent, env, env_kwargs, {
        "hidden_dim": hidden_dim,
        "gamma": gamma,
        "learning_rate": learning_rate,
    }


def evaluate_all_policies(env, agent, n_episodes=50, seed_offset=0):
    dqn_policy = DQNPolicy(agent)
    fixed_keep_policy = FixedTimePolicy(action=1)
    fixed_demand_policy = DemandAwareFixedTimePolicy(arrival_prob=env.arrival_prob)
    random_policy = RandomPolicy(env.action_space)
    queue_policy = QueueBasedPolicy(deadband=0.05)

    fixed_keep_result, fixed_keep_raw, fixed_keep_queues = evaluate_policy(
        fixed_keep_policy, env, n_episodes=n_episodes, seed_offset=seed_offset, track_queue=True,
    )
    fixed_demand_result, fixed_demand_raw, _ = evaluate_policy(
        fixed_demand_policy, env, n_episodes=n_episodes, seed_offset=seed_offset,
    )
    tuned_action, tuned_result, static_summaries = evaluate_static_actions(
        env, n_episodes=n_episodes, seed_offset=seed_offset, candidate_actions=(0, 1, 2),
    )
    random_result, random_raw, _ = evaluate_policy(
        random_policy, env, n_episodes=n_episodes, seed_offset=seed_offset,
    )
    queue_result, queue_raw, _ = evaluate_policy(
        queue_policy, env, n_episodes=n_episodes, seed_offset=seed_offset,
    )
    dqn_result, dqn_raw, dqn_queues = evaluate_policy(
        dqn_policy, env, n_episodes=n_episodes, seed_offset=seed_offset, track_queue=True,
    )

    fixed_demand_result = dict(fixed_demand_result)
    fixed_demand_result["selected_action"] = fixed_demand_policy.action

    tuned_result = dict(tuned_result)
    tuned_result["selected_action"] = tuned_action

    summaries = {
        "Fixed-Time (Keep)": fixed_keep_result,
        "Fixed-Time (Demand-Aware)": fixed_demand_result,
        "Fixed-Time (Tuned-Static)": tuned_result,
        "Fixed-Time (Static a=0)": static_summaries[0],
        "Fixed-Time (Static a=1)": static_summaries[1],
        "Fixed-Time (Static a=2)": static_summaries[2],
        "Random Policy": random_result,
        "Queue-Based Policy": queue_result,
        "DQN Policy": dqn_result,
    }

    raw_results = {
        "Fixed-Time (Keep)": fixed_keep_raw,
        "Fixed-Time (Demand-Aware)": fixed_demand_raw,
        "Random Policy": random_raw,
        "Queue-Based Policy": queue_raw,
        "DQN Policy": dqn_raw,
    }

    queue_evolution = {
        "Fixed-Time (Keep)": fixed_keep_queues,
        "DQN Policy": dqn_queues,
    }

    return summaries, raw_results, queue_evolution


def save_summary_csv(summaries, path="eval_summary.csv"):
    rows = []
    for name, s in summaries.items():
        rows.append({
            "policy": name,
            "avg_reward": f"{s['avg_episode_reward']:.2f}",
            "std_reward": f"{s['std_episode_reward']:.2f}",
            "avg_wait": f"{s['avg_wait']:.4f}",
            "std_wait": f"{s.get('std_wait', 0):.4f}",
            "avg_queue": f"{s['avg_queue']:.4f}",
            "std_queue": f"{s.get('std_queue', 0):.4f}",
            "avg_departed": f"{s['avg_departed']:.1f}",
            "avg_switch_count": f"{s['avg_switch_count']:.1f}",
        })
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved summary to {path}")


def save_raw_csv(raw_results, seed_offset, path="eval_raw.csv"):
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["policy", "episode", "seed", "reward", "avg_wait", "avg_queue", "departed", "switches"])
        for name, episodes in raw_results.items():
            for i, ep in enumerate(episodes):
                writer.writerow([
                    name, i, seed_offset + i,
                    f"{ep['episode_reward']:.2f}",
                    f"{ep['average_wait']:.4f}",
                    f"{ep['average_queue']:.4f}",
                    ep["total_departed"],
                    ep["switch_count"],
                ])
    print(f"Saved raw results to {path}")


def save_queue_evolution_csv(queue_evolution, path="eval_queue_evolution.csv"):
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["policy", "episode", "step", "queue_length"])
        for name, traces in queue_evolution.items():
            for ep_idx, trace in enumerate(traces):
                for step, q in enumerate(trace):
                    writer.writerow([name, ep_idx, step, q])
    print(f"Saved queue evolution to {path}")


def main(cli_args=None):
    parser = build_parser()
    args = parser.parse_args(cli_args)

    model_path = auto_model_path(args.model_path)
    if not model_path:
        print("No model found. Train first with: python traffic_dqn.py")
        raise SystemExit(1)

    override_env_kwargs = build_env_kwargs_from_args(args)

    agent, env, env_kwargs, agent_meta = load_agent_and_env(
        model_path,
        override_env_kwargs=override_env_kwargs,
        ignore_checkpoint_env=args.ignore_checkpoint_env,
        fallback_hidden_dim=args.hidden_dim,
    )

    summaries, raw_results, queue_evolution = evaluate_all_policies(
        env,
        agent,
        n_episodes=args.n_episodes,
        seed_offset=args.seed_offset,
    )

    print(f"Model: {model_path}")
    print(f"Env kwargs: {env_kwargs}")
    print(f"Agent kwargs: {agent_meta}")
    for name, summary in summaries.items():
        print(f"{name}: {summary}")

    save_summary_csv(summaries)
    save_raw_csv(raw_results, args.seed_offset)
    save_queue_evolution_csv(queue_evolution)

    if args.output_json:
        payload = {
            "model_path": model_path,
            "env_kwargs": env_kwargs,
            "agent_kwargs": agent_meta,
            "results": summaries,
        }
        with open(args.output_json, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"Saved evaluation summary to {args.output_json}")


if __name__ == "__main__":
    main()
