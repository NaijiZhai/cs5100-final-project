import argparse
import csv
import json
from argparse import Namespace
from copy import deepcopy
from pathlib import Path
from statistics import mean


ABLATION_GROUPS = {
    "duration": [
        {
            "name": "ep600",
            "overrides": {
                "episodes": 600,
                "epsilon_decay": 180000,
                "hidden_dim": 64,
            },
        },
        {
            "name": "ep1800",
            "overrides": {
                "episodes": 1800,
                "epsilon_decay": 450000,
                "hidden_dim": 64,
            },
        },
    ],
    "model_size": [
        {
            "name": "hidden64",
            "overrides": {
                "episodes": 1200,
                "epsilon_decay": 360000,
                "hidden_dim": 64,
            },
        },
        {
            "name": "hidden128",
            "overrides": {
                "episodes": 1200,
                "epsilon_decay": 360000,
                "hidden_dim": 128,
            },
        },
    ],
    "difficulty": [
        {
            "name": "symmetric",
            "overrides": {
                "episodes": 1200,
                "epsilon_decay": 600000,
                "hidden_dim": 64,
                "arrival_prob": (0.30, 0.30, 0.30, 0.30),
                "demand_variation": 0.25,
                "demand_period": 240,
                "green_delta": 2,
            },
        },
        {
            "name": "asymmetric_peaky",
            "overrides": {
                "episodes": 1200,
                "epsilon_decay": 600000,
                "hidden_dim": 64,
                "arrival_prob": (0.40, 0.20, 0.35, 0.15),
                "demand_variation": 0.45,
                "demand_period": 180,
                "green_delta": 3,
            },
        },
    ],
}

DEFAULT_TRAIN_ARGS = {
    "seed": 42,
    "episodes": 600,
    "max_decisions": 300,
    "max_steps": None,
    "control_interval": None,
    "arrival_prob": (0.30, 0.30, 0.30, 0.30),
    "demand_variation": 0.25,
    "demand_period": 240,
    "history_windows": 3,
    "green_delta": 2,
    "min_green": 6,
    "epsilon_start": 1.0,
    "epsilon_end": 0.05,
    "epsilon_decay": 180000,
    "gamma": 0.99,
    "learning_rate": 1e-3,
    "hidden_dim": 64,
    "memory_size": 100000,
    "batch_size": 64,
    "target_update": 20,
    "reward_wait_weight": 1.0,
    "reward_queue_weight": 0.2,
    "reward_imbalance_weight": 0.2,
    "reward_throughput_weight": 0.5,
    "reward_switch_weight": 0.3,
    "log_path": "training_metrics.csv",
    "model_path": "dqn_model.pth",
    "checkpoint_path": "dqn_checkpoint.pth",
    "print_every": 50,
}


def build_parser():
    parser = argparse.ArgumentParser(description="Run duration/model-size/difficulty ablations")
    parser.add_argument("--group", choices=["duration", "model_size", "difficulty", "all"], default="all")
    parser.add_argument("--seeds", type=str, default="42,43,44")
    parser.add_argument("--eval-episodes", type=int, default=50)
    parser.add_argument("--output-dir", type=str, default="results/ablations")
    parser.add_argument("--print-every", type=int, default=50)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def parse_seeds(text):
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def default_train_args():
    return Namespace(**DEFAULT_TRAIN_ARGS)


def apply_overrides(args, overrides):
    for key, value in overrides.items():
        setattr(args, key, value)


def run_single_experiment(group, exp_name, seed, overrides, output_root, eval_episodes, print_every, dry_run=False):
    run_dir = output_root / group / exp_name / f"seed_{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)

    args = default_train_args()
    args.seed = seed
    args.print_every = print_every
    args.log_path = str(run_dir / "training_metrics.csv")
    args.model_path = str(run_dir / "dqn_model.pth")
    args.checkpoint_path = str(run_dir / "dqn_checkpoint.pth")
    apply_overrides(args, overrides)

    if dry_run:
        return {
            "group": group,
            "experiment": exp_name,
            "seed": seed,
            "train_args": vars(args),
            "results": None,
        }

    from evaluate import evaluate_all_policies, load_agent_and_env
    from traffic_dqn import train_dqn

    train_info = train_dqn(args)

    agent, env, env_kwargs, agent_meta = load_agent_and_env(
        args.checkpoint_path,
        override_env_kwargs=None,
        ignore_checkpoint_env=False,
        fallback_hidden_dim=args.hidden_dim,
    )
    eval_results = evaluate_all_policies(
        env,
        agent,
        n_episodes=eval_episodes,
        seed_offset=seed,
    )

    payload = {
        "group": group,
        "experiment": exp_name,
        "seed": seed,
        "train_args": vars(args),
        "env_kwargs": env_kwargs,
        "agent_kwargs": agent_meta,
        "best_episode": train_info["best_episode"],
        "best_episode_reward": train_info["best_episode_reward"],
        "results": eval_results,
    }
    with (run_dir / "evaluation.json").open("w") as f:
        json.dump(payload, f, indent=2)

    return payload


def get_policy_metrics(rows, policy_key, metric_key):
    return [row["results"][policy_key][metric_key] for row in rows]


def aggregate_results(run_payloads):
    run_payloads = [p for p in run_payloads if p.get("results") is not None]
    grouped = {}
    for payload in run_payloads:
        key = (payload["group"], payload["experiment"])
        grouped.setdefault(key, []).append(payload)

    summary_rows = []
    for (group, exp_name), rows in grouped.items():
        dqn_rewards = get_policy_metrics(rows, "DQN Policy", "avg_episode_reward")
        dqn_waits = get_policy_metrics(rows, "DQN Policy", "avg_wait")

        fixed_keep_rewards = get_policy_metrics(rows, "Fixed-Time (Keep)", "avg_episode_reward")
        fixed_keep_waits = get_policy_metrics(rows, "Fixed-Time (Keep)", "avg_wait")

        fixed_demand_rewards = get_policy_metrics(rows, "Fixed-Time (Demand-Aware)", "avg_episode_reward")
        fixed_demand_waits = get_policy_metrics(rows, "Fixed-Time (Demand-Aware)", "avg_wait")

        fixed_tuned_rewards = get_policy_metrics(rows, "Fixed-Time (Tuned-Static)", "avg_episode_reward")
        fixed_tuned_waits = get_policy_metrics(rows, "Fixed-Time (Tuned-Static)", "avg_wait")
        tuned_actions = get_policy_metrics(rows, "Fixed-Time (Tuned-Static)", "selected_action")

        summary_rows.append(
            {
                "group": group,
                "experiment": exp_name,
                "num_seeds": len(rows),
                "dqn_avg_reward": mean(dqn_rewards),
                "fixed_keep_avg_reward": mean(fixed_keep_rewards),
                "fixed_demand_avg_reward": mean(fixed_demand_rewards),
                "fixed_tuned_avg_reward": mean(fixed_tuned_rewards),
                "reward_gain_vs_keep": mean(dqn_rewards) - mean(fixed_keep_rewards),
                "reward_gain_vs_demand": mean(dqn_rewards) - mean(fixed_demand_rewards),
                "reward_gain_vs_tuned": mean(dqn_rewards) - mean(fixed_tuned_rewards),
                "dqn_avg_wait": mean(dqn_waits),
                "fixed_keep_avg_wait": mean(fixed_keep_waits),
                "fixed_demand_avg_wait": mean(fixed_demand_waits),
                "fixed_tuned_avg_wait": mean(fixed_tuned_waits),
                "wait_reduction_vs_keep": mean(fixed_keep_waits) - mean(dqn_waits),
                "wait_reduction_vs_demand": mean(fixed_demand_waits) - mean(dqn_waits),
                "wait_reduction_vs_tuned": mean(fixed_tuned_waits) - mean(dqn_waits),
                "avg_tuned_action": mean(tuned_actions),
            }
        )

    return summary_rows


def write_summary_csv(path, rows):
    if not rows:
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "group",
        "experiment",
        "num_seeds",
        "dqn_avg_reward",
        "fixed_keep_avg_reward",
        "fixed_demand_avg_reward",
        "fixed_tuned_avg_reward",
        "reward_gain_vs_keep",
        "reward_gain_vs_demand",
        "reward_gain_vs_tuned",
        "dqn_avg_wait",
        "fixed_keep_avg_wait",
        "fixed_demand_avg_wait",
        "fixed_tuned_avg_wait",
        "wait_reduction_vs_keep",
        "wait_reduction_vs_demand",
        "wait_reduction_vs_tuned",
        "avg_tuned_action",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main(cli_args=None):
    parser = build_parser()
    args = parser.parse_args(cli_args)

    seeds = parse_seeds(args.seeds)
    output_root = Path(args.output_dir)

    groups = list(ABLATION_GROUPS.keys()) if args.group == "all" else [args.group]

    all_runs = []
    for group in groups:
        for spec in ABLATION_GROUPS[group]:
            exp_name = spec["name"]
            overrides = deepcopy(spec["overrides"])

            for seed in seeds:
                print(f"Running {group}/{exp_name} seed={seed}")
                payload = run_single_experiment(
                    group=group,
                    exp_name=exp_name,
                    seed=seed,
                    overrides=overrides,
                    output_root=output_root,
                    eval_episodes=args.eval_episodes,
                    print_every=args.print_every,
                    dry_run=args.dry_run,
                )
                all_runs.append(payload)

    if args.dry_run:
        print("Dry run only. No training executed.")
        return

    summary_rows = aggregate_results(all_runs)
    summary_path = output_root / "summary.csv"
    write_summary_csv(summary_path, summary_rows)

    print(f"Saved summary to {summary_path}")
    for row in summary_rows:
        print(
            f"[{row['group']}/{row['experiment']}] "
            f"reward_gain_vs_tuned={row['reward_gain_vs_tuned']:.2f}, "
            f"wait_reduction_vs_tuned={row['wait_reduction_vs_tuned']:.3f}, "
            f"avg_tuned_action={row['avg_tuned_action']:.2f}"
        )


if __name__ == "__main__":
    main()
