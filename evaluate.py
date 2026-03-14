import numpy as np

import traffic_dqn
from baseline import FixedTimePolicy, RandomPolicy, DQNPolicy
from traffic_signal_env import TrafficSignalEnv

EVAL_EPISODES = 50
EVAL_STEPS = 1000
FIXED_SWITCH_INTERVAL = 10


def evaluate_policy(policy, env, n_episodes=EVAL_EPISODES, n_steps=EVAL_STEPS):
    episode_stats = []

    for episode_idx in range(n_episodes):
        obs, _ = env.reset()
        policy.reset()

        episode_reward = 0.0
        final_info = None

        for _ in range(n_steps):
            action = policy.act(obs)
            obs, reward, done, trunc, info = env.step(action)

            episode_reward += reward
            final_info = info

            if done or trunc:
                break

        episode_stats.append(
            {
                'reward': episode_reward,
                'departed': final_info['total_departed'],
                'final_queue': final_info['total_queue'],
                'final_wait': final_info['total_wait'],
                'switch_count': final_info['switch_count'],
            }
        )

    rewards = [item['reward'] for item in episode_stats]
    departed = [item['departed'] for item in episode_stats]
    final_queue = [item['final_queue'] for item in episode_stats]
    final_wait = [item['final_wait'] for item in episode_stats]
    switch_count = [item['switch_count'] for item in episode_stats]

    return {
        'avg_reward': float(np.mean(rewards)),
        'avg_departed': float(np.mean(departed)),
        'avg_final_queue': float(np.mean(final_queue)),
        'avg_final_wait': float(np.mean(final_wait)),
        'avg_switch_count': float(np.mean(switch_count)),
    }


def main():
    env = TrafficSignalEnv(max_steps=EVAL_STEPS)
    dqn_model = traffic_dqn.main()

    policies = {
        'fixed': FixedTimePolicy(switch_interval=FIXED_SWITCH_INTERVAL),
        'random': RandomPolicy(env.action_space),
        'dqn': DQNPolicy(dqn_model),
    }
    results = {}
    for name, policy in policies.items():
        results[name] = evaluate_policy(policy, env)
        print(
            f'{name}' f'reward={results[name]["avg_reward"]:.2f} ' f'departed={results[name]["avg_departed"]:.2f}'
            f'queue={results[name]["avg_final_queue"]:.2f}' f'wait={results[name]["avg_final_wait"]:.2f}'
            f'switch={results[name]["avg_switch_count"]:.2f}'
        )


if __name__ == "__main__":
    main()
