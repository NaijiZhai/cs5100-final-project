import os

import numpy as np
import torch

from agent import Agent
from traffic_signal_env import TrafficSignalEnv
from baseline import FixedTimePolicy, RandomPolicy, DQNPolicy


def evaluate_policy(policy, env, n_episodes=50, n_steps=1000):
    rewards = []
    total_departed_list = []
    total_queue_list = []
    total_wait_list = []
    switch_count_list = []

    for episode in range(n_episodes):
        obs, info = env.reset()
        policy.reset()

        episode_reward = 0.0
        last_info = {}

        for step in range(n_steps):
            action = policy.act(obs)
            obs, reward, done, trunc, info = env.step(action)

            episode_reward += reward
            last_info = info

            if done or trunc:
                break

        rewards.append(episode_reward)
        total_departed_list.append(last_info.get("total_departed", 0))
        total_queue_list.append(last_info.get("total_queue", 0))
        total_wait_list.append(last_info.get("total_wait", 0))
        switch_count_list.append(last_info.get("switch_count", 0))

    return {
        "avg_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
        "avg_departed": float(np.mean(total_departed_list)),
        "avg_final_queue": float(np.mean(total_queue_list)),
        "avg_final_wait": float(np.mean(total_wait_list)),
        "avg_switch_count": float(np.mean(switch_count_list)),
    }


if __name__ == "__main__":
    env = TrafficSignalEnv(max_steps=1000)

    model_path = "dqn_model.pth"
    if not os.path.exists(model_path):
        print(f"No saved model found at {model_path}. Train first with: python traffic_dqn.py")
        exit(1)

    agent = Agent(env.observation_space.shape[0], env.action_space.n)
    agent.online_network.load_state_dict(torch.load(model_path, weights_only=True))

    dqn_policy = DQNPolicy(agent)
    fixed_policy = FixedTimePolicy(switch_interval=10)
    random_policy = RandomPolicy(env.action_space)

    fixed_result = evaluate_policy(fixed_policy, env, n_episodes=50, n_steps=1000)
    random_result = evaluate_policy(random_policy, env, n_episodes=50, n_steps=1000)
    dqn_result = evaluate_policy(dqn_policy, env, n_episodes=50, n_steps=1000)

    print("Fixed-Time Policy:", fixed_result)
    print("Random Policy:", random_result)
    print("DQN Policy:", dqn_result)
