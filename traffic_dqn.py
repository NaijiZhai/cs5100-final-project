import csv
import random

import numpy as np
import torch
import torch.nn as nn

from agent import Agent
from traffic_signal_env import TrafficSignalEnv

def main():
    env = TrafficSignalEnv()

    n_episode = 400
    n_steps = env.max_steps

    EPSILON_DECAY = 500000
    EPSILON_START = 1.0
    EPSILON_END = 0.05

    UPDATE_FREQUENCY = 20
    reward_buffer = np.empty(n_episode, dtype=np.float32)

    agent = Agent(env.observation_space.shape[0], env.action_space.n)

    global_step = 0

    log_path = "training_metrics.csv"
    log_file = open(log_path, "w", newline="")
    writer = csv.writer(log_file)
    writer.writerow(["episode", "reward", "epsilon", "departed", "total_queue", "total_wait", "switch_count"])

    for episode in range(n_episode):
        s, info = env.reset()
        episode_reward = 0.0
        steps_this_episode = 0

        for step in range(n_steps):
            steps_this_episode += 1
            global_step += 1

            epsilon = np.interp(
                global_step,
                [0, EPSILON_DECAY],
                [EPSILON_START, EPSILON_END],
            )

            if random.random() <= epsilon:
                action = env.action_space.sample()
            else:
                action = agent.online_network.act(s)

            s_new, r, done, trunc, info = env.step(action)
            agent.memo.add_memo(s, action, r, done, trunc, s_new)

            episode_reward += r
            s = s_new

            if agent.memo.max_filled >= agent.memo.batch_size:
                batch_s, batch_a, batch_r, batch_done, batch_trunc, batch_s_new = agent.memo.sample()

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

        reward_buffer[episode] = episode_reward

        writer.writerow([
            episode,
            f"{episode_reward:.2f}",
            f"{epsilon:.4f}",
            info.get("total_departed", 0),
            info.get("total_queue", 0),
            info.get("total_wait", 0),
            info.get("switch_count", 0),
        ])

        if episode % UPDATE_FREQUENCY == 0:
            agent.target_network.load_state_dict(agent.online_network.state_dict())
            print(
                f"Episode {episode}: "
                f"steps={steps_this_episode}, "
                f"env_step_count={env.step_count}, "
                f"Reward={episode_reward:.2f}, "
                f"Epsilon={epsilon:.3f}, "
                f"Departed={info.get('total_departed', 0)}"
            )
    log_file.close()
    print(f"Training metrics saved to {log_path}")

    save_path = "dqn_model.pth"
    torch.save(agent.online_network.state_dict(), save_path)
    print(f"Model saved to {save_path}")

    return agent


if __name__ == "__main__":
    main()
