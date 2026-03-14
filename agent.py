import random

import numpy as np
import torch
from torch import nn


class Memotable:
    def __init__(self, n_s, n_a):
        self.n_s = n_s
        self.n_a = n_a
        self.memory_size = 100000
        self.batch_size = 64
        self.flag = 0
        self.max_filled = 0

        self.s = np.empty((self.memory_size, self.n_s), dtype=np.float32)
        self.a = np.empty(self.memory_size, dtype=np.uint8)
        self.done = np.empty(self.memory_size, dtype=np.uint8)
        self.trunc = np.empty(self.memory_size, dtype=np.uint8)
        self.r = np.empty(self.memory_size, dtype=np.float32)
        self.s_ = np.empty((self.memory_size, self.n_s), dtype=np.float32)

    def add_memo(self, s, action, r, done, trunc, s_new):
        self.s[self.flag] = s
        self.a[self.flag] = action
        self.r[self.flag] = r
        self.done[self.flag] = done
        self.trunc[self.flag] = trunc
        self.s_[self.flag] = s_new

        self.flag = (self.flag + 1) % self.memory_size
        self.max_filled = min(self.memory_size, self.max_filled + 1)

    def sample(self):
        indices = random.sample(range(self.max_filled), self.batch_size)
        indices = np.array(indices)

        batch_s = torch.tensor(self.s[indices], dtype=torch.float32)
        batch_a = torch.tensor(self.a[indices], dtype=torch.long).unsqueeze(-1)
        batch_r = torch.tensor(self.r[indices], dtype=torch.float32).unsqueeze(-1)
        batch_done = torch.tensor(self.done[indices], dtype=torch.float32).unsqueeze(-1)
        batch_trunc = torch.tensor(self.trunc[indices], dtype=torch.float32).unsqueeze(-1)
        batch_s_new = torch.tensor(self.s_[indices], dtype=torch.float32)

        return batch_s, batch_a, batch_r, batch_done, batch_trunc, batch_s_new


class DQN(nn.Module):
    def __init__(self, n_input, n_output):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_input, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, n_output),
        )

    def forward(self, x):
        return self.net(x)

    def act(self, obs):
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            q_values = self(obs_tensor)
        action = torch.argmax(q_values, dim=1).item()
        return action


class Agent:
    def __init__(self, n_input, n_output):
        self.n_input = n_input
        self.n_output = n_output
        self.gamma = 0.99
        self.learning_rate = 0.001

        self.memo = Memotable(n_input, n_output)

        self.online_network = DQN(self.n_input, self.n_output)
        self.target_network = DQN(self.n_input, self.n_output)
        self.target_network.load_state_dict(self.online_network.state_dict())
        self.target_network.eval()

        self.optimizer = torch.optim.Adam(
            self.online_network.parameters(),
            lr=self.learning_rate,
        )