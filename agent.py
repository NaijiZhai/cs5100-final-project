import numpy as np
import torch
from torch import nn


class ReplayBuffer:
    def __init__(self, n_s, memory_size=100000, batch_size=64):
        self.n_s = n_s
        self.memory_size = memory_size
        self.batch_size = batch_size

        self.flag = 0
        self.max_filled = 0

        self.s = np.empty((self.memory_size, self.n_s), dtype=np.float32)
        self.a = np.empty(self.memory_size, dtype=np.int64)
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

    def can_sample(self):
        return self.max_filled >= self.batch_size

    def sample(self, device="cpu"):
        indices = np.random.choice(self.max_filled, size=self.batch_size, replace=False)

        batch_s = torch.as_tensor(self.s[indices], dtype=torch.float32, device=device)
        batch_a = torch.as_tensor(self.a[indices], dtype=torch.long, device=device).unsqueeze(-1)
        batch_r = torch.as_tensor(self.r[indices], dtype=torch.float32, device=device).unsqueeze(-1)
        batch_done = torch.as_tensor(self.done[indices], dtype=torch.float32, device=device).unsqueeze(-1)
        batch_trunc = torch.as_tensor(self.trunc[indices], dtype=torch.float32, device=device).unsqueeze(-1)
        batch_s_new = torch.as_tensor(self.s_[indices], dtype=torch.float32, device=device)

        return batch_s, batch_a, batch_r, batch_done, batch_trunc, batch_s_new


# Backward-compatibility alias for previous naming.
Memotable = ReplayBuffer


class DQN(nn.Module):
    def __init__(self, n_input, n_output, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_input, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_output),
        )

    def forward(self, x):
        return self.net(x)

    def act(self, obs, device="cpu"):
        was_training = self.training
        self.eval()

        obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            q_values = self(obs_tensor)
        action = torch.argmax(q_values, dim=1).item()

        if was_training:
            self.train()

        return action


class Agent:
    def __init__(
        self,
        n_input,
        n_output,
        device=None,
        gamma=0.99,
        learning_rate=1e-3,
        hidden_dim=64,
        memory_size=100000,
        batch_size=64,
    ):
        self.n_input = n_input
        self.n_output = n_output

        self.gamma = gamma
        self.learning_rate = learning_rate
        self.hidden_dim = hidden_dim
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.memo = ReplayBuffer(
            n_input,
            memory_size=memory_size,
            batch_size=batch_size,
        )

        self.online_network = DQN(self.n_input, self.n_output, hidden_dim=hidden_dim).to(self.device)
        self.target_network = DQN(self.n_input, self.n_output, hidden_dim=hidden_dim).to(self.device)
        self.target_network.load_state_dict(self.online_network.state_dict())
        self.target_network.eval()

        for p in self.target_network.parameters():
            p.requires_grad = False

        self.optimizer = torch.optim.Adam(
            self.online_network.parameters(),
            lr=self.learning_rate,
        )

    def select_action(self, obs):
        return self.online_network.act(obs, device=self.device)

    def update_target_network(self):
        self.target_network.load_state_dict(self.online_network.state_dict())
