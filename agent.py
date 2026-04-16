"""
DQN agent components for traffic signal control.

This module contains the building blocks of a Deep Q-Network (DQN) agent:
- ReplayBuffer: circular experience-replay storage
- DQN: the Q-value neural network (multi-layer perceptron)
- Agent: top-level wrapper that groups online/target networks, replay buffer,
  and optimizer into a single trainable unit
"""

import numpy as np
import torch
from torch import nn


class ReplayBuffer:
    """Fixed-size circular buffer for storing and sampling experience tuples.

    Experiences are stored in pre-allocated NumPy arrays so that memory usage
    stays constant regardless of how many transitions the agent has seen.
    Once the buffer is full, the oldest experiences are overwritten first
    (FIFO circular replacement).
    """

    def __init__(self, n_s, memory_size=100000, batch_size=64):
        self.n_s = n_s
        self.memory_size = memory_size
        self.batch_size = batch_size

        # Write-head index: points to the next slot to overwrite.
        self.flag = 0
        # Tracks how many slots have been written at least once.
        self.max_filled = 0

        # Pre-allocated arrays for each component of a transition (s, a, r, done, trunc, s'):
        self.s = np.empty((self.memory_size, self.n_s), dtype=np.float32)      # current state
        self.a = np.empty(self.memory_size, dtype=np.int64)                    # action taken
        self.done = np.empty(self.memory_size, dtype=np.uint8)                 # terminal flag
        self.trunc = np.empty(self.memory_size, dtype=np.uint8)                # truncation flag (time-limit)
        self.r = np.empty(self.memory_size, dtype=np.float32)                  # reward received
        self.s_ = np.empty((self.memory_size, self.n_s), dtype=np.float32)     # next state

    def add_memo(self, s, action, r, done, trunc, s_new):
        """Store a single transition, overwriting the oldest entry when full.

        The write-head (`flag`) advances by one after each write and wraps
        around to index 0 once it reaches `memory_size`, implementing the
        circular overwrite behaviour.
        """
        self.s[self.flag] = s
        self.a[self.flag] = action
        self.r[self.flag] = r
        self.done[self.flag] = done
        self.trunc[self.flag] = trunc
        self.s_[self.flag] = s_new

        # Advance write-head with wrap-around for circular overwrite.
        self.flag = (self.flag + 1) % self.memory_size
        # Track total filled slots, capped at memory_size.
        self.max_filled = min(self.memory_size, self.max_filled + 1)

    def can_sample(self):
        """Return True only when enough transitions exist to fill one batch.

        Sampling before the buffer holds at least `batch_size` entries would
        require replacement, which could bias early training.
        """
        return self.max_filled >= self.batch_size

    def sample(self, device="cpu"):
        """Draw a uniformly random batch of transitions and convert to tensors.

        Indices are sampled without replacement from the filled portion of the
        buffer.  Each component is converted to a PyTorch tensor on the
        requested device so it can be fed directly into the Q-network.
        """
        indices = np.random.choice(self.max_filled, size=self.batch_size, replace=False)

        batch_s = torch.as_tensor(self.s[indices], dtype=torch.float32, device=device)
        # unsqueeze(-1) adds a trailing dimension so actions/rewards/flags
        # have shape (batch, 1), matching the gather/broadcast expectations
        # of the DQN loss computation.
        batch_a = torch.as_tensor(self.a[indices], dtype=torch.long, device=device).unsqueeze(-1)
        batch_r = torch.as_tensor(self.r[indices], dtype=torch.float32, device=device).unsqueeze(-1)
        batch_done = torch.as_tensor(self.done[indices], dtype=torch.float32, device=device).unsqueeze(-1)
        batch_trunc = torch.as_tensor(self.trunc[indices], dtype=torch.float32, device=device).unsqueeze(-1)
        batch_s_new = torch.as_tensor(self.s_[indices], dtype=torch.float32, device=device)

        return batch_s, batch_a, batch_r, batch_done, batch_trunc, batch_s_new


# Backward-compatibility alias for previous naming.
Memotable = ReplayBuffer


class DQN(nn.Module):
    """Deep Q-Network implemented as a multi-layer perceptron.

    Architecture: input(n_input) → 1024 → 512 → hidden_dim → output(n_output)
    Each hidden layer uses ReLU activation.  The final layer outputs raw
    Q-values (one per discrete action) with no activation, so they can take
    any real value.
    """

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
        """Compute Q-values for every action given a batch of states."""
        return self.net(x)

    def act(self, obs, device="cpu"):
        """Select the greedy (highest-Q) action for a single observation.

        Temporarily switches the network to eval mode so that layers like
        BatchNorm or Dropout (if added later) behave deterministically, then
        restores the original training mode afterward.
        """
        # Remember current mode so we can restore it after inference.
        was_training = self.training
        self.eval()

        obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            q_values = self(obs_tensor)
        # Pick the action index with the highest Q-value.
        action = torch.argmax(q_values, dim=1).item()

        if was_training:
            self.train()

        return action


class Agent:
    """High-level DQN agent that bundles online/target networks, replay buffer,
    and optimizer into a single object.

    The online network is trained via gradient descent; the target network is a
    periodically-updated frozen copy used to compute stable TD targets, which
    reduces training oscillations (standard DQN trick).
    """

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
        # Auto-detect GPU; fall back to CPU if CUDA is unavailable.
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.memo = ReplayBuffer(
            n_input,
            memory_size=memory_size,
            batch_size=batch_size,
        )

        # Online network: actively trained each step.
        self.online_network = DQN(self.n_input, self.n_output, hidden_dim=hidden_dim).to(self.device)
        # Target network: frozen copy updated periodically for stable TD targets.
        self.target_network = DQN(self.n_input, self.n_output, hidden_dim=hidden_dim).to(self.device)
        # Initialize target weights to match the online network.
        self.target_network.load_state_dict(self.online_network.state_dict())
        self.target_network.eval()

        # Freeze target network parameters so they are excluded from
        # gradient computation and optimizer updates.
        for p in self.target_network.parameters():
            p.requires_grad = False

        self.optimizer = torch.optim.Adam(
            self.online_network.parameters(),
            lr=self.learning_rate,
        )

    def select_action(self, obs):
        """Delegate greedy action selection to the online network."""
        return self.online_network.act(obs, device=self.device)

    def update_target_network(self):
        """Hard-copy all online network weights into the target network.

        This is the standard DQN "hard update" — the target network's
        parameters are replaced wholesale rather than blended (as in soft/
        Polyak updates).
        """
        self.target_network.load_state_dict(self.online_network.state_dict())
