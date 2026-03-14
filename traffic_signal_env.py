import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random

class TrafficSignalEnv(gym.Env):

    def __init__(
        self,
        max_steps=1000,
        arrival_prob=(0.6, 0.6, 0.6, 0.6),
        depart_capacity=2,
        min_green_steps=4,
        max_queue=100,
        max_wait=500,
    ):
        super().__init__()

        self.max_steps = max_steps
        self.arrival_prob = list(arrival_prob)
        self.depart_capacity = depart_capacity
        self.min_green_steps = min_green_steps
        self.max_queue = max_queue
        self.max_wait = max_wait

        self.action_space = spaces.Discrete(2)

        self.observation_space = spaces.Box(
            low=np.zeros(9, dtype=np.float32),
            high=np.ones(9, dtype=np.float32),
            dtype=np.float32
        )

        self.rng = random.Random()
        self.reset()

    def _get_state(self):
        q = [len(x) for x in self.lane_waits]
        w = [sum(x) for x in self.lane_waits]



        q = [min(x / self.max_queue, 1.0) for x in q]
        w = [min(x / self.max_wait, 1.0) for x in w]
        return np.array(q + w + [float(self.phase)], dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        if seed is not None:
            self.rng.seed(seed)

        self.lane_waits = [[], [], [], []]
        self.phase = 0
        self.phase_duration = 0
        self.step_count = 0
        self.total_departed = 0
        self.switch_count = 0

        return self._get_state(), {}

    def step(self, action):


        prev = self.phase
        a = int(action)

        if a != self.phase and self.phase_duration >= self.min_green_steps:
            self.phase = a
            self.phase_duration = 1
            self.switch_count += 1
        else:
            self.phase_duration += 1

        self.step_count += 1

        for i in range(4):
            if self.rng.random() < self.arrival_prob[i]:
                self.lane_waits[i].append(0)

        green = [0, 1] if self.phase == 0 else [2, 3]

        moved = 0
        for i in green:
            n = min(self.depart_capacity, len(self.lane_waits[i]))
            if n:
                self.lane_waits[i] = self.lane_waits[i][n:]
            moved += n

        self.total_departed += moved

        for i in range(4):
            self.lane_waits[i] = [x + 1 for x in self.lane_waits[i]]

        q = [len(x) for x in self.lane_waits]
        w = [sum(x) for x in self.lane_waits]
        total_queue = sum(q)
        total_wait = sum(w)

        reward = -(total_queue + 0.3 * total_wait)
        if self.phase != prev:
            reward -= 1

        terminated = False
        truncated = self.step_count >= self.max_steps

        state = self._get_state()


        info = {
            'total_queue': total_queue,
            'total_wait': total_wait,
            'total_departed': self.total_departed,
            'departed_this_step': moved,
            'switch_count': self.switch_count,
            'phase_duration': self.phase_duration,
        }

        return state, reward, terminated, truncated, info

