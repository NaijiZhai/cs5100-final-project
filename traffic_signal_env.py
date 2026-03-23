import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random

class TrafficSignalEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        max_steps=1000,
        arrival_prob=(0.3, 0.3, 0.3, 0.3),
        depart_capacity=2,
        min_green_steps=3,
        max_queue=50,
        max_wait=500,
        normalize_state=True,
    ):
        super().__init__()

        self.max_steps = max_steps
        self.arrival_prob = list(arrival_prob)
        self.depart_capacity = depart_capacity
        self.min_green_steps = min_green_steps
        self.max_queue = max_queue
        self.max_wait = max_wait
        self.normalize_state = normalize_state

        # 2 discrete actions: NS green or WE green
        self.action_space = spaces.Discrete(2)

        if self.normalize_state:
            self.observation_space = spaces.Box(
                low=np.zeros(9, dtype=np.float32),
                high=np.ones(9, dtype=np.float32),
                dtype=np.float32
            )
        else:
            self.observation_space = spaces.Box(
                low=np.zeros(9, dtype=np.float32),
                high=np.array(
                    [
                        self.max_queue, self.max_queue, self.max_queue, self.max_queue,
                        self.max_wait, self.max_wait, self.max_wait, self.max_wait,
                        1
                    ],
                    dtype=np.float32
                ),
                dtype=np.float32
            )

        self.rng = random.Random()
        self.reset()

    def _sync_stats(self):
        self.queues = [len(lane) for lane in self.lane_waits]
        self.waits = [sum(lane) for lane in self.lane_waits]

    def _get_state(self):
        self._sync_stats()

        raw_state = np.array(self.queues + self.waits + [self.phase], dtype=np.float32)

        if not self.normalize_state:
            return raw_state

        norm_queues = [min(q / self.max_queue, 1.0) for q in self.queues]
        norm_waits = [min(w / self.max_wait, 1.0) for w in self.waits]
        norm_phase = [float(self.phase)]

        return np.array(norm_queues + norm_waits + norm_phase, dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        if seed is not None:
            self.rng.seed(seed)

        self.lane_waits = [[], [], [], []]
        self.queues = [0, 0, 0, 0]
        self.waits = [0, 0, 0, 0]
        self.phase = 0
        self.phase_duration = 0
        self.step_count = 0
        self.total_departed = 0
        self.switch_count = 0

        state = self._get_state()
        info = {}
        return state, info

    def step(self, action):
        assert self.action_space.contains(action), f"Invalid action: {action}"

        requested_phase = int(action)
        prev_phase = self.phase

        if requested_phase != self.phase and self.phase_duration >= self.min_green_steps:
            self.phase = requested_phase
            self.phase_duration = 1
            self.switch_count += 1
        else:
            self.phase_duration += 1

        self.step_count += 1

        # arrivals
        for i in range(4):
            if self.rng.random() < self.arrival_prob[i]:
                self.lane_waits[i].append(0)

        # departures
        green_dirs = [0, 1] if self.phase == 0 else [2, 3]

        departed_this_step = 0
        for d in green_dirs:
            departed = min(self.depart_capacity, len(self.lane_waits[d]))
            if departed > 0:
                self.lane_waits[d] = self.lane_waits[d][departed:]
            departed_this_step += departed

        self.total_departed += departed_this_step

        # waiting update
        for i in range(4):
            self.lane_waits[i] = [w + 1 for w in self.lane_waits[i]]

        self._sync_stats()
        total_queue = sum(self.queues)
        total_wait = sum(self.waits)

        reward = -(total_queue + 0.1 * total_wait)
        if self.phase != prev_phase:
            reward -= 0.5

        terminated = False
        truncated = self.step_count >= self.max_steps

        state = self._get_state()
        assert self.observation_space.contains(state), f"Invalid state: {state}"

        info = {
            "total_queue": total_queue,
            "total_wait": total_wait,
            "total_departed": self.total_departed,
            "departed_this_step": departed_this_step,
            "switch_count": self.switch_count,
            "phase_duration": self.phase_duration,
        }

        return state, reward, terminated, truncated, info

    def render(self):
        print(
            f"Step={self.step_count}, Phase={self.phase}, "
            f"PhaseDuration={self.phase_duration}, Queues={self.queues}, Waits={self.waits}"
        )