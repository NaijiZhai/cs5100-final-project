import random
from collections import deque

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class TrafficSignalEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        max_decisions=300,
        max_steps=None,
        control_interval=None,
        arrival_prob=(0.30, 0.30, 0.30, 0.30),
        demand_variation=0.25,
        demand_period=240,
        depart_capacity=2,
        base_cycle=None,
        green_delta=2,
        min_green=6,
        history_windows=3,
        max_queue=80,
        max_wait=2000,
        normalize_state=True,
        reward_wait_weight=1.0,
        reward_queue_weight=0.2,
        reward_imbalance_weight=0.2,
        reward_throughput_weight=0.5,
        reward_switch_weight=0.3,
    ):
        super().__init__()

        # max_steps is kept for backward compatibility.
        if max_steps is not None:
            max_decisions = max_steps

        self.max_decisions = max_decisions
        self.max_steps = max_decisions

        # Deprecated in cycle-level stepping; retained to avoid breaking older callers.
        self.control_interval = control_interval

        self.arrival_prob = list(arrival_prob)
        self.demand_variation = demand_variation
        self.demand_period = max(demand_period, 1)
        self.depart_capacity = depart_capacity

        self.base_cycle = base_cycle or {
            "ns_green": 10,
            "ns_yellow": 2,
            "ew_green": 10,
            "ew_yellow": 2,
        }
        self.green_delta = green_delta
        self.min_green = min_green
        self.history_windows = history_windows

        self.max_queue = max_queue
        self.max_wait = max_wait
        self.normalize_state = normalize_state

        self.reward_wait_weight = reward_wait_weight
        self.reward_queue_weight = reward_queue_weight
        self.reward_imbalance_weight = reward_imbalance_weight
        self.reward_throughput_weight = reward_throughput_weight
        self.reward_switch_weight = reward_switch_weight

        # 3 actions:
        # 0 -> favor NS (increase NS green, decrease EW green)
        # 1 -> keep base cycle
        # 2 -> favor EW (decrease NS green, increase EW green)
        self.action_space = spaces.Discrete(3)

        self._max_cycle_seconds = max(
            sum(self._build_cycle_durations(a)) for a in range(self.action_space.n)
        )

        # Observation:
        # [4 queue] + [4 wait] + [K*4 arrivals] + [K*4 departures]
        # + [phase_id, phase_remaining_ratio, last_action, prev_action]
        self.obs_dim = 8 + (self.history_windows * 4 * 2) + 4

        self.max_arrivals_per_window = self._max_cycle_seconds
        self.max_departures_per_window = self._max_cycle_seconds * self.depart_capacity

        if self.normalize_state:
            self.observation_space = spaces.Box(
                low=np.zeros(self.obs_dim, dtype=np.float32),
                high=np.ones(self.obs_dim, dtype=np.float32),
                dtype=np.float32,
            )
        else:
            highs = []
            highs.extend([self.max_queue] * 4)
            highs.extend([self.max_wait] * 4)
            highs.extend([self.max_arrivals_per_window] * (self.history_windows * 4))
            highs.extend([self.max_departures_per_window] * (self.history_windows * 4))
            highs.extend([3, 1.0, 2, 2])

            self.observation_space = spaces.Box(
                low=np.zeros(self.obs_dim, dtype=np.float32),
                high=np.array(highs, dtype=np.float32),
                dtype=np.float32,
            )

        self.rng = random.Random()
        self._demand_phase_shift = [0.0, np.pi / 5, np.pi, np.pi + np.pi / 5]

        self.reset()

    def _build_cycle_durations(self, action):
        action = int(action)
        if action == 0:
            ns_green = self.base_cycle["ns_green"] + self.green_delta
            ew_green = self.base_cycle["ew_green"] - self.green_delta
        elif action == 2:
            ns_green = self.base_cycle["ns_green"] - self.green_delta
            ew_green = self.base_cycle["ew_green"] + self.green_delta
        else:
            ns_green = self.base_cycle["ns_green"]
            ew_green = self.base_cycle["ew_green"]

        ns_green = max(self.min_green, ns_green)
        ew_green = max(self.min_green, ew_green)

        return [
            ns_green,
            self.base_cycle["ns_yellow"],
            ew_green,
            self.base_cycle["ew_yellow"],
        ]

    def _sync_stats(self):
        self.queues = [len(lane) for lane in self.lane_waits]
        self.waits = [sum(lane) for lane in self.lane_waits]

    def _get_phase_remaining_ratio(self):
        duration = max(self.current_cycle_durations[self.current_phase], 1)
        remaining = max(duration - self.phase_elapsed, 0)
        return remaining / duration

    def _flatten_history(self, history):
        flat = []
        for h in history:
            flat.extend(h)
        return flat

    @staticmethod
    def _normalize_action_id(action_id):
        return float(action_id) / 2.0

    def _get_state(self):
        self._sync_stats()

        arrivals_hist = self._flatten_history(self.arrival_history)
        departures_hist = self._flatten_history(self.departure_history)

        phase_ratio = self._get_phase_remaining_ratio()

        raw_state = np.array(
            self.queues
            + self.waits
            + arrivals_hist
            + departures_hist
            + [
                self.current_phase,
                phase_ratio,
                self.last_action,
                self.prev_action,
            ],
            dtype=np.float32,
        )

        if not self.normalize_state:
            return raw_state

        norm_queues = [min(q / self.max_queue, 1.0) for q in self.queues]
        norm_waits = [min(w / self.max_wait, 1.0) for w in self.waits]
        norm_arrivals = [min(a / self.max_arrivals_per_window, 1.0) for a in arrivals_hist]
        norm_departures = [min(d / self.max_departures_per_window, 1.0) for d in departures_hist]

        return np.array(
            norm_queues
            + norm_waits
            + norm_arrivals
            + norm_departures
            + [
                self.current_phase / 3.0,
                phase_ratio,
                self._normalize_action_id(self.last_action),
                self._normalize_action_id(self.prev_action),
            ],
            dtype=np.float32,
        )

    def _get_episode_summary(self):
        avg_queue = self.cumulative_queue / max(self.total_sim_seconds, 1)
        avg_wait = self.cumulative_wait / max(self.total_sim_seconds, 1)
        avg_reward = self.cumulative_reward / max(self.step_count, 1)
        avg_imbalance = self.cumulative_imbalance / max(self.total_sim_seconds, 1)

        return {
            "episode_decisions": self.step_count,
            "episode_steps": self.step_count,
            "sim_seconds": self.total_sim_seconds,
            "episode_reward": self.cumulative_reward,
            "average_reward": avg_reward,
            "average_queue": avg_queue,
            "average_wait": avg_wait,
            "average_imbalance": avg_imbalance,
            "cumulative_queue": self.cumulative_queue,
            "cumulative_wait": self.cumulative_wait,
            "total_departed": self.total_departed,
            "avg_departed_per_sec": self.total_departed / max(self.total_sim_seconds, 1),
            "switch_count": self.action_change_count,
            "final_total_queue": sum(self.queues),
            "final_total_wait": sum(self.waits),
        }

    def _get_arrival_probabilities(self):
        probs = []
        cycle_position = (2 * np.pi * self.total_sim_seconds) / self.demand_period

        for lane_idx in range(4):
            base = self.arrival_prob[lane_idx]
            wave = np.sin(cycle_position + self._demand_phase_shift[lane_idx])
            p = base * (1.0 + self.demand_variation * wave)
            probs.append(float(np.clip(p, 0.01, 0.95)))

        return probs

    def _simulate_one_second_in_phase(self, phase_idx, arrivals_interval, departures_interval):
        probs = self._get_arrival_probabilities()
        for lane_idx in range(4):
            if self.rng.random() < probs[lane_idx]:
                self.lane_waits[lane_idx].append(0)
                arrivals_interval[lane_idx] += 1

        green_lanes = []
        if phase_idx == 0:
            green_lanes = [0, 1]
        elif phase_idx == 2:
            green_lanes = [2, 3]

        departed_this_second = 0
        for lane_idx in green_lanes:
            departed = min(self.depart_capacity, len(self.lane_waits[lane_idx]))
            if departed > 0:
                self.lane_waits[lane_idx] = self.lane_waits[lane_idx][departed:]
            departures_interval[lane_idx] += departed
            departed_this_second += departed

        self.total_departed += departed_this_second

        for lane_idx in range(4):
            self.lane_waits[lane_idx] = [w + 1 for w in self.lane_waits[lane_idx]]

        self.total_sim_seconds += 1
        self._sync_stats()

        total_queue = sum(self.queues)
        total_wait = sum(self.waits)
        ns_queue = self.queues[0] + self.queues[1]
        ew_queue = self.queues[2] + self.queues[3]
        imbalance = abs(ns_queue - ew_queue)

        self.cumulative_queue += total_queue
        self.cumulative_wait += total_wait
        self.cumulative_imbalance += imbalance

        return total_queue, total_wait, imbalance, departed_this_second

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        if seed is not None:
            self.rng.seed(seed)

        self.lane_waits = [[], [], [], []]
        self.queues = [0, 0, 0, 0]
        self.waits = [0, 0, 0, 0]

        self.current_phase = 0
        self.phase_elapsed = 0

        keep_action = 1
        self.last_action = keep_action
        self.prev_action = keep_action

        self.current_cycle_durations = self._build_cycle_durations(keep_action)

        self.arrival_history = deque(
            [[0, 0, 0, 0] for _ in range(self.history_windows)],
            maxlen=self.history_windows,
        )
        self.departure_history = deque(
            [[0, 0, 0, 0] for _ in range(self.history_windows)],
            maxlen=self.history_windows,
        )

        self.step_count = 0
        self.total_sim_seconds = 0
        self.total_departed = 0
        self.action_change_count = 0

        self.cumulative_reward = 0.0
        self.cumulative_queue = 0.0
        self.cumulative_wait = 0.0
        self.cumulative_imbalance = 0.0

        state = self._get_state()
        info = self._get_episode_summary()
        return state, info

    def step(self, action):
        assert self.action_space.contains(action), f"Invalid action: {action}"

        action = int(action)
        self.step_count += 1

        action_changed = int(action != self.last_action)
        if action_changed:
            self.action_change_count += 1

        self.current_cycle_durations = self._build_cycle_durations(action)
        cycle_seconds = sum(self.current_cycle_durations)

        arrivals_interval = [0, 0, 0, 0]
        departures_interval = [0, 0, 0, 0]

        queue_sum = 0.0
        wait_sum = 0.0
        imbalance_sum = 0.0
        departed_interval = 0

        for phase_idx, phase_duration in enumerate(self.current_cycle_durations):
            self.current_phase = phase_idx
            self.phase_elapsed = 0

            for _ in range(phase_duration):
                total_queue, total_wait, imbalance, departed_this_second = self._simulate_one_second_in_phase(
                    phase_idx,
                    arrivals_interval,
                    departures_interval,
                )
                queue_sum += total_queue
                wait_sum += total_wait
                imbalance_sum += imbalance
                departed_interval += departed_this_second
                self.phase_elapsed += 1

        mean_queue = queue_sum / cycle_seconds
        mean_wait = wait_sum / cycle_seconds
        mean_imbalance = imbalance_sum / cycle_seconds
        throughput_rate = departed_interval / cycle_seconds

        reward = (
            -self.reward_wait_weight * mean_wait
            -self.reward_queue_weight * mean_queue
            -self.reward_imbalance_weight * mean_imbalance
            +self.reward_throughput_weight * throughput_rate
            -self.reward_switch_weight * action_changed
        )

        self.cumulative_reward += reward

        self.arrival_history.append(arrivals_interval)
        self.departure_history.append(departures_interval)

        self.prev_action = self.last_action
        self.last_action = action

        # Start next decision from cycle boundary.
        self.current_phase = 0
        self.phase_elapsed = 0

        terminated = False
        truncated = self.step_count >= self.max_decisions

        state = self._get_state()
        assert self.observation_space.contains(state), f"Invalid state: {state}"

        info = {
            "total_queue": sum(self.queues),
            "total_wait": sum(self.waits),
            "total_departed": self.total_departed,
            "departed_this_step": departed_interval,
            "arrivals_this_step": arrivals_interval,
            "departures_this_step": departures_interval,
            "switch_count": self.action_change_count,
            "mean_queue": mean_queue,
            "mean_wait": mean_wait,
            "mean_imbalance": mean_imbalance,
            "throughput_rate": throughput_rate,
            "cycle_seconds": cycle_seconds,
            "phase": self.current_phase,
            "phase_remaining_ratio": self._get_phase_remaining_ratio(),
            "cycle_durations": list(self.current_cycle_durations),
        }

        if truncated or terminated:
            info["episode"] = self._get_episode_summary()

        return state, reward, terminated, truncated, info

    def render(self):
        print(
            f"DecisionStep={self.step_count}, "
            f"SimSeconds={self.total_sim_seconds}, "
            f"Phase={self.current_phase}, "
            f"PhaseElapsed={self.phase_elapsed}, "
            f"Cycle={self.current_cycle_durations}, "
            f"Queues={self.queues}, "
            f"Waits={self.waits}, "
            f"TotalDeparted={self.total_departed}, "
            f"ActionChanges={self.action_change_count}"
        )
