"""Gymnasium environment simulating a single 4-way intersection with cycle-level control.

Each decision step corresponds to one full signal cycle (NS green → NS yellow →
EW green → EW yellow).  The agent chooses how to redistribute green time between
the north-south and east-west directions, and the environment simulates traffic
arrivals and departures second-by-second within that cycle.
"""

import random
from collections import deque

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class TrafficSignalEnv(gym.Env):
    """Gymnasium environment for a single 4-way traffic intersection.

    The signal repeats a fixed 4-phase cycle every decision step:
        Phase 0 — NS green  : north-south lanes get a green light
        Phase 1 — NS yellow : north-south transition (no departures)
        Phase 2 — EW green  : east-west lanes get a green light
        Phase 3 — EW yellow : east-west transition (no departures)

    The agent selects one of 3 discrete actions each cycle:
        0 — Favor NS : increase NS green time by ``green_delta``, decrease EW
        1 — Keep      : use the base cycle durations unchanged
        2 — Favor EW : increase EW green time by ``green_delta``, decrease NS

    Stepping is *cycle-level*: one call to ``step()`` simulates the entire
    4-phase cycle second-by-second, then returns the resulting observation,
    reward, and info dict.
    """

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
        """Initialise the traffic signal environment.

        Parameters are organised into several groups:

        **Timing / cycle parameters**
            max_decisions : maximum number of agent decisions (cycles) per episode.
            max_steps     : legacy alias for *max_decisions* (backward compat).
            control_interval : deprecated; kept so old callers don't break.
            base_cycle    : dict with default green/yellow durations for each
                            direction (keys: ns_green, ns_yellow, ew_green,
                            ew_yellow).
            green_delta   : seconds added/subtracted from green when the agent
                            favors one direction.
            min_green     : hard lower bound on green duration after adjustment.

        **Demand parameters**
            arrival_prob      : base per-second arrival probability for each of
                                the 4 lanes [NS-0, NS-1, EW-0, EW-1].
            demand_variation  : amplitude of sinusoidal demand fluctuation
                                (fraction of base probability).
            demand_period     : period of the sinusoidal demand wave (seconds).
            depart_capacity   : max vehicles that can depart a lane per second
                                during a green phase.

        **Reward weights**
            reward_wait_weight      : penalty coefficient for mean waiting time.
            reward_queue_weight     : penalty coefficient for mean queue length.
            reward_imbalance_weight : penalty for NS-vs-EW queue imbalance.
            reward_throughput_weight: bonus coefficient for throughput rate.
            reward_switch_weight    : penalty for changing the action from the
                                      previous cycle.

        **Observation / history**
            history_windows : number of past cycles of arrival/departure counts
                              kept in the observation.
            max_queue       : clipping ceiling for queue normalisation.
            max_wait        : clipping ceiling for wait normalisation.
            normalize_state : if True, all observation components are scaled to
                              [0, 1]; otherwise raw values are returned.

        The **observation space** is a flat vector composed of:
            [4 queue lengths] + [4 cumulative waits] +
            [K×4 arrival counts] + [K×4 departure counts] +
            [phase_id, phase_remaining_ratio, last_action, prev_action]
        where K = history_windows.

        The **action space** is ``Discrete(3)`` — see class docstring.
        """
        super().__init__()

        # max_steps is kept for backward compatibility.
        if max_steps is not None:
            max_decisions = max_steps

        self.max_decisions = max_decisions
        self.max_steps = max_decisions

        # Deprecated in cycle-level stepping; retained to avoid breaking older callers.
        self.control_interval = control_interval

        # --- Demand parameters ---
        self.arrival_prob = list(arrival_prob)
        self.demand_variation = demand_variation
        self.demand_period = max(demand_period, 1)  # floor at 1 to avoid division by zero
        self.depart_capacity = depart_capacity

        # --- Cycle / green-time parameters ---
        self.base_cycle = base_cycle or {
            "ns_green": 10,
            "ns_yellow": 2,
            "ew_green": 10,
            "ew_yellow": 2,
        }
        self.green_delta = green_delta   # seconds to shift between NS/EW green
        self.min_green = min_green       # hard lower bound after delta adjustment
        self.history_windows = history_windows  # how many past cycles to include in obs

        # --- Normalisation ceilings ---
        self.max_queue = max_queue
        self.max_wait = max_wait
        self.normalize_state = normalize_state

        # --- Reward shaping weights ---
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

        # Pre-compute the longest possible cycle (used for normalisation ceilings).
        self._max_cycle_seconds = max(
            sum(self._build_cycle_durations(a)) for a in range(self.action_space.n)
        )

        # Observation:
        # [4 queue] + [4 wait] + [K*4 arrivals] + [K*4 departures]
        # + [phase_id, phase_remaining_ratio, last_action, prev_action]
        self.obs_dim = 8 + (self.history_windows * 4 * 2) + 4

        # Upper bounds for per-window arrival/departure counts (for normalisation).
        self.max_arrivals_per_window = self._max_cycle_seconds
        self.max_departures_per_window = self._max_cycle_seconds * self.depart_capacity

        # When normalised, every observation component lives in [0, 1].
        if self.normalize_state:
            self.observation_space = spaces.Box(
                low=np.zeros(self.obs_dim, dtype=np.float32),
                high=np.ones(self.obs_dim, dtype=np.float32),
                dtype=np.float32,
            )
        else:
            # Raw (un-normalised) observation space with explicit upper bounds.
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

        self.rng = random.Random()  # local RNG for reproducibility via seed
        # Per-lane phase shifts so demand peaks don't hit all lanes simultaneously.
        self._demand_phase_shift = [0.0, np.pi / 5, np.pi, np.pi + np.pi / 5]

        self.reset()

    def _build_cycle_durations(self, action):
        """Translate an action into the 4-phase duration list [ns_green, ns_yellow, ew_green, ew_yellow].

        Action 0 adds ``green_delta`` seconds to NS green and subtracts it from
        EW green.  Action 2 does the opposite.  Action 1 leaves both unchanged.
        After adjustment, each green duration is clamped to at least ``min_green``
        so the signal never stays green for an unreasonably short time.
        """
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

        # Clamp so neither direction gets less than the minimum green time.
        ns_green = max(self.min_green, ns_green)
        ew_green = max(self.min_green, ew_green)

        return [
            ns_green,
            self.base_cycle["ns_yellow"],
            ew_green,
            self.base_cycle["ew_yellow"],
        ]

    def _sync_stats(self):
        """Recompute ``queues`` and ``waits`` from the authoritative ``lane_waits`` lists.

        ``queues[i]`` is the number of vehicles waiting in lane *i* (list length).
        ``waits[i]`` is the total accumulated wait across all vehicles in lane *i*
        (sum of per-vehicle wait counters).  This is called after every simulated
        second so that downstream code always sees consistent statistics.
        """
        self.queues = [len(lane) for lane in self.lane_waits]
        self.waits = [sum(lane) for lane in self.lane_waits]

    def _get_phase_remaining_ratio(self):
        """Return the fraction of the current phase's duration that has not yet elapsed.

        The ratio is ``(total_duration - elapsed) / total_duration``, clamped to
        [0, 1].  A value of 1.0 means the phase just started; 0.0 means it is
        about to end.  The denominator is floored at 1 to avoid division by zero
        for zero-length phases.
        """
        duration = max(self.current_cycle_durations[self.current_phase], 1)
        remaining = max(duration - self.phase_elapsed, 0)
        return remaining / duration

    def _flatten_history(self, history):
        """Flatten a deque of per-lane lists into a single flat list.

        ``history`` is a deque of K entries, each a list of 4 values (one per
        lane).  This concatenates them into a single list of length K×4 suitable
        for inclusion in the observation vector.
        """
        flat = []
        for h in history:
            flat.extend(h)
        return flat

    @staticmethod
    def _normalize_action_id(action_id):
        """Scale a discrete action id (0, 1, or 2) into the [0, 1] range.

        This is used when building the normalised observation vector so that
        the action features have the same scale as other normalised components.
        Dividing by 2 maps {0 → 0.0, 1 → 0.5, 2 → 1.0}.
        """
        return float(action_id) / 2.0

    def _get_state(self):
        """Construct the observation vector for the current environment state.

        The vector is assembled as:
            [4 queue lengths] + [4 cumulative waits]
            + [K×4 arrival history] + [K×4 departure history]
            + [current_phase, phase_remaining_ratio, last_action, prev_action]

        If ``normalize_state`` is True, each component group is divided by its
        respective ceiling (max_queue, max_wait, max_arrivals_per_window, etc.)
        and clipped to [0, 1].  Phase id is divided by 3 (max phase index) and
        action ids are scaled via ``_normalize_action_id``.
        """
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

        # --- Normalise each group to [0, 1] by dividing by its ceiling ---
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
        """Compute aggregated performance metrics for the episode so far.

        Returns a dict with averages (per simulated second or per decision),
        cumulative totals, and final snapshot values.  This is included in the
        ``info`` dict on the last step of an episode and is also returned by
        ``reset()`` (with zeroed counters) for logging convenience.
        """
        # Use max(..., 1) to guard against division by zero at episode start.
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
        """Return the current per-lane arrival probabilities with sinusoidal demand variation.

        Demand follows a sine wave over ``demand_period`` simulated seconds.
        Each lane has its own phase shift (stored in ``_demand_phase_shift``) so
        that traffic peaks arrive at different times for different directions,
        mimicking realistic rush-hour patterns.  The result is clipped to
        [0.01, 0.95] to keep probabilities sensible.
        """
        probs = []
        # Convert elapsed sim-seconds into a position on the sine wave.
        cycle_position = (2 * np.pi * self.total_sim_seconds) / self.demand_period

        for lane_idx in range(4):
            base = self.arrival_prob[lane_idx]
            # Modulate base probability with a sine wave; demand_variation
            # controls the amplitude (e.g. 0.25 means ±25% of base).
            wave = np.sin(cycle_position + self._demand_phase_shift[lane_idx])
            p = base * (1.0 + self.demand_variation * wave)
            probs.append(float(np.clip(p, 0.01, 0.95)))

        return probs

    def _simulate_one_second_in_phase(self, phase_idx, arrivals_interval, departures_interval):
        """Simulate a single second of traffic within the given phase.

        This is the core micro-step of the simulation.  Each call:

        1. **Arrivals** — For every lane, a vehicle arrives with the current
           demand probability.  New vehicles start with a wait counter of 0.
        2. **Departures** — Only lanes that have a green light (phase 0 → NS
           lanes 0-1, phase 2 → EW lanes 2-3) can release vehicles.  Up to
           ``depart_capacity`` vehicles leave per lane per second.  Yellow and
           opposite-green phases allow no departures.
        3. **Wait increment** — Every remaining vehicle in every lane has its
           individual wait counter incremented by 1.
        4. **Stats accumulation** — Queue lengths, total waits, and NS-vs-EW
           imbalance are added to the episode-level running totals.

        Args:
            phase_idx: Index of the current phase (0-3).
            arrivals_interval: Mutable list[4] accumulating per-lane arrivals
                               across the whole cycle (updated in-place).
            departures_interval: Mutable list[4] accumulating per-lane
                                 departures across the whole cycle (updated
                                 in-place).

        Returns:
            Tuple of (total_queue, total_wait, imbalance, departed_this_second).
        """
        # --- 1. Arrivals: stochastic per-lane vehicle generation ---
        probs = self._get_arrival_probabilities()
        for lane_idx in range(4):
            if self.rng.random() < probs[lane_idx]:
                self.lane_waits[lane_idx].append(0)  # new vehicle, 0 wait so far
                arrivals_interval[lane_idx] += 1

        # --- 2. Departures: only green-phase lanes can release vehicles ---
        # Phase 0 = NS green (lanes 0,1); Phase 2 = EW green (lanes 2,3).
        # Phases 1 and 3 are yellow — no departures.
        green_lanes = []
        if phase_idx == 0:
            green_lanes = [0, 1]
        elif phase_idx == 2:
            green_lanes = [2, 3]

        departed_this_second = 0
        for lane_idx in green_lanes:
            # Remove up to depart_capacity vehicles from the front of the queue.
            departed = min(self.depart_capacity, len(self.lane_waits[lane_idx]))
            if departed > 0:
                self.lane_waits[lane_idx] = self.lane_waits[lane_idx][departed:]
            departures_interval[lane_idx] += departed
            departed_this_second += departed

        self.total_departed += departed_this_second

        # --- 3. Wait increment: every queued vehicle waits one more second ---
        for lane_idx in range(4):
            self.lane_waits[lane_idx] = [w + 1 for w in self.lane_waits[lane_idx]]

        self.total_sim_seconds += 1
        self._sync_stats()

        # --- 4. Accumulate per-second stats into episode totals ---
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
        """Reset the environment to the start of a new episode.

        All state is re-initialised:
        - ``lane_waits``: per-lane lists of individual vehicle wait counters
          (the authoritative traffic state).
        - ``queues`` / ``waits``: derived summary stats (recomputed via
          ``_sync_stats``).
        - Phase tracking set to the beginning of a fresh cycle.
        - Action history initialised to the neutral "keep" action (1).
        - Arrival and departure history buffers filled with zeros for
          ``history_windows`` past windows.
        - All episode-level counters (step count, sim seconds, departed
          vehicles, cumulative reward/queue/wait/imbalance) zeroed.

        Returns:
            (state, info): The initial observation and an episode summary dict.
        """
        super().reset(seed=seed)

        if seed is not None:
            self.rng.seed(seed)

        # Per-lane vehicle wait lists: lane_waits[i] is a list of ints, one per
        # queued vehicle, representing how many seconds that vehicle has waited.
        self.lane_waits = [[], [], [], []]
        self.queues = [0, 0, 0, 0]  # derived: number of vehicles per lane
        self.waits = [0, 0, 0, 0]   # derived: total wait per lane

        # Start at the beginning of phase 0 (NS green).
        self.current_phase = 0
        self.phase_elapsed = 0

        # Initialise action history to the neutral "keep" action.
        keep_action = 1
        self.last_action = keep_action
        self.prev_action = keep_action

        self.current_cycle_durations = self._build_cycle_durations(keep_action)

        # Fixed-size history buffers (deques) for the last K cycles' traffic.
        self.arrival_history = deque(
            [[0, 0, 0, 0] for _ in range(self.history_windows)],
            maxlen=self.history_windows,
        )
        self.departure_history = deque(
            [[0, 0, 0, 0] for _ in range(self.history_windows)],
            maxlen=self.history_windows,
        )

        # Episode-level counters.
        self.step_count = 0
        self.total_sim_seconds = 0
        self.total_departed = 0
        self.action_change_count = 0

        # Episode-level running totals for reward and metrics.
        self.cumulative_reward = 0.0
        self.cumulative_queue = 0.0
        self.cumulative_wait = 0.0
        self.cumulative_imbalance = 0.0

        state = self._get_state()
        info = self._get_episode_summary()
        return state, info

    def step(self, action):
        """Execute one full signal cycle and return the resulting transition.

        The step proceeds as follows:
        1. **Action validation & bookkeeping** — verify the action is legal,
           detect whether the agent switched from its previous action.
        2. **Build cycle durations** — translate the action into the 4-phase
           duration list via ``_build_cycle_durations``.
        3. **Simulate all 4 phases** — iterate through NS-green, NS-yellow,
           EW-green, EW-yellow, simulating each second-by-second with
           ``_simulate_one_second_in_phase``.  Accumulate per-cycle queue,
           wait, imbalance, and throughput totals.
        4. **Compute reward** — a weighted combination of mean wait (−),
           mean queue (−), imbalance (−), throughput (+), and action switch (−).
        5. **Update history** — push this cycle's arrival/departure counts
           into the sliding-window history deques.
        6. **Check termination** — the episode is truncated (not terminated)
           when ``step_count`` reaches ``max_decisions``.

        Returns:
            (state, reward, terminated, truncated, info)
        """
        assert self.action_space.contains(action), f"Invalid action: {action}"

        action = int(action)
        self.step_count += 1

        # Track whether the agent changed its action from the previous cycle.
        action_changed = int(action != self.last_action)
        if action_changed:
            self.action_change_count += 1

        # Build the phase durations for this cycle based on the chosen action.
        self.current_cycle_durations = self._build_cycle_durations(action)
        cycle_seconds = sum(self.current_cycle_durations)

        # Per-cycle accumulators for arrival/departure counts (per lane).
        arrivals_interval = [0, 0, 0, 0]
        departures_interval = [0, 0, 0, 0]

        # Per-cycle accumulators for reward computation.
        queue_sum = 0.0
        wait_sum = 0.0
        imbalance_sum = 0.0
        departed_interval = 0

        # Simulate each of the 4 phases, second by second.
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

        # Average the per-second totals over the cycle length for the reward.
        mean_queue = queue_sum / cycle_seconds
        mean_wait = wait_sum / cycle_seconds
        mean_imbalance = imbalance_sum / cycle_seconds
        throughput_rate = departed_interval / cycle_seconds

        # Reward: penalise wait, queue, and imbalance; reward throughput;
        # penalise unnecessary action switching.
        reward = (
            -self.reward_wait_weight * mean_wait
            -self.reward_queue_weight * mean_queue
            -self.reward_imbalance_weight * mean_imbalance
            +self.reward_throughput_weight * throughput_rate
            -self.reward_switch_weight * action_changed
        )

        self.cumulative_reward += reward

        # Push this cycle's per-lane counts into the sliding history windows.
        self.arrival_history.append(arrivals_interval)
        self.departure_history.append(departures_interval)

        # Shift action history: current becomes "last", old "last" becomes "prev".
        self.prev_action = self.last_action
        self.last_action = action

        # Start next decision from cycle boundary.
        self.current_phase = 0
        self.phase_elapsed = 0

        # The episode never terminates early; it is truncated after max_decisions.
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

        # Attach full episode summary on the final step for logging.
        if truncated or terminated:
            info["episode"] = self._get_episode_summary()

        return state, reward, terminated, truncated, info

    def render(self):
        """Print a single-line debug summary of the current environment state.

        Includes the decision step counter, total simulated seconds, current
        phase, cycle durations, per-lane queues and waits, total departed
        vehicles, and the number of action changes so far.
        """
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
