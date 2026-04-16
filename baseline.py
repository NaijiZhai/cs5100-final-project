"""
Baseline policies for traffic signal control comparison.

Each policy exposes the same `.act(obs)` / `.reset()` interface so it can be
dropped into the evaluation harness interchangeably with the trained DQN agent.
These simple strategies serve as performance baselines to quantify how much
the learned policy improves over naïve or heuristic approaches.
"""


class StaticActionPolicy:
    """Policy that always returns the same fixed action, regardless of state.

    Useful as the simplest possible baseline: the signal plan never adapts
    to traffic conditions.
    """

    def __init__(self, action=1):
        self.action = int(action)

    def reset(self):
        pass

    def act(self, obs):
        return self.action


class FixedTimePolicy(StaticActionPolicy):
    """Fixed-time signal policy that keeps the default symmetric cycle.

    By always selecting action 1 ("keep"), the intersection runs its
    pre-configured equal-split timing without any adjustment.
    """

    def __init__(self, action=1):
        # Default action 1 keeps symmetric base cycle.
        super().__init__(action=action)


class DemandAwareFixedTimePolicy(StaticActionPolicy):
    """Static policy that picks a single action based on aggregate demand.

    At construction time, it compares total north-south arrival probability
    against east-west arrival probability.  If one direction dominates beyond
    a tolerance band, the policy permanently favours that direction; otherwise
    it keeps the balanced default.  The decision is made once and never
    revisited, so this is still a fixed-time policy — just demand-informed.
    """

    def __init__(self, arrival_prob, tolerance=0.02):
        # arrival_prob = (N, S, E, W)
        ns_demand = float(arrival_prob[0]) + float(arrival_prob[1])
        ew_demand = float(arrival_prob[2]) + float(arrival_prob[3])

        # Choose the action that favours the heavier direction, or keep
        # the balanced default if the difference is within the tolerance band.
        if ns_demand - ew_demand > tolerance:
            action = 0  # favor NS
        elif ew_demand - ns_demand > tolerance:
            action = 2  # favor EW
        else:
            action = 1  # keep

        super().__init__(action=action)
        self.ns_demand = ns_demand
        self.ew_demand = ew_demand
        self.tolerance = tolerance


class RandomPolicy:
    """Policy that samples a uniformly random action each step.

    Provides a stochastic lower bound: any reasonable learned policy should
    comfortably outperform random signal switching.
    """

    def __init__(self, action_space):
        self.action_space = action_space

    def reset(self):
        pass

    def act(self, obs):
        return self.action_space.sample()


class DQNPolicy:
    """Adapter that wraps a trained DQN Agent into the baseline policy interface.

    This lets the evaluation harness treat the learned agent identically to
    the hand-crafted baselines — same `.act(obs)` / `.reset()` contract.
    """

    def __init__(self, agent):
        self.agent = agent

    def reset(self):
        # Switch the online network to eval mode for deterministic inference.
        self.agent.online_network.eval()

    def act(self, obs):
        return self.agent.select_action(obs)


class QueueBasedPolicy:
    """Reactive policy that switches the signal based on current queue lengths.

    Compares the combined north-south queue to the east-west queue.  If one
    side exceeds the other by more than the deadband threshold, the signal
    favours that side; otherwise it keeps the current phase.  The deadband
    prevents rapid oscillation when queues are nearly equal.
    """

    def __init__(self, deadband=0.05):
        # deadband is applied to NS-vs-EW queue difference in observation scale
        self.deadband = deadband

    def reset(self):
        pass

    def act(self, obs):
        # First 4 dims are [qN, qS, qE, qW], normalized or raw.
        qn, qs, qe, qw = obs[:4]

        ns_queue = qn + qs
        ew_queue = qe + qw
        # Positive diff means NS is more congested; negative means EW is.
        diff = ns_queue - ew_queue

        if diff > self.deadband:
            return 0  # favor NS
        if diff < -self.deadband:
            return 2  # favor EW
        return 1  # keep base cycle
