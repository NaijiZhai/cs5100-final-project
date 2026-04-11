class StaticActionPolicy:
    def __init__(self, action=1):
        self.action = int(action)

    def reset(self):
        pass

    def act(self, obs):
        return self.action


class FixedTimePolicy(StaticActionPolicy):
    def __init__(self, action=1):
        # Default action 1 keeps symmetric base cycle.
        super().__init__(action=action)


class DemandAwareFixedTimePolicy(StaticActionPolicy):
    def __init__(self, arrival_prob, tolerance=0.02):
        # arrival_prob = (N, S, E, W)
        ns_demand = float(arrival_prob[0]) + float(arrival_prob[1])
        ew_demand = float(arrival_prob[2]) + float(arrival_prob[3])

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
    def __init__(self, action_space):
        self.action_space = action_space

    def reset(self):
        pass

    def act(self, obs):
        return self.action_space.sample()


class DQNPolicy:
    def __init__(self, agent):
        self.agent = agent

    def reset(self):
        self.agent.online_network.eval()

    def act(self, obs):
        return self.agent.select_action(obs)


class QueueBasedPolicy:
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
        diff = ns_queue - ew_queue

        if diff > self.deadband:
            return 0  # favor NS
        if diff < -self.deadband:
            return 2  # favor EW
        return 1  # keep base cycle
