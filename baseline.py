class FixedTimePolicy:
    def __init__(self, switch_interval=10):
        self.switch_interval = switch_interval
        self.current_action = 0
        self.steps_in_phase = 0

    def reset(self):
        self.current_action = 0
        self.steps_in_phase = 0

    def act(self, obs):
        if self.steps_in_phase >= self.switch_interval:
            self.current_action = 1 - self.current_action
            self.steps_in_phase = 0

        self.steps_in_phase += 1
        return self.current_action


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
        return self.agent.online_network.act(obs)