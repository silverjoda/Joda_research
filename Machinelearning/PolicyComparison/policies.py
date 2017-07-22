import tflearn as tfl

class ReactivePolicy:
    def __init__(self, env):
        self.env = env
        self.statedim = env.observation_space.shape
        self.action_dim = env.action_space.shape

        # Bounds
        self.action_low = env.action_space.low
        self.action_high = env.action_space.high





    def visualize(self):
        pass

class RecurrentPolicy:
    pass