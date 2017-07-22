import tflearn as tfl
from policies import *

class ReactiveTrainer:
    def __init__(self, env, policy):
        self.env = env
        self.policy = policy

    def train(self, n_episodes, max_iters):
        pass





