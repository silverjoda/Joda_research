import numpy as np
import gym
import math
from gym.spaces import Discrete, Box

env = gym.make("Pong-v0")

for i_episode in range(20):
    observation = env.reset()
    for t in range(100):
        env.render()
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        print reward

        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break
