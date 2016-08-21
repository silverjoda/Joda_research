import gym
import numpy as np
import math

env = gym.make('CartPole-v0')
env.reset()

env.theta_threshold_radians = math.pi/2

for t in range(1000):

    action = env.action_space.sample()


    observation, reward, done, info = env.step(0)
    if done:
        print "Episode finished after {} timesteps".format(t+1)
        break

    env.render()

    print observation

env.monitor.close()