
import gym
import numpy as np
import time

def printEnvInfo(env):
    init_obs = env.reset()
    print 'Initial obs: {}'.format(init_obs)
    print 'Action space: {}'.format(env.action_space)
    print 'Obs space: {}'.format(env.observation_space)
    print 'Bounds: {},{}'.format(env.action_space.low,
                                 env.action_space.high)
    
def randomPolicy(env, param, state):

    # Action dimension
    actiondim = env.action_space.shape

    # Bounds
    low = env.action_space.low
    high = env.action_space.high

    # Return random action
    return np.random.rand(*actiondim) * (high - low) + low



def main():
    pass


if __name__ == '__main__':
    main()