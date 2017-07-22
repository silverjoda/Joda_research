import tflearn as tfl
import gym
from policies import *
from algorithms import *

def printEnvInfo(env):
    init_obs = env.reset()
    print 'Initial obs: {}'.format(init_obs)
    print 'Action space: {}'.format(env.action_space)
    print 'Obs space: {}'.format(env.observation_space)
    print 'Bounds: {},{}'.format(env.action_space.low,
                                 env.action_space.high)

def main():

    # 1) Train reactive policy on chosen environment

    # Used environments list
    envlist = ["Ant-v1", "Hopper-v1", "HalfCheetah-v1", "Walker2d-v1"]

    # Make simulation
    env = gym.make(envlist[3])

    # Info
    printEnvInfo(env)

    # Make reactive policy
    rPol = ReactivePolicy(env)

    # Params
    n_episodes = 1000
    max_iters = 160

    # Train policy
    rPol.train(n_episodes, max_iters)

    # Evaluate and visualize policy
    rPol.visualize()

    # 2) Train recurrent policy on chosen environment

    # Make simulation


    # Make policy


    # Train policy


    # Evaluate and visualize policy

    # 3) Perform temporal analysis on recurrent policy


    print 'Done. '



if __name__ == '__main__':
    main()