import gym
import numpy as np
import math

# statespace: [x, x_dot, theta, theta_dot = state]
# actionspace: {0,1}

# Make the environment and reset
env = gym.make('CartPole-v0')



def draw_theta_samples():
    pass

def simulate_episode(n_steps):
    observation = env.reset()
    for t in range(n_steps):
        env.render()
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)

        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break

def init_weights(shapes, mean, std):
    return [(np.random.rand(*shape) + mean) / std for shape in shapes]

def main():

    # Amount of samples from the gaussian distribution
    n_samples_per_update = 10

    # Amount of training episodes
    n_episodes = 1000

    # Amount of steps of the simulation per episode
    n_steps_per_episode = 100

    # Neural network architecture (nodes in layers)
    nn_architecture = (4, 8, 4, 1)

    # Amount of weight matrices in the network
    n_layers = len(nn_architecture) - 1

    weight_shapes = [(i, j) for i, j in
                     zip(nn_architecture[:-1], nn_architecture[1:])]

    # Declare initial weights
    weight_means = init_weights
    weight_std = []

    for i_episode in range(n_episodes):
        simulate_episode(n_steps_per_episode)



if __name__ == 'main':
    main()