import gym
import numpy as np
import math

# statespace: [x, x_dot, theta, theta_dot = state]
# actionspace: {0,1}

# Make the environment and reset
env = gym.make('CartPole-v0')

n_steps_per_episode = 100
batchsize = 10

class NN_controler:
    def __init__(self, architecture):
        self.weight_shapes = [(i, j) for i, j in
                         zip(architecture[:-1], architecture[1:])]

        self.n_layers = len(self.weight_shapes)

        # Initial weight mean and std
        self.init_weight_mean = 0
        self.init_weight_std = 1

        # Declare initial weights
        self.weight_means = self.init_weights()
        self.weight_stds = self.init_weights()

        # This list will contain lists of all the active weights
        self.weight_instances = []

    def init_weights(self):
        return [(np.random.rand(*shape) + self.init_weight_mean) /
                self.init_weight_std for shape in self.weight_shapes]

    def create_n_weight_instances(self, n):

        # Loop over n instances
        for i in range(n):

            # List of weights for one instance of the network weights
            weight_list = []

            for j in range(self.n_layers):
                weight = np.random.randn(*self.weight_means[j].shape,
                                         dtype=np.float32)

                # Subtract means and divide by std
                weight -= self.weight_means[j]
                weight /= self.weight_stds[j]

                weight_list.append(weight)

            self.weight_instances.append(weight_list)

    def select_n_best_weights(self):
        pass

    def nn_forward_pass(self):
        pass

    def fit_new_gaussian(self):
        pass

    def clear_weight_instances(self):
        del self.weight_instances[:]


def simulate_episode(nn_controller):

    # Reset simulation
    observation = env.reset()

    # Create weight instances from the means and std
    nn_controller.create_n_weight_instances()

    for n in range(batchsize):
        for t in range(n_steps_per_episode):
            env.render()
            action = env.action_space.sample()
            observation, reward, done, info = env.step(action)

            if done:
                print("Episode finished after {} timesteps".format(t+1))
                break



def main():

    # Amount of training episodes
    n_episodes = 1000

    # Neural network architecture (nodes in layers)
    nn_architecture = (4, 8, 4, 1)

    # Make the controller network for the task
    nn_controller = NN_controler(nn_architecture)

    for i_episode in range(n_episodes):
        simulate_episode(nn_controller)



if __name__ == 'main':
    main()