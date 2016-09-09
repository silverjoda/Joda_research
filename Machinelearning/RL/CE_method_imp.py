import gym
import numpy as np
import math
import os.path

# statespace: [x, x_dot, theta, theta_dot = state]
# actionspace: {0,1}

class NN_controler:
    def __init__(self, architecture):

        # Shapes of the weights of the network
        self.weight_shapes = [(j, i) for i, j in
                         zip(architecture[:-1], architecture[1:])]

        # Amount of layers that the network contains
        self.n_layers = len(self.weight_shapes)

        # Initial weight mean and std
        self.init_weight_mean = 0
        self.init_weight_std = 1

        # Declare initial weights
        self.weight_means = self.init_weights()
        self.weight_stds = self.init_weights()

    def init_weights(self):
        """
        Initialize the weights
        Returns initialized weights
        -------

        """
        return [(np.random.randn(*shape) + self.init_weight_mean) /
                self.init_weight_std for shape in self.weight_shapes]


    def create_n_weight_instances(self, n):
        """
        Draw n weight instances from the weight means and stds
        Parameters
        ----------
        n: int, amount of instances to be generated

        Returns: list of lists of np array matrices which are the instances
        of the weights
        -------

        """

        # Declare instances list
        weight_instances = []

        # Loop over n instances
        for i in range(n):

            # List of weights for one instance of the network weights
            weight_list = []

            for j in range(self.n_layers):
                weight = np.random.randn(*self.weight_means[j].shape)

                # Subtract means and divide by std
                weight -= self.weight_means[j]
                weight /= self.weight_stds[j]

                weight_list.append(weight)

            weight_instances.append(weight_list)

        if n == 1:
            return weight_instances[0]
        else:
            return weight_instances


    def nn_forward_pass(self, X, weights):
        """
        Performs forward pass on the network given by weights and input X
        Parameters
        ----------
        X: np array with shape (input_dim)
        weights: list of np array matrices

        Returns: np array of the final layer vector
        -------

        """

        # Current layer
        layer = X

        # Forward pass for all layers (matmul + relu)
        for w in weights:
            layer = np.maximum(0, np.matmul(w, layer))

        return np.sign(layer)

    def load_weights(self):
        pass

    def save_weights(self):
        pass




def simulate(env, nn_controller):

    # Reset simulation
    observation = env.reset()

    # Create weight instances from the means and std
    weights = nn_controller.create_n_weight_instances(1)

    while True:
        env.render()
        action = nn_controller.nn_forward_pass(np.array(observation), weights)
        observation, _, done, info = env.step(action[0].astype(int))


def train_batch_episode(env, nn_controller, batchsize, n_steps_per_episode):

    # Create weight instances from the means and std
    weight_list = nn_controller.create_n_weight_instances(batchsize)

    # List of the episode rewards for each of the generated weights
    total_rewards = []

    for n in range(batchsize):

        # Reset simulation for each weight instance
        observation = env.reset()

        # Total reward for whole episode for this weight instance
        total_reward = 0

        for t in range(n_steps_per_episode):

            # Choose action according to the given observation
            action = nn_controller.nn_forward_pass(np.array(observation),
                                                   weight_list[n])

            # Perform step in the simulator
            observation, reward, done, info = env.step(action)

            # Accumulate episode reward
            total_reward += reward

            if done:
                break

        # Append the total reward for this instance
        total_rewards.append(total_reward)

    # === Test the new network and render it === #
    done = False
    total_reward = 0

    for t in range(n_steps_per_episode):
        env.render()
        weight_list = nn_controller.create_n_weight_instances(1)
        action = nn_controller.nn_forward_pass(np.array(observation),
                                               weight_list)
        observation, reward, done, info = env.step(action)
        total_reward += reward

    return total_reward


def main():

    # Make the environment and reset
    env = gym.make('CartPole-v0')

    # Amount of training episodes
    n_episodes = 1000

    # Neural network architecture (nodes in layers)
    nn_architecture = (4, 8, 2, 1)

    # Make the controller network for the task
    nn_controller = NN_controler(nn_architecture)

    # Simulate using the available weights
    simulate(env, nn_controller)



if __name__ == "__main__":
    main()