from snippets import *
import numpy as np


def valueIteration(n_epochs, max_iters):
    # Empty q value matrix
    Q = np.zeros((4, H, W))

    # Run
    for i in range(n_epochs):

        # Perform one epoch
        S, A, R = epoch(initial_state,
                        greedy_policy,
                        Q,
                        max_iters=max_iters)


        for i in reversed(range(len(R) - 1, 1, -1)):

            # Get values from i_th step
            s = S[i]
            a = A[i]
            r = R[i]

            # Get values from previous step
            s_p = S[i - 1]
            a_p = A[i - 1]

            if S[i + 1] in goal_states:
                Q[a, s[0], s[1]] = r
                Q[a_p, s_p[0], s_p[1]] = r + gamma * np.max(Q[:, s[0], s[1]])
            else:
                Q[a_p, s_p[0], s_p[1]] = r + gamma * np.max(Q[:, s[0], s[1]])

    show_epoch(initial_state, greedy_policy, Q, max_iters=100)
    plt.show()

def SARSA(n_epochs, max_iters, alpha, eps):

    # Empty q value matrix
    Q = np.zeros((4, H, W))

    # Epsilon greedy policy
    policy = epsilon_greedy_policy(eps)


    for i in range(n_epochs):

        # Initializa state
        s = initial_state

        # Pick action a
        a = chooseAction(s,policy,Q)

        # Go over each step of the episode (until termination)
        i = 0
        while s not in goal_states and i < max_iters:

            # apply action a, observe r and s_
            r, s_ = step(s, a)

            # Choose new action from s_
            a_ = chooseAction(s_,policy,Q)

            # Perform update
            Q[a, s[0], s[1]] += alpha*(r + gamma * Q[a_, s_[0], s_[1]] -
                                       Q[a, s[0], s[1]])

            # Reassign the current values
            s = s_
            a = a_

            i += 1

    show_epoch(initial_state, greedy_policy, Q, max_iters=100)
    plt.show()

def Q_learning(n_epochs, max_iters, alpha, eps):

    # Empty q value matrix
    Q = np.zeros((4, H, W))

    # Epsilon greedy policy
    policy = epsilon_greedy_policy(eps)


    for i in range(n_epochs):

        # Initializa state
        s = initial_state

        # Go over each step of the episode (until termination)
        i = 0
        while s not in goal_states and i < max_iters:

            # Choose new action from s_
            a = chooseAction(s, policy, Q)

            # Apply action a, observe r and s_
            r, s_ = step(s, a)

            # Perform update
            Q[a, s[0], s[1]] += alpha*(r + gamma * np.max(Q[:, s_[0], s_[1]]) -
                                       Q[a, s[0], s[1]])

            # Reassign the current values
            s = s_

            i += 1

    show_epoch(initial_state, greedy_policy, Q, max_iters=100)
    plt.show()


def chooseAction(s, policy, Q):
    # Pick action greedily
    action_probs = policy(s, Q)
    return np.random.choice(actions, p=action_probs)

def step(s, a):

    # Observe reward
    r = reward(s, a)

    # Observe state s_
    trans = transitions(s)[a]
    state_probs = [tran.prob for tran in trans]
    s_ = trans[np.random.choice(len(trans), p=state_probs)].state

    return r,s_

def main():

    # Simulation parameters
    n_epochs = 1000
    max_iters = 100
    sarsa_alpha = 0.1
    sarsa_eps = 0.1


    # Perform value iteration
    #valueIteration(n_epochs, max_iters)

    # Perform Sarsa algorithm
    #SARSA(n_epochs, max_iters, sarsa_alpha, sarsa_eps)

    # Perform Q-learning algorithm
    Q_learning(n_epochs, max_iters, sarsa_alpha, sarsa_eps)






if __name__ == "__main__":
    main()
