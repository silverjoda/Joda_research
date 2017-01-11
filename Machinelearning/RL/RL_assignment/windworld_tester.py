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

def SARSA(n_epochs, max_iters, alpha):

    # Empty q value matrix
    Q = np.zeros((4, H, W))

    # Epsilon greedy policy
    policy = epsilon_greedy_policy(0.1)


    for i in range(n_epochs):
        s = initial_state
        S = [s]
        A = []
        R = []
        i = 0
        while s not in goal_states and i < max_iters:
            action_probs = policy(s, Q)
            a = np.random.choice(actions, p=action_probs)
            A.append(a)
            R.append(reward(s, a))
            trans = transitions(s)[a]
            state_probs = [tran.prob for tran in trans]
            sp = trans[np.random.choice(len(trans), p=state_probs)].state
            S.append(sp)
            s = sp
            i += 1




    pass

def main():

    # Simulation parameters
    n_epochs = 1000
    max_iters = 100
    sarsa_alpha = 0.1


    # Perform value iteration
    #valueIteration(n_epochs, max_iters)

    # Perform Sarsa algorithm
    SARSA(n_epochs, max_iters, sarsa_alpha)







if __name__ == "__main__":
    main()
