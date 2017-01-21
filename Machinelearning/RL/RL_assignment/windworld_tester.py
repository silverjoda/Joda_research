from snippets import *
import numpy as np

def bmatrix(a):
    """Returns a LaTeX bmatrix

    :a: numpy array
    :returns: LaTeX bmatrix as a string
    """
    if len(a.shape) > 2:
        raise ValueError('bmatrix can at most display two dimensions')
    lines = str(a).replace('[', '').replace(']', '').splitlines()
    rv = [r'\begin{bmatrix}']
    rv += ['  ' + ' & '.join(l.split()) + r'\\' for l in lines]
    rv +=  [r'\end{bmatrix}']
    return '\n'.join(rv)

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



        if sum(R) == -15:
            print "Found optimal Q-function at epoch: {}".format(i)
            break


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
    #plt.title("Gridworld, value iteration. R = -15")
    #plt.show()

    #for j in range(len(Q)):
    #    print bmatrix(Q[j])

    return Q


def SARSA(n_epochs, max_iters, alpha, eps, optQ):


    # Empty q value matrix
    Q = np.zeros((4, H, W))

    # Epsilon greedy policy
    policy = epsilon_greedy_policy(eps)

    # Q error matrix
    q_err_mat = []

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

        q_err_mat.append(np.sum(np.square(Q - optQ)))

    #show_epoch(initial_state, greedy_policy, Q, max_iters=100)


    return Q, q_err_mat

def Q_learning(n_epochs, max_iters, alpha, eps, optQ):

    # Empty q value matrix
    Q = np.zeros((4, H, W))

    # Epsilon greedy policy
    policy = epsilon_greedy_policy(eps)

    # Q error matrix
    q_err_mat = []

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

        q_err_mat.append(np.sum(np.square(Q - optQ)))

    #show_epoch(initial_state, greedy_policy, Q, max_iters=100)
    #plt.show()

    return Q, q_err_mat

def policyIteration(n_epochs, max_iters, theta):

    # Random value matrix
    V = np.random.randint(-10,0, size=(H, W))

    # Random policy initialization
    P = np.random.randint(0,4, size=(H, W))

    # Epoch counter
    e = 0

    # Run
    while e < n_epochs:

        e += 1

        print "Policy iteration, iter: {}".format(e)

        # ==================
        # Policy evaluation
        # ==================

        delta = 0

        for i in range(3):
            for s in istates():
                v = V[s[0],s[1]]

                # Get action from our current policy
                a = P[s[0],s[1]]

                # Observe reward
                r, s_ = step(s,a)

                V[s[0], s[1]] = r + V[s_[0], s_[1]]

                #delta = np.max([delta, np.abs(v - V[s[0], s[1]]) ])


        # ==================
        # Policy improvement
        # ==================

        policy_stable = True

        for s in istates():
            b = P[s[0],s[1]]

            # Get destinations and rewards for all actions
            dest = [step(s, i) for i in range(4)]

            # Take argmax to improve policy
            P[s[0], s[1]] = np.argmax([r + V[s_[0], s_[1]] for (r, s_) in dest])

            if b != P[s[0], s[1]]:
                policy_stable = False

        if policy_stable:
            break

    # Empty q value matrix
    Q = np.zeros((4, H, W))

    # Make Q matrix
    for s in istates():
        Q[P[s[0],s[1]],s[0],s[1]] = V[s[0],s[1]]

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
    sarsa_alpha = [0.01, 0.1, 0.3]
    sarsa_eps = [0.01, 0.1, 0.3]
    ql_alpha = [0.05, 0.2, 0.7]
    ql_eps = [0.05, 0.1, 0.5]
    policy_iter_theta = 3


    # Perform value iteration
    optQ = valueIteration(n_epochs, max_iters)
    q_err_mats = []

    # Perform Sarsa algorithm
    # for i,e in enumerate(sarsa_eps):
    #     plt.figure()
    #
    #     for a in sarsa_alpha:
    #         Q, q_err_mat = SARSA(n_epochs, max_iters, a, e, optQ)
    #         q_err_mats.append(q_err_mat)
    #
    #     title = "SARSA, eps: {}".format(e)
    #
    #     for q_m in q_err_mats:
    #         plt.plot(range(n_epochs), q_m)
    #
    #     plt.title(title)
    #     plt.xlabel("Epochs")
    #     plt.ylabel("Q MSE")
    #     plt.legend(["a = 0.01","a = 0.1","a = 0.3"])
    #
    #     plt.savefig("{}.png".format(title.replace(" ", "")), dpi=100)
    #
    #     q_err_mats = []
    #

    q_err_mats = []

    # Perform Q-learning algorithm
    for i, e in enumerate(ql_eps):
        plt.figure()

        for a in ql_alpha:
            Q, q_err_mat = Q_learning(n_epochs, max_iters, a, e, optQ)
            q_err_mats.append(q_err_mat)

        title = "Q-learning, eps: {}".format(e)

        for q_m in q_err_mats:
            plt.plot(range(n_epochs), q_m)

        plt.title(title)
        plt.xlabel("Epochs")
        plt.ylabel("Q MSE")
        plt.legend(["a = 0.01", "a = 0.1", "a = 0.3"])

        plt.savefig("{}.png".format(title.replace(" ", "")), dpi=100)

        q_err_mats = []

    for q in q_err_mats:
        print np.min(q)


    # Perform Policy iteration
    #policyIteration(n_epochs, max_iters, policy_iter_theta)


if __name__ == "__main__":
    main()
