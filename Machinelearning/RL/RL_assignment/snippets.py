import numpy as np
import matplotlib.pyplot as plt
from itertools import chain
from collections import namedtuple

# Windy gridworld MDP definition
# -------------------------------

# helper data structures:
# a state is given by row and column positions designated (y, x)
State = namedtuple('State', ['y', 'x'])

# encapsulates a transition to state and its probability
Transition = namedtuple('Transition', ['state', 'prob'])  # transition to state with probability prob

# height and width of the gridworld
H, W = 7, 10

# available actions
ACTION_LEFT = 0
ACTION_RIGHT = 1
ACTION_UP = 2
ACTION_DOWN = 3
actions = [ACTION_LEFT, ACTION_RIGHT, ACTION_UP, ACTION_DOWN]

# wind strength per column of the gridworld
WIND = [0, 0, 0, 1, 1, 1, 2, 2, 1, 0]
# WIND = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

# initial state for debugging
initial_state = State(3, 0)

# set of goal states
goal_states = {State(3, 7)}

# undiscounted rewards
gamma = 1.0


# iterator over all possible states
def istates():
    for y in xrange(H):
        for x in xrange(W):
            yield State(y, x)


# returns a list of Transitions from the state s for each action, only non zero probabilities are given
# serves the lists for all actions at once
def transitions(s):
    if s in goal_states:
        return [[Transition(state=s, prob=1.0)] for a in actions]
    y, x = s
    return [
        # transitions for ACTION_LEFT
        [Transition(state=State(max(y - WIND[x], 0), max(x - 1, 0)), prob=1.0)],
        # transitions for actions = ACTION_RIGHT
        [Transition(state=State(max(y - WIND[x], 0), min(x + 1, W - 1)), prob=1.0)],
        # transitions for ACTION_UP
        [Transition(state=State(max(y - 1 - WIND[x], 0), x), prob=1.0)],
        # transitions for ACTION_DOWN
        [Transition(state=State(max(min(y + 1 - WIND[x], H - 1), 0), x), prob=1.0)]
    ]


# reward function
def reward(s, a):
    if s in goal_states:
        return 0.0
    return -1.0


# gives a list of states reachable from state s by any available action
def neighbours(s):
    return set([r.state for r in chain(*transitions(s).values())])


# random policy: each action has the same probability
def random_policy(s, Q):
    return [1.0 / len(actions) for a in actions]


# greedy policy gives the best action (based on action value function Q) a probability of 1, others are given 0
def greedy_policy(s, Q):
    probs = np.zeros_like(actions, dtype=float)
    probs[np.argmax(Q[:, s.y, s.x])] = 1.0
    return probs


# epsilon-greedy policy gives random action with probability eps or greedy one otherwise
def epsilon_greedy_policy(eps=0.0):
    def epsilon_greedy_policy_helper(s, Q):
        if np.random.uniform() < eps:
            return random_policy(s, Q)
        else:
            return greedy_policy(s, Q)

    return epsilon_greedy_policy_helper


# gives a state-value function v(s) based on action-value function q(s, a) and policy
# for debugging and visualization
def q_to_v(Q, policy):
    Vnew = np.zeros((H, W))
    for s in istates():
        activity_probs = policy(s, Q)
        for a in actions:
            Vnew[s.y, s.x] += activity_probs[a] * Q[a, s.y, s.x]
    return Vnew


# evaluates a single epoch starting at start_state, using a policy which can use
# an action-value function Q as a parameter
# returns a triple: states visited, actions taken, rewards taken
def epoch(start_state, policy, Q, max_iters=100):
    s = start_state
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
    return S, A, R


# visualizes a single epoch starting at start_state, using a policy which can use
# an action-value function Q as a parameter
# shows actions taken as arrows, the color of cells is based on a state-value function
# which is computed from the action-value function Q
def show_epoch(start_state, policy, Q, max_iters=100):
    offsets = {0: (0.4, 0), 1: (-0.4, 0), 2: (0, 0.4), 3: (0, -0.4)}
    dirs = {0: (-0.8, 0), 1: (0.8, 0), 2: (0, -0.8), 3: (0, 0.8)}
    S, A, R = epoch(start_state, greedy_policy, Q, max_iters=max_iters)
    ax = plt.gca()
    ax.imshow(q_to_v(Q, policy), interpolation='nearest', origin='upper')
    #     ax.plot([s.x for s in S], [s.y for s in S], c='k')
    ax.text(start_state[1], start_state[0], 'S', ha='center', va='center', fontsize=20)
    for s in goal_states:
        ax.text(s[1], s[0], 'G', ha='center', va='center', fontsize=20)
    for s, a in zip(S[:-1], A):
        ax.add_patch(plt.Arrow(s.x + offsets[a][0], s.y + offsets[a][1], dirs[a][0], dirs[a][1]))

    plt.title('reward: {}'.format(np.sum(R)))

