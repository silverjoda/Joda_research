from copy import deepcopy
from itertools import *
import numpy as np

class Node:
    def __init__(self, action_sequence, newact, isleaf=False, value = (0,0)):
        self.action_sequence = deepcopy(action_sequence)
        self.action_sequence.append(newact)
        self.value = value
        self.isleaf = isleaf

class BimatrixNE:
    def __init__(self, value, p1_seq, p2_seq):
        self.value = value
        self.p1_seq = p1_seq
        self.p2_seq = p2_seq

    def __str__(self):
        return "Value: {} \n Player 1 sequences: {} \n Player 2 sequences: {}"\
            .format(self.value, self.p1_seq, self.p2_seq)


class NFG:
    def __init__(self, action_sequences=None, game_matrix=None):
        self.game_matrix = game_matrix
        self.action_sequences = action_sequences

        self.NE = None

        # Make NFG matrix from the sequences
        if self.game_matrix is None and self.action_sequences is not None:
            pass

    def getNE(self):
        if self.NE is not None:
            return self.NE
        else:
            # Calculate
            self.NE = self._calculateNE()
            return self.NE

    def _calculateNE(self):
        pass

def RSPoutcome(p1a, p2a, s, c, actions):

    assert actions == ('R','S','P','F')

    valueMat = np.array([(c, c), (c + s, c - s), (c - s, c + s), (0, 0)],
                        [(c - s, c + s), (c, c), (c + s, c - s), (0, 0)],
                        [(c + s, c - s), (c - s, c + s), (c, c), (0, 0)],
                        [(0, 0), (0, 0), (0, 0), (0, 0)])

    return valueMat[actions.index(p1a),actions.index(p2a)]


def makeGameTree(star_val, card_discard_val, actions, agreed_upon_sequence):

    # All possible combinations of actions of both players
    action_perms = [c for c in product(actions, repeat=2)]

    # Make root node of tree with null sequence
    root = Node([])
    current_sequence = []
    nodeList = []
    round = 0
    current_value = 0

    # Go over the whole sequence
    for i in range(agreed_upon_sequence):

        # In each sequence enumerate all possible action combinations
        for (ap1, ap2) in action_perms:

            # Cooperation
            if ap1 == ap2 and ap1 == agreed_upon_sequence[round]:
                currentNode = Node(current_sequence,
                                   (ap1, ap2),
                                   value=(current_value + 1, current_value + 1))
                nodeList.append(currentNode)
            else:
                # Defection
                (vp1, vp2) = RSPoutcome(ap1, ap2)

                nodeList.append(Node(current_sequence,
                                     (ap1, ap2),
                                     value=(
                                     current_value + 1 + vp1,
                                     current_value + 1 + vp2),
                                     isleaf=True))

        # Both played agreed actions
        current_sequence.append((agreed_upon_sequence[i],
                                 agreed_upon_sequence[i]))

        current_value += 1

    return nodeList


def gameDFS(star_val, card_discard_val, actions, agreed_upon_sequence):
    pass


def makeNFG(leafnodes):
    pass


def main():

    # Make complete game tree to find all leaves ===========

    # Value parameters
    star_val = 3
    card_discard_val = 1
    actions = ('R','S','P','F')
    agreed_upon_sequence = ('R','R','R','R','P','P','P','P','S','S','S','S')

    # Make game tree
    nodes = makeGameTree(star_val,
                         card_discard_val,
                         actions,
                         agreed_upon_sequence)

    leaves = [n for n in nodes if n.isleaf]

    # Make NFG out of all the outcomes
    NFG = makeNFG(leaves)

    # Calculate Nash EQ of the NFG which maximizes payoffs
    NE = NFG.getNE()

    # Print info
    if NE is not None:
        print "Found NE in transformed NFG by ILP: "
        print NE



if __name__ == "__main__":
    main()