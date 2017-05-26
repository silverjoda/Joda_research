from copy import deepcopy
from itertools import *
import numpy as np

class Node:
    def __init__(self, history, newact, depth, cur_acts, predecessor,
                 isleaf=False, value = (0,0)):

        # Copy and append history
        self.history = deepcopy(history)

        if newact is not None:
            self.history.append(newact)

        # Get currently available actions
        self.available_acts = cur_acts

        # Current node depth
        self.depth = depth

        # Value of the node (Nash Equillibrium)
        self.value = value

        # True if node finishes the game
        self.isleaf = isleaf

        # Nash equillibrium at this node
        self.NE = None

    def getNE(self):
        pass

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


class Game:
    def __init__(self,actions,agreed_upon_sequence,star_val,card_discard_val):
        self.actions = actions
        self.agreed_upon_sequence = agreed_upon_sequence
        self.star_val = star_val
        self.card_discard_val = card_discard_val

        # Make the game tree
        self._maketree()

    def _maketree(self):

        # Keep list of all nodes of tree
        self.nodeList = []

        # Current action history
        history = []

        # Currently available actions
        cur_acts = self.actions

        # Current depth of the tree.
        cur_depth = 0

        # Current node
        cur_node = Node(history, None, 0, cur_acts, None)

        # At most D games (we only branch once everytime).
        while True:

            # All possible combinations of actions of both players
            action_perms = [c for c in product(cur_acts, repeat=2)]

            for a in action_perms:

                # Skip the part where we go to the next depth
                if a == self.agreed_upon_sequence[cur_depth]:
                    continue

                # Make new node
                new_node = Node(history, a, cur_depth + 1,
                                   self._get_available_actions(cur_depth + 1),
                                   cur_node,
                                   isleaf = True)

                # Add node to list
                self.nodeList.append(new_node)

            # Make current node the one which continues the game
            c_act = (self.agreed_upon_sequence[cur_depth], self.agreed_upon_sequence[cur_depth])

            new_node = Node(history, c_act, cur_depth + 1,
                            self._get_available_actions(cur_depth + 1),
                            cur_node,
                            isleaf=False)

            if cur_depth == len(self.agreed_upon_sequence):
                break

            cur_depth += 1




    def _get_available_actions(self, depth):
        return list(set(self.agreed_upon_sequence[depth:]))


def RSPoutcome(p1a, p2a, s, c):

    actions = ('R','S','P','F')

    valueMat = np.array([(c, c), (c + s, c - s), (c - s, c + s), (0, 0)],
                        [(c - s, c + s), (c, c), (c + s, c - s), (0, 0)],
                        [(c + s, c - s), (c - s, c + s), (c, c), (0, 0)],
                        [(0, 0), (0, 0), (0, 0), (0, 0)])

    return valueMat[actions.index(p1a),actions.index(p2a)]


def main():

    # Make complete game tree to find all leaves ===========

    # Value parameters
    star_val = 3
    card_discard_val = 1
    actions = ('R','S','P')
    agreed_upon_sequence = ('R','R','R','R','P','P','P','P','S','S','S','S')

    # Make Game
    game = Game(star_val, card_discard_val, actions, agreed_upon_sequence)

    exit()

    # Find NE of whole tree by backwards induction
    NE = NFG.BI()

    # Print info
    if NE is not None:
        print "Found NE in transformed NFG by ILP: "
        print NE



if __name__ == "__main__":
    main()