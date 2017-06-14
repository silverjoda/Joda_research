from copy import deepcopy
from itertools import *
import numpy as np
from NEGameCalc import *

class Node:
    def __init__(self, game, depth):

        # Instance of game
        self.game = game

        # Current node depth
        self.depth = depth

        # Nash equillibrium at this node
        self.NE = None

        # Nodes game matrices
        self.a = None
        self.b = None

    def __str__(self):
        return 'Node at depth ' + str(self.depth) + '\n' + \
               'a: ' + '\n' + str(self.a) + '\n' + 'b: ' + '\n' + str(self.b) + '\n' \
               + 'u: {},'.format(self.NE[0]) + ' v: {}'.format(self.NE[1]) + \
               '\n' + 'Sa: ' + str(self.NE[2]) + '\n' + 'Sb: ' + str(self.NE[3]) \
               + '\n' + 'Available acts: {}'.format(self.game._get_available_actions(self.depth)) + '\n'

    def getNE(self):

        if self.NE is not None: return self.NE

        # Available actions in this node
        available_acts = self.game._get_available_actions(self.depth)

        # Calculate game matrices
        if self.a is None:
            self.a, self.b = self.game.getGameMatrix(available_acts)

        # Pre-last node
        if self.depth < 11:

            nextNode = self.game.nodeList[self.depth + 1]

            # Get nash equillibrium of descendant
            u,v,_,_ = nextNode.getNE()

            # Actions to indeces
            action = actTonum(available_acts, self.game.agreed_upon_sequence[self.depth])

            self.a[action, action] += u
            self.b[action, action] += v

        self.NE = calcILPNE(self.a, self.b)

        return self.NE



class Game:
    def __init__(self, actions, agreed_upon_sequence, star_val, card_discard_val):
        self.actions = actions
        self.agreed_upon_sequence = agreed_upon_sequence
        self.star_val = star_val
        self.card_discard_val = card_discard_val

        # Make the game tree
        self._maketree()

    def _maketree(self):

        # Keep list of all nodes of tree
        self.nodeList = []

        for i in range(len(self.agreed_upon_sequence)):
            # Make new node
            new_node = Node(self, i)

            # Add node to list
            self.nodeList.append(new_node)

    def _get_available_actions(self, depth):
        return list(set(self.agreed_upon_sequence[depth:])) + ['F']

    def coopAction(self, act):
        if act == []: return True
        return act[0] == act[1]

    def getNE(self):

        NEseq = []

        for n in self.nodeList:

            # Append only the actions
            NEseq.append(n.getNE())

        # Return whole NE sequence and value at root node
        return NEseq

    def getGameMatrix(self, available_acts):

        a = np.zeros((len(available_acts), len(available_acts)))
        b = np.zeros_like(a)

        for i in range(len(available_acts)):
            for j in range(len(available_acts)):
                u,v = RSPoutcome(available_acts[i], available_acts[j],
                                self.star_val, self.card_discard_val)

                a[i, j] = u
                b[i, j] = v

        return a, b

def actTonum(available_acts, act):
    return available_acts.index(act)

def RSPoutcome(p1a, p2a, s, c):

    actions = ('R', 'S', 'P', 'F')

    valueMat = np.array([[(c, c), (c + s, c - s), (c - s, c + s), (0, 0)],
                        [(c - s, c + s), (c, c), (c + s, c - s), (0, 0)],
                        [(c + s, c - s), (c - s, c + s), (c, c), (0, 0)],
                        [(0, 0), (0, 0), (0, 0), (0, 0)]])

    return valueMat[actions.index(p1a),actions.index(p2a)]


def main():

    # Make complete game tree to find all leaves ===========

    # Value parameters
    star_val = 3
    card_discard_val = 1
    actions = ('R','S','P','F')
    agreed_upon_sequence = ('R','R','R','R','P','P','P','P','S','S','S','S')

    # Make Game
    game = Game(actions, agreed_upon_sequence, star_val, card_discard_val)

    # Find NE of whole tree by backwards induction
    NE = game.getNE()

    for node in game.nodeList:
        print node



if __name__ == "__main__":
    main()