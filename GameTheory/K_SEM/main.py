from copy import deepcopy
from itertools import *
import numpy as np
from NEGameCalc import *

class Node:
    def __init__(self, game, history, prev_action, depth, cur_acts, predecessor,
                 isleaf=False, value = None):

        # Instance of game
        self.game = game

        # Node predecessor
        self.predecessor = predecessor

        # Descendants of the node
        self.descendants = []

        # Copy and append history
        if prev_action is not None or prev_action != []:
            self.history = deepcopy(history + [prev_action])

        # Last action
        self.prev_action = prev_action

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

    def calcNE(self):

        if self.NE is not None:
            return self.NE

        # Calculate game matrixes
        a, b = self.game.getGameMatrix()

        if self.isleaf:
            self.NE = calcNE(a,b)
            return self.NE
        else:
            for n in self.descendants:

                n.NE = n.calcNE()

                a[n.prev_action[0], n.prev_action[1]] += n.NE[0]
                b[n.prev_action[0], n.prev_action[1]] += n.NE[1]


        return self.NE

    def _calculateNE(self):

        # Calculate game matrixes
        a,b = self.game.getGameMatrix()

        # Calculate NE using ILP
        return calcNE(a,b)


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

        # Current action history
        history = []

        # Currently available actions
        cur_acts = self.actions

        # Current depth of the tree.
        cur_depth = 0

        # Current node
        cur_node = Node(self, history, [], 0, cur_acts, None)
        self.nodeList.append(cur_node)
        self.root = cur_node

        # At most D games (we only branch once everytime).
        while True:

            # All possible combinations of actions of both players
            action_perms = [c for c in product(cur_acts, repeat=2)]

            for a in action_perms:

                # Skip the part where we go to the next depth
                if a[0] == a[1] == self.agreed_upon_sequence[cur_depth]:
                    continue

                # Make new node
                new_node = Node(self, history, a, cur_depth + 1,
                                   self._get_available_actions(cur_depth + 1),
                                   cur_node,
                                   isleaf = True)

                # Add node to list
                self.nodeList.append(new_node)

                # Add node,action pair to descendants
                cur_node.descendants.append(new_node)

            # Make current node the one which continues the game
            c_act = (self.agreed_upon_sequence[cur_depth], self.agreed_upon_sequence[cur_depth])

            # Make new node
            new_node = Node(self, history, c_act, cur_depth + 1,
                            self._get_available_actions(cur_depth + 1),
                            cur_node,
                            isleaf=False)

            # Add node to list
            self.nodeList.append(new_node)

            # Add node,action pair to descendants
            cur_node.descendants.append(new_node)

            cur_node = new_node

            # Append latest action to history
            history.append(c_act)

            # Terminate if we played the whole sequence
            if cur_depth == len(self.agreed_upon_sequence) - 1:
               break

            # Increment depth counter
            cur_depth += 1

    def _get_available_actions(self, depth):
        return list(set(self.agreed_upon_sequence[depth:]))

    def getNE(self):

        NEseq = []

        for n in self.nodeList:
            # Append only the actions
            NEseq.append((n.calcNE()[2],n.calcNE()[3]))

            # If deviation at this point then consider game finished
            if n.calcNE != n.calcNE():
                break

        # Return whole NE sequence and value at root node
        return NEseq, self.root.calcNE()[1]

    def getGameMatrix(self):

        a = np.zeros((len(self.actions), len(self.actions)))
        b = np.zeros_like(a)

        for i in range(len(self.actions)):
            for j in range(len(self.actions)):
                u,v = RSPoutcome(('R','S','P','F')[i],
                                    ('R','S','P','F')[j],
                                    self.star_val,
                                    self.card_discard_val)

                a[i, j] = u
                b[i, j] = v

        return a, b


def RSPoutcome(p1a, p2a, s, c):

    actions = ('R','S','P','F')

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
    actions = ('R','S','P')
    agreed_upon_sequence = ('R','R','R','R','P','P','P','P','S','S','S','S')

    # Make Game
    game = Game(actions, agreed_upon_sequence, star_val, card_discard_val)

    # Find NE of whole tree by backwards induction
    NE = game.getNE()

    exit()

    # Print info
    if NE is not None:
        print "Found NE in transformed NFG by ILP: "
        print NE



if __name__ == "__main__":
    main()