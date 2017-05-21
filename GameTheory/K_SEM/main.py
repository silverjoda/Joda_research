


class SimultaneousAction:
    def __init__(self):
        self.P1_action = None
        self.P2_action = None

class Node:
    def __init__(self, action_sequence):
        self.action_sequence = action_sequence
        self.value = -1
        self.isleaf = False

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

def makeGameTree(star_val, card_discard_val):
    pass


def makeNFG(leafnodes):
    pass

def main():

    # Make complete game tree to find all leaves ===========

    # Value parameters
    star_val = 3
    card_discard_val = 1

    # Make game tree
    nodes = makeGameTree(star_val, card_discard_val)
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