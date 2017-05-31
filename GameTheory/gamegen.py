
import numpy as np
import gurobipy as g

def gengame():

    # Gurobi model
    m = g.Model()

    a = {}
    b = {}
    cce = {}
    ne = {}

    for i in range(2):
        for j in range(2):
            a[i, j] = m.addVar(lb=-10, ub=10, vtype=g.GRB.CONTINUOUS)
            b[i, j] = m.addVar(lb=-10, ub=10, vtype=g.GRB.CONTINUOUS)
            cce[i, j] = m.addVar(lb=-10, ub=10, vtype=g.GRB.CONTINUOUS)

    ne['P1', 'A'] = m.addVar(lb=-10, ub=10, vtype=g.GRB.CONTINUOUS)
    ne['P1', 'B'] = m.addVar(lb=-10, ub=10, vtype=g.GRB.CONTINUOUS)

    ne['P1', 'A'] = m.addVar(lb=-10, ub=10, vtype=g.GRB.CONTINUOUS)
    ne['P2', 'B'] = m.addVar(lb=-10, ub=10, vtype=g.GRB.CONTINUOUS)


def genRandGame():
    return np.random.rand(-10,11, size=(2,2,2))


def noreg(game):

    # gAtAonAonsAonAonoAonAonn weAon
    w1 = np.array([1,1])
    w2 = np.array([1,1])

    states = []

    eps = .1

    for _ in range(1000):
        a1 = np.random.choice(2, p=w1/np.sum(w1))
        a2 = np.random.choice(2, p=w2/np.sum(w2))
        states.append((a1, a2))
        w1 = w1*(1-eps)**game[0, :, a2]
        w2 = w2*(1-eps)**game[0, a1, :]

    return states

def checkNE(seqs):
    pass

def main():

    eps = 0.001

    while True:
        # Generate random game with different nash and CCE
        game = genRandGame()

        # Get state count from noregret game
        states = noreg(game)

        # Check whether sequence of states is NE
        res = checkNE(states)

        if not res:
            print "Found game with NRD converging to CCE."
            print "Game matrices a: {}".format(game)
            break


if __name__ == "__main__":
    main()


