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
    return np.random.randint(-10, 11, size=(2,2,2))

def noreg(game, iters):

    w1 = np.array([1.,1.])
    w2 = np.array([1.,1.])

    state_counts = np.zeros((2,2))

    eps = .1

    for _ in range(iters):

        a1 = np.random.choice(2, p=w1/np.sum(w1))
        a2 = np.random.choice(2, p=w2/np.sum(w2))

        state_counts[a1, a2] += 1

        w1 = w1*(1-eps)**game[0, :, a2]
        w2 = w2*(1-eps)**game[0, a1, :]

    return state_counts / float(iters)

def checkNE(u, noreg_cce, eps):

    # Get marginal probabilities from cce
    p1_sup = np.sum(noreg_cce, 1)
    p2_sup = np.sum(noreg_cce, 0)

    # Calculate nash equillibrium from game
    pa = (u[1, 1, 1] - u[1, 1, 0]) / (u[1, 0, 0] - u[1, 1, 0] - u[1, 0, 1] + u[1, 1, 1])
    pc = (u[0, 1, 1] - u[0, 1, 0]) / (u[0, 0, 0] - u[0, 1, 0] - u[0, 0, 1] + u[0, 1, 1])

    # Make NE strategy
    ne_p1_sup = np.array([pa, 1 - pa])
    ne_p2_sup = np.array([pc, 1 - pc])

    return np.linalg.norm(p1_sup - ne_p1_sup, 2) < eps and np.linalg.norm(p2_sup - ne_p2_sup, 2) < eps


def main():

    eps = 0.001
    iters = 0

    while True:

        iters += 1

        # Generate random game
        game = genRandGame()

        # Get cce from noreg dynamics
        noreg_cce = noreg(game, 1000)

        # Check whether game nash is same as found cce by no regret dynamics
        eps = 0.01
        res = checkNE(game, noreg_cce, eps)

        if not res:
            print "Found game with NRD converging to CCE after {} iters.".format(iters)
            print "Game matrices a: {}".format(game)
            break




if __name__ == "__main__":
    main()


