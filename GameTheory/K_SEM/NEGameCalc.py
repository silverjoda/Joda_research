import numpy as np
import gurobipy as g

def actTonum(act):
    actions = ('R', 'S', 'P', 'F')
    return actions.index(act)

def calcILPNE(a, b, acts):
    '''

    Parameters
    ----------
    a Game matrix for player a
    b Game matrix for player b

    Returns
    -------
    u,v,s(a),s(b)

    '''

    support = np.zeros(a.shape[0])

    for act in acts:
        support[actTonum(act)] = 1

    M, N = a.shape

    # Gurobi model
    m = g.Model()

    # Variables
    x = {}
    y = {}
    p = {}
    q = {}
    w = {}
    z = {}

    # Big Z
    Z = 1000

    u = m.addVar(lb=-Z, ub=Z, vtype=g.GRB.CONTINUOUS, obj=-1)
    v = m.addVar(lb=-Z, ub=Z, vtype=g.GRB.CONTINUOUS, obj=-1)

    for i in range(M):
        x[i] = m.addVar(lb=0, ub=support[i], vtype=g.GRB.CONTINUOUS)
        p[i] = m.addVar(lb=0, ub=g.GRB.INFINITY, vtype=g.GRB.CONTINUOUS)
        w[i] = m.addVar(vtype=g.GRB.BINARY)

    for j in range(N):
        y[j] = m.addVar(lb=0, ub=support[i], vtype=g.GRB.CONTINUOUS)
        q[j] = m.addVar(lb=0, ub=g.GRB.INFINITY, vtype=g.GRB.CONTINUOUS)
        z[j] = m.addVar(vtype=g.GRB.BINARY)

    # Update variables
    m.update()

    # Add constraints
    for i in range(M):
        m.addConstr(g.quicksum([a[i,j] * y[j] + q[i] for j in range(N)]) ,
                    g.GRB.EQUAL, u)

        m.addConstr(w[i], g.GRB.GREATER_EQUAL, x[i])
        m.addConstr(p[i], g.GRB.LESS_EQUAL, (1 - w[i]) * Z)

    for j in range(N):
        m.addConstr(g.quicksum([b[i,j] * x[i] + p[j] for i in range(M)]) ,
                    g.GRB.EQUAL, v)

        m.addConstr(z[j], g.GRB.GREATER_EQUAL, y[j])
        m.addConstr(q[j], g.GRB.LESS_EQUAL, (1 - z[j]) * Z)

    # Probabilities should sum to 1
    m.addConstr(g.quicksum(x[i] for i in range(M)), g.GRB.EQUAL, 1)
    m.addConstr(g.quicksum(y[j] for j in range(N)), g.GRB.EQUAL, 1)

    # Set objective if general sum game
    # m.setObjective()

    # Solve
    m.setParam('outputFlag', 0)
    m.optimize()

    if m.status == g.GRB.INFEASIBLE:
        m.computeIIS()
        m.write("model.ilp")
        print "Error ... NE is infeasible, exiting..."
        exit()

    # # Info
    # print "Value u for player 1: {}".format(u.x)
    # print "Strategy support for player 1: {}".format([x[i].x for i in range(M)])
    #
    # print "Value v for player 2: {}".format(v.x)
    # print "Strategy support for player 2: {}".format([y[j].x for j in range(N)])

    return u.x, v.x, [x[i].x for i in range(M)], [y[j].x for j in range(N)]