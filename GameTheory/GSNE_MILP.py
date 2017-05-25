import numpy as np
import gurobipy as g

a = np.matrix([[2, 1],
               [1, 4]])

b = np.matrix([[-2, 3],
               [-3, 2]])

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

u = m.addVar(lb=-Z, ub=Z, vtype=g.GRB.CONTINUOUS)
v = m.addVar(lb=-Z, ub=Z, vtype=g.GRB.CONTINUOUS)

for i in range(M):
    x[i] = m.addVar(lb=0, ub=1, vtype=g.GRB.CONTINUOUS)
    p[i] = m.addVar(lb=0, vtype=g.GRB.CONTINUOUS)
    w[i] = m.addVar(vtype=g.GRB.BINARY)

for j in range(N):
    y[j] = m.addVar(lb=0, ub=1, vtype=g.GRB.CONTINUOUS)
    q[j] = m.addVar(lb=0, vtype=g.GRB.CONTINUOUS)
    z[j] = m.addVar(vtype=g.GRB.BINARY)

# Update variables
m.update()

# Add constraints
for i in range(M):
    m.addConstr(g.quicksum([a[i,j] * y[j] + q[j] for j in range(N)]) ,
                g.GRB.EQUAL, u)

    m.addConstr(w[i], g.GRB.GREATER_EQUAL, x[i])
    m.addConstr(p[i], g.GRB.LESS_EQUAL, (1 - w[i]) * Z)

for j in range(N):
    m.addConstr(g.quicksum([b[i,j] * x[i] + p[i] for i in range(M)]) ,
                g.GRB.EQUAL, v)

    m.addConstr(z[j], g.GRB.GREATER_EQUAL, y[j])
    m.addConstr(q[j], g.GRB.LESS_EQUAL, (1 - z[j]) * Z)

m.addConstr(g.quicksum(x[i] for i in range(M)), g.GRB.EQUAL, 1)
m.addConstr(g.quicksum(y[j] for j in range(N)), g.GRB.EQUAL, 1)

# Set objective if general sum game
# m.setObjective()

# Solve
m.optimize()

# Info
print "Value u for player 1: {}".format(u.x)
print "Strategy support for player 1: {}".format([x[i].x for i in range(M)])

print "Value v for player 2: {}".format(v.x)
print "Strategy support for player 2: {}".format([y[j].x for j in range(N)])
