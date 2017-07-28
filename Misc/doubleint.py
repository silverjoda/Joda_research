import gurobipy as g


UB = 9 #g.GRB.INFINITY

# Cycle through i amount of digits
for i in range(2,20):

    print "Attempting to find solution for {} digits: ".format(i)

    model = g.Model()

    # Variable dictionary
    x = {}

    for k in range(i):
        x[k] = model.addVar(lb=1, ub=UB, vtype=g.GRB.INTEGER)

    # Update variables
    model.update()

    # Constraints
    lhs = g.quicksum([x[k] * (10 ** (i - k - 1)) for k in range(i)])
    rhs = g.quicksum([x[k] * (10 ** (i - k - 2)) for k in range(i - 1)] +
                     [x[i - 1] * (10 ** (i - 1))])

    model.addConstr(2 * lhs, g.GRB.EQUAL, rhs)

    # Solve model
    model.setParam('outputFlag', 0)
    model.optimize()

    if model.status == g.GRB.INFEASIBLE:
        print "Model is infeasible, continuing to next iteration."
        continue

    # Print out solution

    outputStr = ""
    for m in range(i):
        outputStr += str(int(x[m].x))

    print "Found solution for {} digits: {}".format(i, outputStr)

    print "Done. "
    exit()






