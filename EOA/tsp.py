import networkx as nx
import matplotlib.pyplot as plt

# ======== Define simulation parameters
n_Vertices = 20 # Amount of vertices (only valid if generating random graph)
n_ts = 3 # Amount of travelling salesmen
n_iters = 1000 # Amount of iterations for algorithm
algorithm = 'local' # Choose algorithm from {local, evo, meme}

def main():
    """
    Run the main multi-traveller TSP optimization

    """

    # Generate random undirected connected graph of n_V vertices
    G = generate_random_tsp_graph(n_Vertices)

    # Make a solver object
    if algorithm == 'local':
        from tsp_local_search import local_solver
        solver = local_solver(G, n_ts)
    elif algorithm == 'evo':
        from tsp_local_search import evo_solver
        solver = evo_solver(G, n_ts)
    elif algorithm == 'meme':
        from tsp_local_search import meme_solver
        solver = meme_solver(G, n_ts)
    else:
        solver = None
        print 'Error, algorithm not found, exiting simulation. '
        exit()

    # Perform optimization for n_iters amount
    for i in range(n_iters):

        # Compute solution to problem
        solution = solver.solve(G)

        # Perform an evaluation of current solution on graph G
        fitness = solution_fitness(G, solution)

        # Perform optimization step
        solver.update()



def solution_fitness(G, sol):
    """
    Compute the fitness to solution given graph G

    Parameters
    ----------
    G: graph object
    solution: list ... don't know yet

    Returns: float32, fitness value
    -------

    """
    return None

def generate_graph_from_input(input):
    return None

def generate_random_tsp_graph(m):
    return None

if __name__ == "__main__":
    main()
