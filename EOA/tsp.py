import networkx as nx
import matplotlib.pyplot as plt
import random

# TODO: Write fitness function
# TODO: Write local search optimization

# ======== Define simulation parameters
n_Vertices = 20 # Amount of vertices (only valid if generating random graph)
n_ts = 3 # Amount of travelling salesmen
n_iters = 1000 # Amount of iterations for algorithm
algorithm = 'local' # Choose algorithm from {local, evo, meme}

class Solver:
    def __init__(self):
        pass

def main():
    """
    Run the main multi-traveller TSP optimization

    """

    # Generate a random complete undirected graph of n_V vertices
    # The first vertex is the Depot!
    G = generate_random_tsp_vertices(n_Vertices)

    # Make a solver object
    if algorithm == 'local':
        local_search(G)
    elif algorithm == 'evo':
        evo_search(G)
    elif algorithm == 'meme':
        meme_search(G)
    else:
        print 'Error, algorithm not found, exiting simulation. '
        exit()



def local_search(G_in):

    # Deep copy the graph
    G = G_in[:]

    # First vertex is the depot
    depot = G[0]
    del G[0]

    # Shuffle graph
    random.shuffle(G)

    # Initial boundaries that separate individual agent sequences
    boundaries = [(len(G) / n_ts) * i for i in xrange(1, n_ts)]

    # Define solution: [sequence, boundaries]
    solution = [G, boundaries]

    # Refine G n_iters times using local_search
    for i in xrange(n_iters):

        # Perform an evaluation of current solution on graph G
        solution_info = solution_fitness(solution, depot)

        # Print information
        if i % (n_iters / 100):
            print "Iter: {}/{}: Fitness: {}".format(i, n_iters, solution_info)

        # Perform optimization step



def evo_search(G):
    pass

def meme_search(G):
    pass

def solution_fitness(solution, depot):
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


def generate_random_tsp_vertices(m):
    return None

if __name__ == "__main__":
    main()
