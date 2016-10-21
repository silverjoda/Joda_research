import matplotlib.pyplot as plt
import random
import math

# TODO: Write fitness function
# TODO: Write local search optimization

# ======== Define simulation parameters
n_Vertices = 12 # Amount of vertices (only valid if generating random graph)
n_ts = 3 # Amount of travelling salesmen
n_iters = 50000 # Amount of iterations for algorithm
algorithm = 'local' # Choose algorithm from {local, evo, meme}
vertex_xy_range = [0,1000] # Range of coordinates that vertices can have
depot_xy_range = [400,600] # Range of coordinates that depot can have

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

    # Initialize plot
    plt.axis([0, 1000, 0, 1000])
    plt.ion()

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


def plot_graph(G, boundaries, depot):

    # Scatter vertices
    plt.cla()
    plt.scatter(depot[0], depot[1], color='red')
    plt.scatter([v[0] for v in G], [v[1] for v in G], color='blue')

    # Plot agent paths
    plot_boundaries = boundaries[:]
    plot_boundaries.insert(0, 0)
    plot_boundaries.insert(len(plot_boundaries), len(G))

    colors = ['c','m','y','k','r','g','b']

    for i in range(len(plot_boundaries) - 1):
        # Get x and y coordinates of the route
        G_xs = [v[0] for v in G[plot_boundaries[i]:plot_boundaries[i + 1]]]
        G_ys = [v[1] for v in G[plot_boundaries[i]:plot_boundaries[i + 1]]]

        # Insert depo coordinates
        G_xs.insert(0, depot[0])
        G_xs.insert(len(G_xs), depot[0])
        G_ys.insert(0, depot[1])
        G_ys.insert(len(G_ys), depot[1])

        plt.plot(G_xs, G_ys, color=colors[i])


def local_search(G_in):

    # Deep copy the graph
    G = G_in[:]

    # First vertex is the depot
    depot = G[0]
    del G[0]

    # Length of sequence
    seq_length = len(G)


    # Shuffle graph
    random.shuffle(G)

    # Initial boundaries that separate individual agent sequences
    boundaries = [(seq_length / n_ts) * i for i in xrange(1, n_ts)]

    # Define solution: [sequence, boundaries]
    solution = [G, boundaries]

    # Best fitness so far
    best_fitness = 0


    # Refine G n_iters times using local_search
    for i in xrange(n_iters):

        # Perform an evaluation of current solution on graph G
        solution_info = solution_fitness(solution, depot)

        # Print information
        if i % (n_iters / 100) == 0:
            print "Iter: {}/{}: Fitness: {}".format(i, n_iters, solution_info)

        if i % (n_iters / 10) == 0:
            plot_graph(G, boundaries, depot)
            plt.pause(0.1)

        # Perform optimization step: ===========

        # Generate two random numbers
        rand_v1, rand_v2 = generate_two_unique_random_numbers(0, seq_length)

        # Swap the corresponding vertices
        v_tmp = G[rand_v2]
        G[rand_v2] = G[rand_v1]
        G[rand_v1] = v_tmp

        # Evaluate new solution
        fitness = solution_fitness(solution, depot)

        # If worse then swap back
        if fitness > best_fitness:
            best_fitness = fitness
        else:
            # Swap back with small probability
            if random.random() < 0.95:
                v_tmp = G[rand_v2]
                G[rand_v2] = G[rand_v1]
                G[rand_v1] = v_tmp

        # Move boundaries
        old_boundaries = boundaries[:]

        for m in range(len(boundaries)):

            # Change with small probability
            if random.random() < 0.2:
                rand_shift = random.randint(0, 2) - 1

                if rand_shift == -1:
                    if m > 0:
                        if boundaries[m] > boundaries[m - 1] + 1:
                            boundaries[m] -= 1
                    elif boundaries[m] - 1 > 1:
                        boundaries[m] -= 1

                elif rand_shift == 1:
                    if m < len(boundaries) - 1:
                        if boundaries[m] < boundaries[m + 1] - 1:
                            boundaries[m] += 1
                    elif boundaries[m] + 1 < seq_length:
                        boundaries[m] += 1


                    # Evaluate new solution
        fitness = solution_fitness(solution, depot)

        # If worse then swap back
        if fitness > best_fitness:
            best_fitness = fitness
        else:
            if random.random() < 0.95:
                boundaries = old_boundaries[:]


    plt.pause(5)


def evo_search(G):
    pass


def meme_search(G):
    pass


def solution_fitness(solution, depot):
    """
    Compute the fitness to solution

    Parameters
    ----------
    solution: List of vertices, boundaries

    Returns: float32, fitness value
    -------

    """

    # Unpack
    sequence, boundaries = solution

    # Total length of tour(s)
    tour_lengths = []

    for n in range(len(boundaries) - 1):

        # Start by taking length from depot to first vertex
        tour_length = distance(depot, sequence[0])

        # Go over each vertex and increment length
        for i in range(boundaries[n], boundaries[n + 1]):
            tour_length += distance(sequence[i], sequence[i + 1])

        # Add distance from last vertex to depot
        tour_length += distance(sequence[-1], depot)

        # Add to total tour length
        tour_lengths.append(tour_length)

    # Part of the fitness is the total length
    total_tour_length = sum(tour_lengths)

    # Penalize for the maximum tour length (force all to be equal length)
    max_tourlength_penalty = max(tour_lengths)

    # Add all components of penalty
    total_penalty = total_tour_length + max_tourlength_penalty

    # We want fitness to be lower with higher penalty
    fitness = 1/total_penalty

    return fitness


def distance(v1, v2):
    """

    Parameters  Vertices v1, v2
    ----------
    v1 list, (x,y) coordinates
    v2 list, (x,y) coordinates

    Returns euclidean distance between the two verteces
    -------

    """
    return math.sqrt((v1[0] - v2[0])**2 + (v1[1] - v2[1])**2)


def generate_random_tsp_vertices(m):

    G = []

    for i in range(m):
        G.append([random.randint(vertex_xy_range[0], vertex_xy_range[1]),
                  random.randint(vertex_xy_range[0], vertex_xy_range[1])])

    # Generate depot separately so it's close to the center

    G[0] = [random.randint(depot_xy_range[0], depot_xy_range[1]),
            random.randint(depot_xy_range[0], depot_xy_range[1])]

    return G

def generate_two_unique_random_numbers(low, high):

    rand_1 = random.randint(low, high - 1)
    rand_2 = None

    while True:
        rand_2 = random.randint(low, high - 1)
        if not rand_1 == rand_2:
            break

    return rand_1, rand_2

if __name__ == "__main__":
    main()
