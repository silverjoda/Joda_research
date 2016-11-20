import matplotlib.pyplot as plt
import random
import math
import numpy as np

# ======== Define simulation parameters
n_Vertices = 12 # Amount of vertices (only valid if generating random graph)
n_ts = 3 # Amount of travelling salesmen
n_iters = int(1e4) # Amount of iterations (generations) for algorithm
algorithm = 'evo' # Choose algorithm from {local, evo, meme}
vertex_xy_range = [0,1000] # Range of coordinates that vertices can have
depot_xy_range = [400,600] # Range of coordinates that depot can have
n_random_perms = n_Vertices*5 # Amount of random permutations
fitness_scaling_constant = 1e6 # Scale the fitness for visual convenience
path_distance_delta_penalty = 5 # Constant which penalizes the difference
# between max and min tours

# Evo algorithm parameters
n_generations = int(1e4) # Amount of times that the algorithm will be run
pop_size = 100 # Size of the population
pop_selection_size = 20 # Amount of parent individuals selected from population


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

    plt.pause(5)

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
        solution_info = evaluate_fitness(solution, depot)

        # Print information
        if i % (n_iters / 100) == 0:
            print "Iter: {}/{}: Fitness: {}".format(i, n_iters, solution_info)

        if i % (n_iters / 100) == 0:
            plot_graph(G, boundaries, depot)
            plt.pause(0.1)

        # Perform optimization step: ===========

        # Keep tabs of the best swap
        best_swap = None

        best_iteration_fitness = 0

        for n in xrange(n_random_perms):

            # SWAP VERTICES ==========================

            #  Generate two random numbers
            rand_v1, rand_v2 = generate_two_unique_random_numbers(0, seq_length)

            # Swap the corresponding vertices
            v_tmp = G[rand_v2]
            G[rand_v2] = G[rand_v1]
            G[rand_v1] = v_tmp

            # # SWAP BOUNDARIES ========================
            #
            # # Generate random boundary index
            # rand_b = random.randint(0, len(boundaries) - 1)
            # rand_swap_dir = random.randint(-1,1)
            #
            # # Augmented boundaries
            # aug_bound = boundaries[:]
            # aug_bound.insert(0, 0)
            # aug_bound.insert(len(aug_bound), seq_length - 1)
            #
            # boundaries_swapped = False
            #
            # if random.random() > 0.8:
            #     if rand_swap_dir == 1:
            #         if aug_bound[rand_b] + 1 < aug_bound[rand_swap_dir]:
            #             # Perform operation
            #             boundaries_swapped = True
            #     elif rand_swap_dir == -1:
            #         if aug_bound[rand_b] - 1 > aug_bound[rand_swap_dir]:
            #             # Perform operation
            #             boundaries_swapped = True
            #
            # boundaries[rand_b] += rand_swap_dir

            # EVALUATE AND UPDATE ====================

            # Evaluate new solution
            fitness = evaluate_fitness(solution, depot)


            if fitness > best_iteration_fitness:
                best_iteration_fitness = fitness
                best_swap = (rand_v1, rand_v2)

            # Swap back
            v_tmp = G[rand_v2]
            G[rand_v2] = G[rand_v1]
            G[rand_v1] = v_tmp

            #boundaries[rand_b] -= rand_swap_dir


        if best_swap is not None:
            # Swap the best greedy solution
            v_tmp = G[best_swap[0]]
            G[best_swap[0]] = G[best_swap[1]]
            G[best_swap[1]] = v_tmp
            #boundaries[best_swap[2]] += best_swap[3]



    plt.pause(5)


def evo_search(G_in):
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

    # Generate population
    population = []
    for i in range(pop_size):

        # Generate random solution
        random.shuffle(G)

        # Add solution to population
        population.append([G[:], boundaries])

    # Run evolutionary update for n_iters generations
    for i in xrange(n_iters):

        # Evaluate fitness for each individual in population
        fitnesses = [evaluate_fitness(p, depot) for p in population]

        # Print fitness of best individual
        if i % (n_iters / 100) == 0:
            print "Iter: {}/{}: Fitness: {}".format(i, n_iters,
                                                    fitnesses[0])

        if i % (n_iters / 100) == 0:
            plot_graph(G, boundaries, depot)
            plt.pause(0.1)

        # Sort population in descending fitness
        sorted_pop = [P for (F, P) in sorted(zip(fitnesses, population),
                                             reverse=True)]

        # Select N from population
        selected_pop = sorted_pop[:pop_selection_size]

        # New population
        new_population = []

        # Make 100 new individuals from selected units
        for i in range(pop_size):

            # Selelct 2 random parents
            r_nums = generate_two_unique_random_numbers(0,pop_selection_size-1)
            p1 = selected_pop[r_nums[0]]
            p2 = selected_pop[r_nums[1]]

            # Make new individual
            offspring = make_new_individual(p1,p2)

            # Add individual to new population
            new_population.append(offspring)

        # Make current population new
        population = new_population[:]




def make_new_individual(p1,p2):

    # Take genotype out of individuals
    gen_1 = p1[0]
    gen_2 = p2[0]

    # Generate crossoverpoints
    random_cutpoints = generate_two_unique_random_numbers(0,len(gen_1)-1)

    # Copy parent to offspring
    offspring = gen_1[:]

    # Copy all values outside the cutpoints from parent 2 to the offspring
    # in an orderly fashion (to maintain validity of solution)




def select_best_N(population, N):
    pass


def meme_search(G):
    pass


def evaluate_fitness(solution, depot):
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

    # Augmented boundaries
    aug_boundaries = boundaries[:]
    aug_boundaries.insert(0, 0)
    aug_boundaries.insert(len(aug_boundaries), len(sequence) - 1)

    for n in range(len(aug_boundaries) - 1):

        # Start by taking length from depot to first vertex
        tour_length = distance(depot, sequence[aug_boundaries[n]])

        # Go over each vertex and increment length
        for i in range(aug_boundaries[n], aug_boundaries[n + 1]):
            tour_length += distance(sequence[i], sequence[i + 1])

        # Add distance from last vertex to depot
        tour_length += distance(sequence[aug_boundaries[n + 1]], depot)

        # Add to total tour length
        tour_lengths.append(tour_length)

    # Part of the fitness is the total length
    total_tour_length = sum(tour_lengths)

    # Penalize for the maximum tour length (force all to be equal length)
    max_tourlength_penalty = max(tour_lengths) - min(tour_lengths)

    # Add all components of penalty
    total_penalty = total_tour_length + \
                    max_tourlength_penalty*path_distance_delta_penalty

    # We want fitness to be lower with higher penalty
    fitness = fitness_scaling_constant*(1/total_penalty)

    return fitness


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


if __name__ == "__main__":
    main()
