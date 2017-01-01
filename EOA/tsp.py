import matplotlib.pyplot as plt
import random
import math
import numpy as np
from copy import copy
from copy import deepcopy

# ======== Define simulation parameters
n_Vertices = 32 # Amount of vertices (only valid if generating random graph)
n_ts = 3 # Amount of travelling salesmen
n_iters = int(4000) # Amount of iterations (generations) for algorithm
algorithm = 'meme' # Choose algorithm from {local, evo, meme}
vertex_xy_range = [0,1000] # Range of coordinates that vertices can have
depot_xy_range = [400,600] # Range of coordinates that depot can have
n_random_perms = n_Vertices*5 # Amount of random permutations

# between max and min tours

# Evo algorithm parameters
n_generations = int(4000) # Amount of times that the algorithm will be run
pop_size = 400 # Size of the population
pop_selection_size = pop_size/10 # Amount of parent individuals selected from
# population
mutation_alpha = 0.06 # Chance to mutate
bad_acceptance_prob =  0.1

def main():
    """
    Run the main multi-traveller TSP optimization

    """

    PATH = "/home/shagas/Data/SW/Joda_research/EOA/mtsp_data/"
    files = [PATH + "mTSP_50.data",
             PATH + "mTSP_100.data",
             PATH + "mTSP_200.data"]

    n_reps = 7
    fitnesses = np.zeros((4, n_reps, n_iters))

    for i in range(n_reps):
        # Generate a random complete undirected graph of n_V vertices
        # The first vertex is the Depot!
        #G = generate_random_tsp_vertices(n_Vertices)
        G = read_graph_from_file(files[0])

        # Initialize plot
        plt.axis([0, 1000, 0, 1000])
        plt.ion()

        fitnesses[0, i] = local_search(G)
        fitnesses[1, i] = local_imp_search(G)
        fitnesses[2, i] = evo_search(G)
        fitnesses[3, i] = meme_search(G)

    # Save average performance of algorithms
    np.save('Progresses.npy', fitnesses)

    exit()


    np.random.seed(0)
    random.seed(0)
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

    plt.pause(500)

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
    best_fitness = 10000000

    progress = []

    # Refine G n_iters times using local_search
    for i in xrange(n_iters):
        iter_fitness = 10000000

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

        for n in xrange(n_random_perms):

            # SWAP VERTICES ==========================

            #  Generate two random numbers
            rand_v1, rand_v2 = generate_two_unique_random_numbers(0, seq_length)

            # Swap the corresponding vertices
            v_tmp = G[rand_v2]
            G[rand_v2] = G[rand_v1]
            G[rand_v1] = v_tmp

            # EVALUATE AND UPDATE ====================

            # Evaluate new solution
            fitness = evaluate_fitness(solution, depot)

            if fitness < best_fitness:
                best_fitness = fitness
                best_swap = (rand_v1, rand_v2)

            # Swap back
            v_tmp = G[rand_v2]
            G[rand_v2] = G[rand_v1]
            G[rand_v1] = v_tmp

        if best_swap is not None:
            # Swap the best greedy solution
            v_tmp = G[best_swap[0]]
            G[best_swap[0]] = G[best_swap[1]]
            G[best_swap[1]] = v_tmp

        #best_fitness = iter_fitness
        progress.append(best_fitness)

    return progress


def local_imp_search(G_in):

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
    best_fitness = 10000000

    progress = []

    # Refine G n_iters times using local_search
    for i in xrange(n_iters):
        iter_fitness = 10000000

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

        for n in xrange(n_random_perms):

            # SWAP VERTICES ==========================

            #  Generate two random numbers
            rand_v1, rand_v2 = generate_two_unique_random_numbers(0, seq_length)

            # Swap the corresponding vertices
            v_tmp = G[rand_v2]
            G[rand_v2] = G[rand_v1]
            G[rand_v1] = v_tmp

            # EVALUATE AND UPDATE ====================

            # Evaluate new solution
            fitness = evaluate_fitness(solution, depot)

            if fitness < iter_fitness:
                iter_fitness = fitness
                best_swap = (rand_v1, rand_v2)

            # Swap back
            v_tmp = G[rand_v2]
            G[rand_v2] = G[rand_v1]
            G[rand_v1] = v_tmp

        if best_swap is not None:
            # Swap the best greedy solution
            v_tmp = G[best_swap[0]]
            G[best_swap[0]] = G[best_swap[1]]
            G[best_swap[1]] = v_tmp

        #best_fitness = iter_fitness
        progress.append(iter_fitness)

    return progress


def evo_search(G_in):
    # Deep copy the graph
    G = deepcopy(G_in)

    # First vertex is the depot
    depot = G[0]
    del G[0]

    # Length of sequence
    seq_length = len(G)

    # Shuffle graph
    random.shuffle(G)

    # Initial boundaries that separate individual agent sequences
    boundaries = [(seq_length / n_ts) * i for i in xrange(1, n_ts)]

    # Opt progress array
    progress = []

    # Generate population
    population = []

    for i in range(pop_size):

        # Generate random solution
        random.shuffle(G)

        # Add solution to population
        population.append([deepcopy(G), deepcopy(boundaries)])

    # Run evolutionary update for n_iters generations
    for i in xrange(n_generations):

        # Evaluate fitness for each individual in population
        fitnesses = [evaluate_fitness(p, depot) for p in population]


        # Sort population in descending fitness ***
        population = [P for (F, P) in sorted(zip(fitnesses, population),
                                             reverse=False,
                                             key=lambda pair: pair[0])]

        # Sort fitnesses
        fitnesses.sort(reverse=False)

        # Progress array
        progress.append(evaluate_fitness(population[0], depot))

        if i % (n_iters / 100) == 0:
            plot_graph(population[0][0], population[0][1], depot)

        # Print fitness of best individual
        if i % (n_iters / 100) == 0:
            print "Generation: {}/{}: Fitness of best: {}, mean: {}, " \
                  "median: {}".format(i,
                                      n_iters,
                                      evaluate_fitness(population[0], depot),
                                      np.mean(fitnesses),
                                      np.median(fitnesses))

            plt.hist(fitnesses, bins=50)
            plt.pause(0.1)


        # Invert fitnesses
        ev_fitnesses = np.array([1/f for f in fitnesses])

        # Average fitness
        average_fitness = np.mean(ev_fitnesses)

        # Max fitness
        max_fitness = np.max(ev_fitnesses)

        # Find rescale mappings
        sig = max_fitness/average_fitness
        a = (-average_fitness*(1-sig))/(max_fitness - average_fitness)
        b = average_fitness*(1-a)

        ev_fitnesses = a*ev_fitnesses + b

        # Calculate total fitness
        total_fitness = sum(ev_fitnesses)
        fit_probs = np.array(ev_fitnesses)/total_fitness

        # Make the choices of individuals
        idxs1 = [np.random.choice(len(fitnesses), p=fit_probs) for _ in
                 range(pop_selection_size)]
        idxs2 = [np.random.choice(len(fitnesses), p=fit_probs) for _ in
                 range(pop_selection_size)]

        # Make n new individuals from selected units
        for idx1, idx2 in zip(idxs1, idxs2):

            p1 = population[idx1]
            p2 = population[idx2]

            # Make new individuals
            offspring1 = make_new_individual(p1,p2)
            offspring2 = make_new_individual(p1,p2)

            if evaluate_fitness(offspring1, depot) < evaluate_fitness(p1, depot):
                population[idx1] = deepcopy(offspring1)
            elif random.random() < bad_acceptance_prob:
                population[idx1] = deepcopy(offspring1)

            if evaluate_fitness(offspring2, depot) < evaluate_fitness(p2, depot):
                population[idx2]= deepcopy(offspring2)
            elif random.random() < bad_acceptance_prob:
                population[idx2] = deepcopy(offspring2)

        for n in xrange(np.random.randint(1,pop_size/10)):
            # SWAP VERTICES ==========================

            #  Generate two random numbers
            rand_v1, rand_v2 = generate_two_unique_random_numbers(0, seq_length)

            # Swap the corresponding vertices
            v_tmp = population[rand_v2]
            population[rand_v2] = deepcopy(population[rand_v1])
            population[rand_v1] = deepcopy(v_tmp)

    return progress

def meme_search(G_in):
    # Deep copy the graph
    G = deepcopy(G_in)

    # First vertex is the depot
    depot = G[0]
    del G[0]

    # Length of sequence
    seq_length = len(G)

    # Shuffle graph
    random.shuffle(G)

    # Initial boundaries that separate individual agent sequences
    boundaries = [(seq_length / n_ts) * i for i in xrange(1, n_ts)]

    # Best fitness so far
    best_fitness = 0
    progress = []

    # Generate population
    population = []

    for i in range(pop_size):

        # Generate random solution
        random.shuffle(G)

        # Add solution to population
        population.append([deepcopy(G), deepcopy(boundaries)])


    # Run evolutionary update for n_iters generations
    for i in xrange(n_generations):

        # Evaluate fitness for each individual in population
        fitnesses = [evaluate_fitness(p, depot) for p in population]


        # Sort population in descending fitness ***
        population = [P for (F, P) in sorted(zip(fitnesses, population),
                                             reverse=False,
                                             key=lambda pair: pair[0])]

        # Sort fitnesses
        fitnesses.sort(reverse=False)

        progress.append(evaluate_fitness(population[0], depot))

        if i % (n_iters / 100) == 0:
            plot_graph(population[0][0], population[0][1], depot)
            plt.hist(fitnesses, bins=50)
            plt.pause(0.1)

        # Print fitness of best individual
        if i % (n_iters / 100) == 0:
            print "Generation: {}/{}: Fitness of best: {}, mean: {}, " \
                  "median: {}".format(i,
                                      n_iters,
                                      evaluate_fitness(population[0], depot),
                                      np.mean(fitnesses),
                                      np.median(fitnesses))



        # Invert fitnesses
        ev_fitnesses = np.array([1 / f for f in fitnesses])

        # Average fitness
        average_fitness = np.mean(ev_fitnesses)

        # Max fitness
        max_fitness = np.max(ev_fitnesses)

        # Find rescale mappings
        sig = max_fitness / average_fitness
        a = (-average_fitness * (1 - sig)) / (max_fitness - average_fitness)
        b = average_fitness * (1 - a)

        ev_fitnesses = a * ev_fitnesses + b

        # Calculate total fitness
        total_fitness = sum(ev_fitnesses)
        fit_probs = np.array(ev_fitnesses) / total_fitness

        # Make the choices of individuals
        idxs1 = [np.random.choice(len(fitnesses), p=fit_probs) for _ in
                 range(pop_selection_size)]
        idxs2 = [np.random.choice(len(fitnesses), p=fit_probs) for _ in
                 range(pop_selection_size)]


        # Make 100 new individuals from selected units
        for idx1, idx2 in zip(idxs1, idxs2):

            p1 = population[idx1]
            p2 = population[idx2]

            # Make new individuals
            offspring1 = make_new_individual(p1,p2)
            offspring2 = make_new_individual(p1,p2)

            # Perform 2-opt on individuals
            offspring1 = perform_2opt(offspring1, depot)
            offspring2 = perform_2opt(offspring2, depot)

            if evaluate_fitness(offspring1, depot) < evaluate_fitness(p1, depot):
                population[idx1] = deepcopy(offspring1)
            elif random.random() < mutation_alpha:
                population[idx1] = deepcopy(offspring1)

            if evaluate_fitness(offspring2, depot) < evaluate_fitness(p2, depot):
                population[idx2]= deepcopy(offspring2)
            elif random.random() < mutation_alpha:
                population[idx2] = deepcopy(offspring2)

        # Perform 2-opt on a portion of the population
        idx_arr = np.arange(pop_size)
        np.random.shuffle(idx_arr)
        idxs2opt = idx_arr[:pop_size / 10]

        for idx in idxs2opt:
            population[idx] = perform_2opt(population[idx], depot)


        # Perform random mutations
        for n in xrange(np.random.randint(1, pop_size / 10)):
            # SWAP VERTICES ==========================

            #  Generate two random numbers
            rand_v1, rand_v2 = generate_two_unique_random_numbers(0, seq_length)

            # Swap the corresponding vertices
            v_tmp = population[rand_v2]
            population[rand_v2] = deepcopy(population[rand_v1])
            population[rand_v1] = deepcopy(v_tmp)

    # Print info to file
    f = open('/home/shagas/Data/SW/Joda_research/EOA/output.txt', 'w')
    f.write("50\n")
    f.write("3\n")
    f.write("mTSP_50.data\n")

    seq = population[0][0]
    bounds = population[0][1]

    f.write(seq[:bounds[0]])
    f.write(seq[bounds[0]:bounds[1]])
    f.write(seq[bounds[1]:])

    f.close()



    return progress

def perform_2opt(sol, depot):

    new_ind = deepcopy(sol)


    # Go over each node in the solution
    for i in range(n_Vertices/7):

        # Pick random node from the solution
        rnd_num = random.randint(0, len(new_ind[0]) - 2)

        # Find second suitable edge
        while True:
            rnd_num_2 = random.randint(0, len(new_ind[0]) - 2)
            if abs(rnd_num_2 - rnd_num) > 1:
                break


        # Edge 1
        edge_1 = [new_ind[0][rnd_num], new_ind[0][rnd_num + 1]]

        # Edge 1
        edge_2 = [new_ind[0][rnd_num_2], new_ind[0][rnd_num_2 + 1]]

        # If edges cross then swap and break
        if edges_cross(edge_1, edge_2):

            # Make two sequences
            testseq1 = deepcopy(new_ind[0])
            testseq2 = deepcopy(new_ind[0])

            # Make swap on first sequence
            tmp = testseq1[rnd_num + 1]
            testseq1[rnd_num + 1] = testseq1[rnd_num_2]
            testseq1[rnd_num_2] = tmp

            testIndividual1 = deepcopy([testseq1, new_ind[1]])

            # Make swap on second sequence
            tmp = testseq2[rnd_num + 1]
            testseq2[rnd_num + 1] = testseq2[rnd_num_2 + 1]
            testseq2[rnd_num_2 + 1] = tmp

            testIndividual2 = deepcopy([testseq2, new_ind[1]])

            if evaluate_fitness(testIndividual1, depot) > \
                    evaluate_fitness(
                    testIndividual2, depot):
                new_ind = deepcopy(testIndividual1)
            else:
                new_ind = deepcopy(testIndividual2)

    return new_ind

def edges_cross(edge1, edge2):
    [A, B] = edge1
    [C, D] = edge2
    return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)


def ccw(A, B, C):
    return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])


def make_new_individual(p1,p2):

    # Take genotype out of individuals
    gen_1 = deepcopy(p1[0])
    gen_2 = deepcopy(p2[0])

    gene_length = len(gen_1)

    # Generate crossoverpoints
    r_vec = generate_two_unique_random_numbers(0, gene_length-1)

    if r_vec[0] < r_vec[1]:
        r1 = r_vec[0]
        r2 = r_vec[1]
    else:
        r2 = r_vec[0]
        r1 = r_vec[1]

    # Copy parent to offspring
    offspring = deepcopy(gen_1)

    # TODO:  FIX: NOT CORRECT ORDER FILTERING
    # Filter remaining material from gen_2
    remainder = deepcopy([g for g in gen_2 if g not in offspring[r1:r2]])

    # Copy all values outside the cutpoints from parent 2 to the offspring
    # in an orderly fashion (to maintain validity of solution)
    idx = 0
    for i in range(0, r1):
        offspring[idx] = deepcopy(remainder[idx])
        idx += 1

    for i in range(r2, gene_length):
        offspring[idx + r2 - r1] = deepcopy(remainder[idx])
        idx += 1


    # Random swap
    if random.random() < mutation_alpha:
        r_vec = generate_two_unique_random_numbers(0, gene_length - 1)
        tmp = offspring[r_vec[0]]
        offspring[r_vec[0]] = deepcopy(offspring[r_vec[1]])
        offspring[r_vec[1]] = tmp


    # print 'Xoverpoints: {},{}'.format(r1,r2)
    # print gen_1
    # print gen_2
    # print offspring
    # exit()

    boundaries = deepcopy(p1[1])
    #
    # Make mutation
    for i in range(len(boundaries)):
        # Chance
        if random.random() < (mutation_alpha/n_ts):
            # Shift to right or left
            randshift = random.randint(0,1)*2-1

            if randshift == -1:
                if (i == 0 and boundaries[i] > 1) or (i > 0 and boundaries[i] > boundaries[i-1] + 1):
                    boundaries[i] -= 1
            else:
                if (i == (len(boundaries) - 1) and boundaries[i] < gene_length - 2) \
                        or (i < (len(boundaries) - 1) and boundaries[i] < boundaries[i+1] - 1):
                    boundaries[i] -= 1

    # Return new individual
    return [offspring, boundaries]


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

    return max(tour_lengths)


def generate_random_tsp_vertices(m):

    G = []

    for i in range(m):
        G.append([random.randint(vertex_xy_range[0], vertex_xy_range[1]),
                  random.randint(vertex_xy_range[0], vertex_xy_range[1])])

    # Generate depot separately so it's close to the center

    G[0] = [random.randint(depot_xy_range[0], depot_xy_range[1]),
            random.randint(depot_xy_range[0], depot_xy_range[1])]

    return G

def read_graph_from_file(filename):
    file = open(filename, 'r')

    # Trash
    [file.readline() for _ in range(4)]

    G = []

    for str in file:
        G.append([int(s) for s in str.split() if s.isdigit()])

    file.close()

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