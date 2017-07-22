from funcgenerators import *
from classifiers import *
import tflearn as tfl

np.random.seed(134)

def main():

    # 1) Test average sample complexity required to achieve
    # epsilon approximation error for fixed NN for fixed
    # distribution of functions ===========================

    # eps validation error
    epsilon = 1

    # Amount of sampled functions
    n_funcs = 1000

    # Function query noise
    func_noise = 0.2

    # Function domain resolution
    func_res = 1000

    # Fixed NN hyperparameters:
    input_dim = 1
    output_dim = 1
    n_hidden = 2
    n_hidden_units = 8
    l2_decay = 0.001
    dropout_keep = 0.7
    minibatch_size = 16
    epochs = 30

    # Function generator
    funcgen = Funcgen()

    # Create NN classifier with random weights
    NN = Approximator(input_dim,
                      output_dim,
                      n_hidden,
                      n_hidden_units,
                      l2_decay,
                      dropout_keep)

    # Sample stats
    sample_count_list = []

    print "Starting average sample count evaluation."

    for i in range(n_funcs):

        # Sample random function
        func = funcgen.sampleGMM(2, func_noise, func_res)

        #func.plotfun()
        #exit()

        # Train classifier until eps error is achieved on validation
        sample_count = NN.fituntileps(func, epochs, minibatch_size, epsilon)

        # Register required sample count
        sample_count_list.append(sample_count)

    # Print average sample complexity for vanilla training:
    avg_sample_count = np.mean(sample_count_list)
    print "Average sample count: {}".format(avg_sample_count)

    # 2) Learn the meta ============================================
    print "Starting to learn meta."

    # TODO:


    # Evaluate

    print 'Done. '



if __name__ == '__main__':
    main()