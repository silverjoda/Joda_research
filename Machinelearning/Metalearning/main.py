from funcgenerators import *
import tflearn as tfl

def main():

    # 1) Test average sample complexity required to achieve
    # epsilon approximation error for fixed NN for fixed
    # distribution of functions ===========================

    # eps validation error
    epsilon = 0.1

    # Amount of sampled functions
    n_funcs = 1000

    # Function query noise
    func_noise = 0.1

    # Function domain resolution
    func_res = 1000

    # Global NN hyperparameters:
    n_hidden = 2
    dropout_keep = 0.8
    l2_decay = 0.001

    # Function generator
    func_gen = Funcgen()

    # Initialize tensorflow
    tfl.init_graph(num_cores=2, gpu_memory_fraction=0.5)

    for i in range(n_funcs):

        # Sample random function
        func = func_gen.sampleGMM(2, func_noise, func_res)

        # Create NN classifier with random weights

        # Train classifier until eps error is achieved on validation

        # Register required sample count

    # Print average sample complexity for vanilla training:


    # 2) Learn the meta ============================================



    # Evaluate

    print 'Done. '



if __name__ == '__main__':
    main()