import numpy as np
import matplotlib.pyplot as plt



def plot_acc_matrix(test_acc_matrix, title):

    # Averaged test errors
    #mat = np.mean(test_acc_matrix, axis=1)
    mat = test_acc_matrix

    # Print out final scores
    print np.mean(mat[:,:,-100:], axis=(1,2))

    # Average progress
    #mat = np.mean(mat, axis=1)

    # Plot vector
    t = np.arange(mat.shape[2])

    # Make figure
    fig = plt.figure()

    # red dashes, blue squares and green triangles
    plt.plot(t, mat[0,0], 'r-', t, mat[1,0], 'b-', t, mat[2,0], 'g-')

    #plt.xscale('log')
    plt.title(title)
    plt.xlabel("Iters")
    plt.ylabel("Distance")
    plt.legend(["Local","Evo","Meme"])
    #plt.ylim(0.6,1.7)
    plt.xlim(0, 6000)

    plt.savefig("{}.png".format(title.replace(" ", "")), dpi=100)

    plt.show()


a = np.load("Progresses.npy")

plot_acc_matrix(a, 'Average progress plot for n = 32, m = 3')