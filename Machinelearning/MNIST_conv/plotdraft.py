

import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy

def plot_acc_matrix(test_acc_matrix, title):

    # Averaged test errors
    mat = np.mean(test_acc_matrix, axis=1)

    # Plot vector
    t = np.arange(mat.shape[1])

    # Make figure
    fig = plt.figure()

    # red dashes, blue squares and green triangles
    plt.plot(t, mat[0], 'r-', t, mat[1], 'b-', t, mat[2], 'g-', t, mat[3],
             'y-')

    plt.title(title)
    plt.xlabel("Iters x 1000")
    plt.ylabel("Test accuracy")
    plt.legend(["p = 0.005","p = 0.01","p = 0.1","p = 1"])
    plt.ylim(0,90)

    plt.savefig("{}.png".format(title.replace(" ", "")), dpi=100)
    plt.show()


test_mat = np.random.randint(50, size=(4,6,20))
plot_acc_matrix(test_mat, "sheize")