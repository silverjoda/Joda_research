import numpy as np
from numpy import sqrt, pi, exp
import matplotlib.pyplot as plt

class Funcgen:
    '''
    This class serves to generate distributions from function
    model description
    '''
    def __init__(self):
        pass

    def sampleGMM(self, n_comp, samplenoise, res):

        # Generate means and stds
        means = 1.6 * np.random.rand(n_comp) - 0.8
        stds = np.random.rand(n_comp) / 5.

        return GMM(means, stds, samplenoise, res)

    def sampleSins(self):
        pass

    def sampleLinfuncs(self):
        pass

    def sampleQuadfuncs(self):
        pass


class GMM:
    def __init__(self, means, stds, samplenoise, res):
        assert len(means) == len(stds)
        self.n_comp = len(means)
        self.samplenoise = samplenoise
        self.means = means
        self.stds = stds
        self.res = res

        # function look up table
        self.fundomain = np.linspace(-1, 1, self.res)

        # Create function
        self.funlut = np.zeros_like(self.fundomain)

        for m, s in zip(means, stds):
            self.funlut += 1/(s * sqrt(2 * pi)) * \
                           exp(-(self.fundomain - m)**2 / (2 * s**2))


        # Normalize
        self.funlut /= (np.sum(self.funlut) / res)

    def sample(self, x):
        assert -1 <= x <= 1
        idx = (x * (self.res / 2.) + ((self.res - 1) / 2.)).astype(int)
        assert 0 <= idx < self.res
        return self.funlut[idx] + np.random.randn() * self.samplenoise

    def sampleManyRandom(self, n):
        X = np.random.randint(0, self.res, size=(n))
        Y = self.funlut[X] + np.random.randn(n) * self.samplenoise
        return X, Y

    def plotfun(self):
        plt.plot(self.fundomain, self.funlut)
        plt.show()

def sampleBatch(X, Y, batchsize):

    # Dataset size
    n = len(X)

    # Array
    rnd_ind = np.arange(n)

    # Random shuffle
    np.random.shuffle(rnd_ind)

    # Random dataset
    rndX = X[rnd_ind[:batchsize]]
    rndY = Y[rnd_ind[:batchsize]]

    # Expand dimension for tensorflow
    rndX = np.expand_dims(rndX, 1)
    rndY = np.expand_dims(rndY, 1)

    # Return random subset
    return rndX, rndY
