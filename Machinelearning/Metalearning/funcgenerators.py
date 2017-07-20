import numpy as np
from numpy import sqrt, pi, exp


class Funcgen:
    '''
    This class serves to generate distributions from function
    model description
    '''
    def __init__(self):
        pass

    def sampleGMM(self, n_comp, samplenoise, res):

        # Generate means and stds
        means = np.random.rand(n_comp) - 0.5
        stds = np.random.rand(n_comp) / 10.

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
        self.funlut = np.sum([1/(s * sqrt(2 * pi)) *
                              exp(-(self.fundomain - m)**2 / (2 * s**2))
                              for m,s in zip(means, stds)])

        # Normalize
        self.funlut /= (np.sum(self.funlut) / res)

    def sample(self, x):
        assert np.all(-1 <= x <= 1)
        idx = (x * (self.res / 2.) + ((self.res - 1) / 2.)).astype(int)
        assert np.all(0 <= idx < self.res)
        return self.funlut[idx] + np.random.randn(len(x)) * self.samplenoise


class Dataprovider:
    def __init__(self, func):
        self.func = func

    def getBatch(self, batchsize):
        pass



