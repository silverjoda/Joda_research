import numpy as np
from numpy import sqrt, pi, exp



class Funcgen:
    '''
    This class serves to generate distributions from function
    model description
    '''
    def __init__(self):
        pass

    def sampleGMM(self):
        return GMM()


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
        pass




