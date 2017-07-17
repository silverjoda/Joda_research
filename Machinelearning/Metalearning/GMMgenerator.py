
class GMMgen:
    '''
    This class serves to generate distributions from given mixture
    model description
    '''
    def __init__(self, n_comp, variability):
        self.n_comp = n_comp
        self.variability = variability

    def sampleGMM(self):
        return GMM()


class GMM:
    def __init__(self, means, stds):
        assert len(means) == len(stds)
        self.n_comp = len(means)
        



