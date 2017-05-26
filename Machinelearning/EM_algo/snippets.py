import numpy as np
from PIL import Image

# --------------------------------------------------
# Example 1
# load a colour image and put it into a 3D numpy array
name = "foo.png"
im = Image.open(name).convert("RGB")
arr = np.array(im, dtype=np.float64) / 255.0

# -------------------------------------------------
# Example 2
# assume that arr is an n-dimensional numpy array

# apply a function to all elements of the array
parr = np.log(arr / (1 - arr))

# cut of the values in arr by some maxval
arr[arr > maxval] = maxval

# ----------------------------------------------------
# Example 3
# class definition of a multivariate Gaussian (fixed dimension = 3)
class GaussMVD:
    """ Multivariate normal distribution """

    # ===============================================
    def __init__(self):
        self.mean = np.zeros(3, dtype=np.float64)
        self.cov = np.eye(3, dtype=np.float64)
        self.det = np.linalg.det(np.matrix(self.cov))
        self.normc = 1.0 / np.sqrt(np.power((2.0 * np.pi), 3) * self.det)

    # ===============================================
    def modify(self, mean, cov):
        if not ((mean.shape == (3,)) and (cov.shape == (3, 3))):
            raise Exception("Gaussian: shape mismatch!")

        self.mean = np.array(mean, dtype=np.float64)
        self.cov = np.array(cov, dtype=np.float64)

        self.det = np.linalg.det(np.matrix(self.cov))
        self.normc = 1.0 / np.sqrt(np.power((2.0 * np.pi), self.dim) * self.det)

        return None

    # ===============================================
    def compute_probs(self, arr):
        """ compute probabilities for an array of values """

        inv_cov = np.asarray(np.linalg.inv(np.matrix(self.cov)))
        darr = arr - self.mean
        varr = np.sum(darr * np.inner(darr, inv_cov), axis=-1)
        varr = - varr * 0.5
        varr = np.exp(varr) * self.normc

        return varr

    # ===============================================
    def estimate(self, arr, weight):
        """ estimate parameters from data (array of values & array of weights) """
        eweight = weight[..., np.newaxis]
        wsum = np.sum(weight)

 
        dimlist = list(range(len(arr.shape) - 1))
        dimtup = tuple(dimlist)

        # estimate mean
        mean = np.sum(eweight * arr, axis=dimtup) / wsum

        # estimate covariance
        darr = arr - mean
        cov = np.tensordot(darr, darr * ebeta, axes=(dimlist, dimlist)) / wsum

        self.modify(mean, cov)

        return None
    
    # other functions ...