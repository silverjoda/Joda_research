import numpy as np
import tensorflow as tf
import tflearn as tfl
from embedder import *

import sys
import os
import urllib
import gzip
import cPickle
from PIL import Image

def _network():
    pass


def main():

    # Create the autoencoder network

    # Create the dataset
    fname = 'mnist.pkl.gz'
    if not os.path.isfile(fname):
        testfile = urllib.URLopener()
        testfile.retrieve("http://deeplearning.net/data/mnist/mnist.pkl.gz",
                          fname)
    f = gzip.open(fname, 'rb')
    train_set, valid_set, test_set = cPickle.load(f)
    f.close()
    X, y = train_set
    X = X.reshape((-1, 1, 28, 28))
    mu, sigma = np.mean(X.flatten()), np.std(X.flatten())

    print "mu, sigma:", mu, sigma

    X_normalized = (X - mu) / sigma

    # we need our target to be 1 dimensional
    X_out = X_normalized.reshape((X_normalized.shape[0], -1))

    pass


if __name__ == "__main__":
    main()