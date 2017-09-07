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
    X_normalized = np.transpose(X_normalized, axes=[0,2,3,1])

    # Train baseline autoencoder

    # Train virtual autoencoder

    # Compare



if __name__ == "__main__":
    main()