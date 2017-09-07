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
import time

import matplotlib.pyplot as plt


def main():


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

    batchsize = 128
    n_episodes = 50000/batchsize

    # Train baseline autoencoder
    base_AE = Autoencoder([28,28], None)

    t1 = time.time()
    for i in range(n_episodes):
        err = base_AE.train(X_normalized[i*batchsize:i*batchsize + batchsize])
        print("Episode {}/{}, err: {}".format(i, n_episodes, err))
    print("Training of base_AE took: {}".format(time.time() - t1))


    # Train virtual autoencoder
    virt_AE = Vencoder([28, 28], None)

    t1 = time.time()
    for i in range(n_episodes):
        err = virt_AE.train(
            X_normalized[i * batchsize:i * batchsize + batchsize])
        print("Episode {}/{}, err: {}".format(i, n_episodes, err))
    print("Training of virt_AE took: {}".format(time.time() - t1))

    # Compare
    rand_vec = np.random.randint(0, 1000, size=(5))

    imgs = X[rand_vec]
    imgs_norm = X_normalized[rand_vec]

    base_recon = base_AE.reconstruct(imgs_norm)
    base_recon = np.squeeze(base_recon, axis=3)
    base_recon = (base_recon * sigma) + mu

    v_recon = virt_AE.reconstruct(imgs_norm)
    v_recon = np.squeeze(v_recon, axis=3)
    v_recon = (v_recon * sigma) + mu

    fig = plt.figure()
    for i in range(5):
        ax1 = fig.add_subplot(3, 5, i + 1)
        ax1.imshow(imgs[i,0,:,:])

        ax2 = fig.add_subplot(3, 5, i + 6)
        ax2.imshow(base_recon[i,:,:])

        ax3 = fig.add_subplot(3, 5, i + 11)
        ax3.imshow(v_recon[i,:,:])

    plt.show()#


if __name__ == "__main__":
    main()