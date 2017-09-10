import numpy as np
from embedder import *
from sklearn.metrics import mean_squared_error
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
import time


def main():

    mnist = input_data.read_data_sets('MNIST_data')
    X, _ = mnist.train.images, mnist.train.labels

    X = X.reshape((-1, 1, 28, 28))
    mu, sigma = np.mean(X.flatten()), np.std(X.flatten())

    print("mu, sigma:", mu, sigma)

    X_normalized = (X - mu) / sigma
    X_normalized = np.transpose(X_normalized, axes=[0,2,3,1])

    batchsize = 64
    n_episodes = 2000

    # Train baseline autoencoder
    base_AE = Autoencoder([28,28], None)

    t1 = time.time()
    for i in range(n_episodes):
        err = base_AE.train(
            X_normalized[np.random.randint(0,50000, size=batchsize)])
        print("Episode {}/{}, err: {}".format(i, n_episodes, err))
    print("Training of base_AE took: {}".format(time.time() - t1))

    # Train virtual autoencoder
    virt_AE = Vencoder([28, 28], None)

    t1 = time.time()
    for i in range(int(n_episodes)):
        err = virt_AE.train(
            X_normalized[np.random.randint(0,50000, size=batchsize)])
        print("Episode {}/{}, err: {}".format(i, n_episodes, err))
    print("Training of virt_AE took: {}".format(time.time() - t1))

    t1 = time.time()
    for i in range(int(n_episodes)):
        err = virt_AE.train_upconv_decoder(
            X_normalized[np.random.randint(0, 50000, size=batchsize)])
        print("Episode {}/{}, err: {}".format(i, n_episodes, err))
    print("Training of virt_AE decoder took: {}".format(time.time() - t1))

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

    v_recon_upconv = virt_AE.upconv_reconstruct(imgs_norm)
    v_recon_upconv = np.squeeze(v_recon_upconv, axis=3)
    v_recon_upconv = (v_recon_upconv * sigma) + mu

    fig = plt.figure()
    for i in range(5):
        ax1 = fig.add_subplot(4, 5, i + 1)
        ax1.imshow(imgs[i,0,:,:], cmap="gray")

        ax2 = fig.add_subplot(4, 5, i + 6)
        ax2.imshow(base_recon[i,:,:], cmap="gray")
        ax2.set_xlabel(imgmse(imgs[i,0,:,:], base_recon[i,:,:]))

        ax3 = fig.add_subplot(4, 5, i + 11)
        ax3.imshow(v_recon[i,:,:], cmap="gray")
        ax3.set_xlabel(imgmse(imgs[i,0,:,:], v_recon[i,:,:]))

        ax4 = fig.add_subplot(4, 5, i + 16)
        ax4.imshow(v_recon_upconv[i, :, :], cmap="gray")
        ax4.set_xlabel(imgmse(imgs[i, 0, :, :], v_recon_upconv[i, :, :]))

    plt.show()#

def imgmse(im1, im2):
    return mean_squared_error(im1,im2)



if __name__ == "__main__":
    main()