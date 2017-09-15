from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
import matplotlib.pyplot as plt
import numpy as np
from vae_network import *

def main():

    n_episodes = 3000
    batchsize = 64
    z_dim = 64
    lr = 1e-4

    vae = VAE([28,28,1], z_dim, lr)

    # Train
    for i in range(n_episodes):
        X, _ = mnist.train.next_batch(batchsize)
        X_tf = np.reshape(X, [batchsize, 28, 28, 1])
        mse, kl_loss = vae.train(X_tf)
        if i % 100 == 0:
            print "Training ep {}/{}, mse: {}, kl_loss: {}".\
                format(i, n_episodes, mse, kl_loss)

    # Sample images
    n_images = 5
    z = np.random.randn(n_images, z_dim)
    images = vae.sample(z)
    images = np.squeeze(images, axis=3)

    # Plot samples
    fig = plt.figure()
    for i in range(n_images):
        ax = fig.add_subplot(1,5,i+1)
        ax.imshow(images[i], cmap='gray')

    plt.show()

if __name__ == "__main__":
    main()


