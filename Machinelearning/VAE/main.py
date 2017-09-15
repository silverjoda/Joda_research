from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
import matplotlib.pyplot as plt
import numpy as np
from vae_network import *

def main():

    n_episodes = 1000
    batchsize = 64
    z_dim = 32
    lr = 5e-4

    vae = VAE([28,28,1], z_dim, lr)

    exit()

    # Train
    for i in range(n_episodes):
        batch = mnist.train.next_batch(batchsize)
        mse, kl_loss = vae.train(batch)
        if i % 10 == 0:
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
        ax.imshow(images[i])


if __name__ == "__main__":
    main()

