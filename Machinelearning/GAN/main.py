from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
import matplotlib.pyplot as plt
import numpy as np
from gan_network import *

def main():

    n_episodes = 2000
    batchsize = 128
    z_dim = 32
    lr = 1e-4

    gan = GAN([28,28,1], z_dim, lr, mnist.train)

    # Train
    for i in range(n_episodes):
        d_loss, g_loss = gan.train_vanilla(batchsize)
        if i % 50 == 0:
            print "Training ep {}/{}, d_loss: {}, g_loss: {}".\
                format(i, n_episodes, d_loss, g_loss)
            X, _ = mnist.train.next_batch(batchsize)
            X_tf = np.reshape(X, [batchsize, 28, 28, 1])
            gan.summarize(X_tf)

    # Sample images
    n_images = 10
    z = np.random.randn(n_images, z_dim)
    samples = gan.generate(z)
    samples = np.squeeze(samples, axis=3)

    # Plot samples
    fig = plt.figure()
    for i in range(n_images):
        ax = fig.add_subplot(2, 5, i + 1)
        ax.imshow(samples[i], cmap='gray')

    plt.show()

if __name__ == "__main__":
    main()


