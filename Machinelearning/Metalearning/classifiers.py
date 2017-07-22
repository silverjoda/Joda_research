import tensorflow as tf
import tflearn as tfl
import numpy as np
from funcgenerators import sampleBatch

np.random.seed(1337)

# Define GPU usage
GPU = 0

class Approximator:
    def __init__(self, input_dim, output_dim, n_hidden, n_hidden_units, l2_decay, dropout_keep):

        self.X = tf.placeholder("float", shape=[None, input_dim])
        self.Y = tf.placeholder("float", shape=[None, output_dim])

        self.net = tfl.fully_connected(self.X, n_hidden_units,activation='relu')
        self.net = tfl.dropout(self.net, dropout_keep)

        for i in range(n_hidden):
            self.net = tfl.fully_connected(self.net, n_hidden_units, activation='relu')
            self.net = tfl.dropout(self.net, dropout_keep)

        self.prediction = tfl.fully_connected(self.net, output_dim, activation='linear')

        # Backward propagation
        self.loss = tf.reduce_mean(tf.squared_difference(self.prediction, self.Y))
        self.train = tf.train.AdamOptimizer(0.001).minimize(self.loss)

        config = tf.ConfigProto(
            device_count={'GPU': GPU}
        )

        self.sess = tf.Session(config=config)

    def fituntileps(self, func, epochs, batchsize, epsilon):

        # Initial sample count
        samplesize = 10
        while(True):

            self.sess.run(tf.initialize_all_variables())

            # Obtain batch of data
            X, Y = func.sampleManyRandom(samplesize)

            # Train proportionally to dataset size
            for i in range(5 * (samplesize / batchsize + 1)):

                bX, bY = sampleBatch(X, Y, batchsize)

                # Train on batch
                self.sess.run(self.train, feed_dict={self.X : bX,
                                                     self.Y : bY})

            # Evaluate
            err = self.eval(func)
            print err, samplesize; exit()

            #print err
            #exit()

            if err < epsilon:
                break

            samplesize += 10

        return samplesize

    def eval(self, func):

        # Amount of test points
        n = 1024

        # Obtain batch of data
        X, Y = func.sampleManyRandom(n)

        # Expand dimension for tensorflow
        X = np.expand_dims(X, 1)
        Y = np.expand_dims(Y, 1)

        # Evaluate MSE loss on test dataset
        mse = self.sess.run(self.loss, feed_dict={self.X: X, self.Y: Y})

        return mse


class GradNet:
    pass

class QueryNet:
    pass

#tclass
