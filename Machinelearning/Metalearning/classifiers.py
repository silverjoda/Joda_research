import tensorflow as tf
import tflearn as tfl
import numpy as np

class Approximator:
    def __init__(self, input_dim, output_dim, n_hidden, n_hidden_units, l2_decay, dropout_keep):

        self.X = tf.placeholder("float", shape=[None, input_dim])
        self.Y = tf.placeholder("float", shape=[None, output_dim])

        for i in range(n_hidden):
            self.net = tfl.fully_connected(self.X, n_hidden_units, activation='relu')
            self.net = tfl.dropout(self.net, dropout_keep)

        self.prediction = tfl.fully_connected(self.net, output_dim, activation='linear')

        # Backward propagation
        self.loss = tf.reduce_mean(tf.squared_difference(self.prediction, self.Y))
        self.train = tf.train.GradientDescentOptimizer(0.001).minimize(self.loss)


    def fituntileps(self, dataprovider, batchsize, epsilon):

        sess = tf.Session()
        sess.run(tf.initialize_all_variables())

        ctr = 0
        while(True):

            ctr += 1

            # Obtain batch of data
            X, Y = dataprovider.getBatch(batchsize)

            # Train on batch
            sess.run(self.train, feed_dict={self.X : X, self.Y : Y})

            # Evaluate
            err = self.eval(dataprovider)

            if err < 0.1:
                break

        return ctr

    def eval(self, dataprovider):
        pass

class GradNet:
    pass

class QueryNet:
    pass

#tclass
