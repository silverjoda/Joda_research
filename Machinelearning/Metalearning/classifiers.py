import tensorflow as tf
import tflearn as tfl
import numpy as np

class Approximator:
    def __init__(self, input_dim, output_dim, n_hidden, n_units, l2_decay, dropout_keep):

        self.net = tfl.input_data(shape=[None, input_dim])

        for i in range(n_hidden):
            self.net = tfl.fully_connected(self.net, n_units, activation='relu')
            self.net = tfl.dropout(self.net, dropout_keep)

        self.net = tfl.fully_connected(self.net, output_dim, activation='linear')

    def fit(self, X, Y, batchsize):
        pass

class GradNet:
    pass

class QueryNet:
    pass

#tclass
