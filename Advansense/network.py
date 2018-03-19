import tensorflow as tf
import tflearn as tfl
import numpy as np
import os

class VizEncoder:
    def __init__(self, name, res, rate):

        self.name = name
        self.res = res
        self.rate = rate
        self.weights_path = 'models/{}'.format(name)
        if not os.path.exists(self.weights_path):
            os.makedirs(self.weights_path)

        self.g = tf.Graph()
        with self.g.as_default():
            with tf.name_scope("Placeholders"):
                self.obs_ph = tf.placeholder(dtype=tf.float32,
                                             shape=(None, self.res, self.res),
                                             name='frame_ph')

            # Encoder
            self.encoded = self.make_encoder(self.obs_ph)

            # Decoder
            self.decoded = self.make_decoder(self.encoded)

            # Loss function
            self.mse = tfl.mean_square(self.decoded, self.obs_ph)

            # Optimization function
            self.optim = tf.train.AdamOptimizer(1e-4).minimize(self.mse)

            # Initialization function
            self.init = tf.global_variables_initializer()

        tfconfig = tf.ConfigProto(
            device_count={'GPU': 0}
        )

        # Make sessio using given configuration
        self.sess = tf.Session(graph=self.g, config=tfconfig)
        self.sess.run(self.init)


    def make_encoder(self, input):
        pass

    def make_decoder(self, input):
        pass

    def train(self, data):
        '''
        Train network on data
        Parameters
        ----------
        data: numpy array of frames

        Returns: training mse
        -------

        '''

        # operations
        fetches = [self.optim, self.mse]

        # Data feed
        fd = {self.obs_ph : data}

        # Running operation
        _, mse = self.sess.run(fetches, fd)

        return mse


    def encode(self, data):
        '''
        Encode frames into audio array
        Parameters
        ----------
        data: np array of frames

        Returns: np array of audio normalized to +-1
        -------

        '''
        return self.sess.run(self.encoded, {self.obs_ph : data})


    def save_weights(self):
        with self.g.as_default():
            saver = tf.train.Saver()
            saver.save(self.sess, self.weights_path + "/trained_model")

    def restore_weights(self):
        with self.g.as_default():
            saver = tf.train.Saver()
            saver.restore(self.sess, tf.train.latest_checkpoint(self.weights_path))