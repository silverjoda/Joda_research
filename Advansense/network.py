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

        # Make session using given configuration
        self.sess = tf.Session(graph=self.g, config=tfconfig)
        self.sess.run(self.init)


    def make_encoder(self, input):
        # Include extra dimension for convolutions
        exp_input = tf.expand_dims(input, 3)

        conv_l1 = tfl.conv_2d(exp_input, 16, 5, 3, 'valid', 'relu', True, 'xavier')
        conv_l2 = tfl.conv_2d(conv_l1, 16, 5, 3, 'valid', 'relu', True, 'xavier')
        conv_l3 = tfl.conv_2d(conv_l2, 16, 3, 2, 'valid', 'relu', True, 'xavier')
        conv_l4 = tfl.conv_2d(conv_l3, 16, 3, 2, 'valid', 'relu', True, 'xavier')

        flattened = tfl.flatten(conv_l4)
        fc = tfl.fully_connected(flattened, 200, 'relu', True, 'xavier')
        fc_conv = tf.reshape(fc, (-1, 1, 200, 1))

        audio_deconv_l1 = tf.layers.conv2d_transpose(inputs=fc_conv,
                                                     filters=16,
                                                     kernel_size=(1, 3),
                                                     strides=(1, 2),
                                                     padding='valid',
                                                     activation=tf.nn.relu,
                                                     kernel_initializer=tf.contrib.layers.xavier_initializer())

        audio_deconv_l2 = tf.layers.conv2d_transpose(inputs=audio_deconv_l1,
                                                     filters=16,
                                                     kernel_size=(1, 3),
                                                     strides=(1, 2),
                                                     padding='valid',
                                                     activation=tf.nn.relu,
                                                     kernel_initializer=tf.contrib.layers.xavier_initializer())

        audio_deconv_l3 = tf.layers.conv2d_transpose(inputs=audio_deconv_l2,
                                                     filters=16,
                                                     kernel_size=(1, 5),
                                                     strides=(1, 3),
                                                     padding='valid',
                                                     activation=tf.nn.relu,
                                                     kernel_initializer=tf.contrib.layers.xavier_initializer())

        audio_deconv_l4 = tf.layers.conv2d_transpose(inputs=audio_deconv_l3,
                                                     filters=1,
                                                     kernel_size=(1, 5),
                                                     strides=(1, 3),
                                                     padding='valid',
                                                     activation=tf.nn.relu,
                                                     kernel_initializer=tf.contrib.layers.xavier_initializer())

        squeezed_output = tf.squeeze(audio_deconv_l4, squeeze_dims=(1,3))

        return squeezed_output


    def make_decoder(self, input):
        exp_input = tf.expand_dims(input, 2)
        audio_conv_l1 = tfl.conv_1d(exp_input, 16, 7, 5, 'valid', 'relu', True, 'xavier')
        audio_conv_l2 = tfl.conv_1d(audio_conv_l1, 16, 7, 5, 'valid', 'relu', True, 'xavier')
        audio_conv_l3 = tfl.conv_1d(audio_conv_l2, 16, 5, 3, 'valid', 'relu', True, 'xavier')
        audio_conv_l4 = tfl.conv_1d(audio_conv_l3, 16, 5, 3, 'valid', 'relu', True, 'xavier')

        flattened = tfl.flatten(audio_conv_l4)
        n_units = flattened.get_shape()[1]
        reshaped = tf.reshape(audio_conv_l4, (-1, 1, 1, n_units))

        deconv_l1 = tf.layers.conv2d_transpose(inputs=reshaped,
                                                     filters=16,
                                                     kernel_size=(3, 3),
                                                     strides=(1, 1),
                                                     padding='valid',
                                                     activation=tf.nn.relu,
                                                     kernel_initializer=tf.contrib.layers.xavier_initializer())

        deconv_l2 = tf.layers.conv2d_transpose(inputs=deconv_l1,
                                               filters=16,
                                               kernel_size=(3, 3),
                                               strides=(2, 2),
                                               padding='valid',
                                               activation=tf.nn.relu,
                                               kernel_initializer=tf.contrib.layers.xavier_initializer())

        deconv_l3 = tf.layers.conv2d_transpose(inputs=deconv_l2,
                                               filters=16,
                                               kernel_size=(3, 3),
                                               strides=(2, 2),
                                               padding='valid',
                                               activation=tf.nn.relu,
                                               kernel_initializer=tf.contrib.layers.xavier_initializer())

        deconv_l4 = tf.layers.conv2d_transpose(inputs=deconv_l3,
                                               filters=8,
                                               kernel_size=(5, 5),
                                               strides=(3, 3),
                                               padding='valid',
                                               activation=tf.nn.relu,
                                               kernel_initializer=tf.contrib.layers.xavier_initializer())

        deconv_l5 = tf.layers.conv2d_transpose(inputs=deconv_l4,
                                               filters=1,
                                               kernel_size=(5, 5),
                                               strides=(3, 3),
                                               padding='valid',
                                               activation=tf.nn.relu,
                                               kernel_initializer=tf.contrib.layers.xavier_initializer())

        resized = tf.image.resize_bilinear(deconv_l5, (self.res, self.res))
        return tf.squeeze(resized, 3)


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