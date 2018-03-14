import tensorflow as tf
import tflearn as tfl
import numpy as np

# Ver 1: Just audio conv from last 2 frames
# Ver 2: Audio conv + conditioning on last frame
# Ver 3: Audio conv + recurrency in frame space (or something)

class AudioVizNetVer1:
    def __init__(self, sample_length):

        self.sample_length = sample_length
        self.g = tf.get_default_graph()

        self.datain, self.out = self._makenet()
        self.loss = tfl.mean_square(self.datain, self.out)
        self.optim = tf.train.AdamOptimizer(1e-4).minimize(self.loss)

        self.sess = tf.Session()


    def _makenet(self):
        self.audio_in = tfl.input_data(shape=(None, self.sample_length),
                                       name='audio_in')

        l1_aud = tfl.conv_1d(self.audio_in,
                           nb_filter=16,
                           filter_size=5,
                           strides=3,
                           padding='valid',
                           activation='relu',
                           regularizer='L2',
                           name='l1_aud')

        l2_aud = tfl.conv_1d(l1_aud,
                           nb_filter=16,
                           filter_size=5,
                           strides=3,
                           padding='valid',
                           activation='relu',
                           regularizer='L2',
                           name='l2_aud')

        l3_aud = tfl.conv_1d(l2_aud,
                             nb_filter=16,
                             filter_size=5,
                             strides=3,
                             padding='valid',
                             activation='relu',
                             regularizer='L2',
                             name='l3_aud')


        l1_enc_f = tf.layers.conv2d_transpose(inputs=l3_aud,
                                              filters=16,
                                              kernel_size=[3, 3],
                                              strides=(1, 1),
                                              padding='valid',
                                              activation='relu',
                                              name='l1_enc_f')

        l2_enc_f = tf.layers.conv2d_transpose(inputs=l1_enc_f,
                                              filters=16,
                                              kernel_size=[3, 3],
                                              strides=(2, 2),
                                              padding='valid',
                                              activation='relu',
                                              name='l2_enc_f')

        l3_enc_f = tf.layers.conv2d_transpose(inputs=l2_enc_f,
                                              filters=16,
                                              kernel_size=[3, 3],
                                              strides=(2, 2),
                                              padding='valid',
                                              activation='relu',
                                              name='l3_enc_f')

        l4_enc_f = tf.layers.conv2d_transpose(inputs=l3_enc_f,
                                              filters=16,
                                              kernel_size=[3, 3],
                                              strides=(2, 2),
                                              padding='valid',
                                              activation='relu',
                                              name='l4_enc_f')

        self.frame = l4_enc_f

        # Now turn frame back into audio
        l1_dec = tfl.conv_2d(l4_enc_f, 16, (3,3), (2,2), padding='valid', activation='relu')
        l2_dec = tfl.conv_2d(l1_dec, 16, (3, 3), (2, 2), padding='valid', activation='relu')
        l3_dec = tfl.conv_2d(l2_dec, 16, (3, 3), (2, 2), padding='valid', activation='relu')
        l4_dec = tfl.conv_2d(l3_dec, 8, (3, 3), (2, 2), padding='valid', activation='relu')

        conv_flat = tfl.flatten(l4_dec)
        conv_flat = tf.expand_dims(conv_flat, 1)
        conv_flat = tf.expand_dims(conv_flat, 3)

        l1_aud_dec = tf.layers.conv2d_transpose(l4_dec, 16, (1,3), (1,3), activation='relu')
        l2_aud_dec = tf.layers.conv2d_transpose(l4_dec, 16, (1, 3), (1, 3), activation='relu')
        l3_aud_dec = tf.layers.conv2d_transpose(l4_dec, 16, (1, 3), (1, 3), activation='relu')
        l4_aud_dec = tf.layers.conv2d_transpose(l4_dec, 16, (1, 3), (1, 3), activation='relu')

        return l4_aud_dec


    def train(self, track):
        pass

    def predictframes(self, tracksample):
        pass





class VizAudioNet:
    def __init__(self):
        pass