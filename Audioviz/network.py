import tensorflow as tf
import tflearn as tfl
import numpy as np

# Ver 1: Just audio conv from last 2 frames
# Ver 2: Audio conv + conditioning on last frame
# Ver 3: Audio conv + recurrency in frame space (or something)

class AudioVizNetVer1:
    def __init__(self, sample_length):

        self.sample_length = sample_length

        self._makenet()
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

        l1_enc_f = tfl.conv_2d_transpose(incoming=l3_aud,
                                         nb_filter=16,
                                         filter_size=[1, 1],
                                         strides=[1,1,1,1],
                                         output_shape=[3, 3],
                                         padding='valid',
                                         activation='relu',
                                         name='l1_enc_f')

        l2_enc_f = tfl.conv_2d_transpose(incoming=l1_enc_f,
                                         nb_filter=16,
                                         filter_size=[2, 2],
                                         strides=[1, 1, 1, 1],
                                         output_shape=[6, 6],
                                         padding='valid',
                                         activation='relu',
                                         name='l2_enc_f')

        l3_enc_f = tfl.conv_2d_transpose(incoming=l2_enc_f,
                                         nb_filter=16,
                                         filter_size=[3, 3],
                                         strides=[1, 2, 2, 1],
                                         output_shape=[12, 12],
                                         padding='valid',
                                         activation='relu',
                                         name='l3_enc_f')

        l4_enc_f = tfl.conv_2d_transpose(incoming=l3_enc_f,
                                         nb_filter=16,
                                         filter_size=[3, 3],
                                         strides=[1, 3, 3, 1],
                                         output_shape=[36, 36],
                                         padding='valid',
                                         activation='relu',
                                         name='l4_enc_f')

        self.frame = tfl.conv_2d_transpose(incoming=l4_enc_f,
                                         nb_filter=16,
                                         filter_size=[3, 3],
                                         strides=[1, 3, 3, 1],
                                         output_shape=[128, 128],
                                         padding='valid',
                                         activation='relu',
                                         name='l5_enc_f')

        # Now turn frame back into audio

    def train(self, track):
        pass

    def predictframes(self, tracksample):
        pass





class VizAudioNet:
    def __init__(self):
        pass