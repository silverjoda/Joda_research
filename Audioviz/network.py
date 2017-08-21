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




    def train(self, track):
        pass

    def predictframes(self, tracksample):
        pass





class VizAudioNet:
    def __init__(self):
        pass