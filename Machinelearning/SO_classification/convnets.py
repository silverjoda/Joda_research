import numpy as np
import tensorflow as tf
import tflearn

import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.normalization import local_response_normalization
from tflearn.layers.estimator import regression



def lettertonum(s):
    return int([str(ord(c)&31) for c in s][0]) - 1


class CharWiseConvnet:
    def __init__(self, m, n, n_classes):

        # Img dims
        self.m = m
        self.n = n

        self.n_classes = n_classes

        network = input_data(shape=[None, self.m, self.n, 1], name='input')
        network = conv_2d(network, 32, 3, activation='relu', regularizer="L2")
        network = local_response_normalization(network)
        network = conv_2d(network, 64, 3, activation='relu', regularizer="L2")
        network = max_pool_2d(network, 2)
        network = local_response_normalization(network)
        network = fully_connected(network, 128, activation='tanh')
        network = dropout(network, 0.8)
        network = fully_connected(network, 128, activation='tanh')
        network = dropout(network, 0.8)
        network = fully_connected(network, self.n_classes, activation='softmax')
        network = regression(network, optimizer='adam', learning_rate=0.001,
                             loss='categorical_crossentropy', name='target')

        # Training
        self.model = tflearn.DNN(network, tensorboard_verbose=0)

    def onehot(self, label):
        onehotlab = np.zeros(self.n_classes)
        onehotlab[label] = 1
        return onehotlab

    def fit(self, images, Y):

        img_list = []
        label_list = []

        for ims, labs in zip(images, Y):
            for i in range(len(labs[0])):

                c_img = ims[:, self.n*i:self.n*(i + 1)]
                c_lab = self.onehot(lettertonum(labs[0][i]))

                img_list.append(c_img)
                label_list.append(c_lab)





        # self.model.fit({'input': X}, {'target': Y}, n_epoch=20,
        #           validation_set=({'input': testX}, {'target': testY}),
        #           snapshot_step=100, show_metric=True, run_id='convnet_mnist')

    def predict(self, image):
        # Prepare batch and channel dimensions for tensorflow
        X = np.expand_dims(image, axis=0)
        X = np.expand_dims(X, axis=3)

        # Check size just in case
        assert X.shape == (1,16,8,1)

        return self.model.predict(X)
