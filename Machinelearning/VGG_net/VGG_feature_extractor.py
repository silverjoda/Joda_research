"""
Class of the vgg network architecture which transforms an image into the last
hidden vactor of size 4096
"""

import tensorflow as tf
import numpy as np
import os


IMAGENET_MEAN = [103.939, 116.779, 123.68]

class VGG_feature_extractor:

    def __init__(self, model_path):

        # Path to vgg weights
        self.model_path = model_path

        # VGG weights and biases
        self.params = self._make_vgg_weights()

        # Input placeholder
        self.X = tf.placeholder(shape=(224, 224, 3), dtype=tf.float32)

        # Create tensorflow session
        self.session = tf.Session()
        self.session.run(tf.initialize_all_variables())

    def _make_vgg_weights(self):
        """
        Create the necessary VGG weights for the models
        -------

        """

        # Init empty dictionaries for weights and biases
        weights = {}
        biases = {}

        for i in xrange(16):
            weights['w{}'.format(i + 1)] = tf.Variable(
                np.load(os.path.join(self.model_path, 'w{}.npy'.format(i + 1))))

            biases['b{}'.format(i + 1)] = tf.Variable(
                np.load(os.path.join(self.model_path, 'b{}.npy'.format(i + 1))))

        return weights, biases

    def _VGG_forward_pass(self):
        """
        Forward pass of the VGG network
        Parameters
        ----------
        X: tf tensor, input image with dims (224,224,3)

        Returns tf tensor of the 4096 long feature vector
        -------

        """

        weights, biases = self.params

        conv_strides = [1, 1, 1, 1]

        # Block 1
        l1 = tf.nn.relu(
            tf.nn.conv2d(self.X, weights['w1'], conv_strides, 'SAME') +
            biases['b1'])

        l2 = tf.nn.relu(
            tf.nn.conv2d(l1, weights['w2'], conv_strides, 'SAME') + biases[
                'b2'])

        l2 = tf.nn.max_pool(l2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                            padding='SAME')

        # Block 2
        l3 = tf.nn.relu(
            tf.nn.conv2d(l2, weights['w3'], conv_strides, 'SAME') + biases[
                'b3'])

        l4 = tf.nn.relu(
            tf.nn.conv2d(l3, weights['w4'], conv_strides, 'SAME') + biases[
                'b4'])

        l4 = tf.nn.max_pool(l4, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                            padding='SAME')

        # Block 3
        l5 = tf.nn.relu(
            tf.nn.conv2d(l4, weights['w5'], conv_strides, 'SAME') + biases[
                'b5'])


        l6 = tf.nn.relu(
            tf.nn.conv2d(l5, weights['w6'], conv_strides, 'SAME') + biases[
                'b6'])

        l7 = tf.nn.relu(
            tf.nn.conv2d(l6, weights['w7'], conv_strides, 'SAME') + biases[
                'b7'])

        l7 = tf.nn.max_pool(l7, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                              padding='SAME')

        # Block 4
        l8 = tf.nn.relu(
            tf.nn.conv2d(l7, weights['w8'], conv_strides, 'SAME') +
            biases['b8'])

        l9 = tf.nn.relu(
            tf.nn.conv2d(l8, weights['w9'], conv_strides, 'SAME') + biases[
                'b9'])

        l10 = tf.nn.relu(
            tf.nn.conv2d(l9, weights['w10'], conv_strides, 'SAME') +
            biases['b10'])

        l10 = tf.nn.max_pool(l10, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                               padding='SAME')

        # Block 5
        l11 = tf.nn.relu(
            tf.nn.conv2d(l10, weights['w11'], conv_strides, 'SAME') +
            biases['b11'])


        l12 = tf.nn.relu(
            tf.nn.conv2d(l11, weights['w12'], conv_strides, 'SAME') +
            biases['b12'])

        l13 = tf.nn.relu(
            tf.nn.conv2d(l12, weights['w13'], conv_strides, 'SAME') +
            biases['b13'])

        l13 = tf.nn.max_pool(l13, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                             padding='SAME')

        # 7x7 conv across 7x7 input to give a vector size 4096.
        # First hidden layer
        l_fc1 = tf.nn.relu(
            tf.nn.conv2d(l13, weights['w14'], conv_strides, 'SAME') +
            biases['b14'])

        l_fc2 = tf.nn.relu(
            tf.nn.conv2d(l_fc1, weights['w15'], conv_strides, 'SAME') +
            biases['b15'])

        return l_fc1, l_fc2

    def VGG_extract_features(self, input):
        """
        Extracts features from input image
        Parameters
        ----------
        input: float32 ndarray, image

        Returns
        -------

        """

        if input.shape() != (224,224,3):
            raise ValueError, "Image shape should be (224,224,3)"

        return self.session.run(input, feed_dict = {self.X : input})

