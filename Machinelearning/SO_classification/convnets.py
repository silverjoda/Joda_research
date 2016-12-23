import numpy as np
import tensorflow as tf
import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
from tflearn.layers.normalization import local_response_normalization


def lettertonum(s):
    return int([str(ord(c)&31) for c in s][0]) - 1


class CharWiseConvnet:
    def __init__(self, m, n, n_classes):

        # Img dims
        self.m = m
        self.n = n

        self.n_classes = n_classes

        w_init = tflearn.initializations.xavier(uniform=True, seed=None,
                                       dtype=tf.float32)

        network = input_data(shape=[None, self.m, self.n, 1], name='input')
        network = conv_2d(network, 32, 3, activation='relu',
                          regularizer="L2", weights_init=w_init)
        network = local_response_normalization(network)
        network = conv_2d(network, 64, 3, activation='relu', regularizer="L2", weights_init=w_init)
        network = max_pool_2d(network, 2)
        network = local_response_normalization(network)
        self.conv_feats = tflearn.reshape(network, new_shape=[64*8*4])
        network = fully_connected(network, 128, activation='relu')
        network = dropout(network, 0.8)
        network = fully_connected(network, 128, activation='relu')
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

    def makedata(self, images, labels):
        img_list = []
        label_list = []

        for ims, labs in zip(images, labels):
            for i in range(len(labs[0])):
                c_img = ims[:, self.n * i:self.n * (i + 1)]
                c_lab = self.onehot(lettertonum(labs[0][i]))

                img_list.append(np.expand_dims(c_img, 2))
                label_list.append(c_lab)

        X = np.array(img_list)
        Y = np.array(label_list)

        return X, Y

    def fit(self, images_trn, Y_trn, images_tst, Y_tst):

        # Turn data into proper format
        self.X_trn, self.Y_trn = self.makedata(images_trn, Y_trn)
        self.X_tst, self.Y_tst = self.makedata(images_tst, Y_tst)

        self.model.fit({'input': self.X_trn}, {'target': self.Y_trn},n_epoch=15,
                  validation_set=({'input': self.X_tst}, {'target': self.Y_tst}),
                  snapshot_step=100, show_metric=True, run_id='convnet_mnist')

        self.model.save("convnet.tfl")


    def predict(self, image):
        # Prepare batch and channel dimensions for tensorflow
        X = np.expand_dims(image, axis=0)
        X = np.expand_dims(X, axis=3)

        # Check size just in case
        assert X.shape == (1,16,8,1)

        return np.argmax(self.model.predict(X))

    def evaluate(self, X, Y):
        n_sequences = len(X)
        n_examples = 0
        n_chars_wrong = 0
        n_seq_wrong = 0

        # Go over all examples here and evaluate
        for ims, labs in zip(X, Y):

            seq_marked = False

            for i in range(len(labs[0])):

                n_examples += 1

                c_img = ims[:, self.n * i:self.n * (i + 1)]
                c_lab = lettertonum(labs[0][i])

                prediction = self.predict(c_img)

                if prediction != c_lab:
                    n_chars_wrong += 1
                    if not seq_marked:
                        seq_marked = True
                        n_seq_wrong += 1

        return [(n_sequences - n_seq_wrong) / float(n_sequences)
            , (n_examples - n_chars_wrong) / float(n_examples)]

    def makedataset(self, X_trn, Y_trn, X_tst, Y_tst):

        # Go over all examples here and evaluate
        for ims, labs in zip(X_trn, Y_trn):
            for i in range(len(labs[0])):

                c_img = ims[:, self.n * i:self.n * (i + 1)]
                c_img = np.expand_dims(c_img)
                c_lab = lettertonum(labs[0][i])

                features = self.model.session.run(self.conv_feats, feed_dict
                = {input_data : c_img })


