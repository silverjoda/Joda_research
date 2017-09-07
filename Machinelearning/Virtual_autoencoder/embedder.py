import tensorflow as tf
import tflearn

class Autoencoder:
    def __init__(self, sess, i_dim, e_dim):
        self.sess = sess
        self.i_dim = i_dim
        self.e_dim = e_dim
        self.img, self.embedding, self.reconstruction = \
            self.create_embedder_network()

        self.learning_rate = 0.0001

        # Loss function
        self.cost = tflearn.mean_square(self.img, self.reconstruction)

        # Optimization Op
        self.optimize = tf.train.AdamOptimizer(self.learning_rate).minimize(
            self.cost)

    def create_embedder_network(self):

        # Input depth image
        img = tflearn.input_data(shape=[None,
                                        self.i_dim[0],
                                        self.i_dim[1],
                                        1])

        # Xavier initializer
        conv_init = tflearn.initializations.xavier()

        # First convolutional layer
        conv1 = tflearn.conv_2d(img, 16, 3, regularizer="L2",
                                activation='relu', weights_init=conv_init,
                                strides=(1, 2, 2, 1))

        # Second convolutional layer
        conv2 = tflearn.conv_2d(conv1, 16, 3, regularizer="L2",
                                activation='relu', weights_init=conv_init,
                                strides=(1, 2, 2, 1))

        # third convolutional layer
        conv3 = tflearn.conv_2d(conv2, 1, 3, regularizer="L2",
                                activation='relu', weights_init=conv_init,
                                strides=(1, 2, 2, 1))

        # Bottle neck layer
        bottleneck = tflearn.dropout(conv3, keep_prob=0.7)

        embedding = tflearn.flatten(bottleneck)

        conv_up_1 = tflearn.layers.conv.conv_2d_transpose(incoming=bottleneck,
          nb_filter=16, filter_size=3, output_shape=[self.i_dim[0]/4,
                                                     self.i_dim[1]/4],
                                                          strides=(1, 2, 2, 1),
                                                          activation='relu')


        conv_up_2 = tflearn.layers.conv.conv_2d_transpose(incoming=conv_up_1,
                                                          nb_filter=16,
                                                          filter_size=3,
                                                          output_shape=[
                                                          self.i_dim[0] / 2,
                                                          self.i_dim[1] / 2],
                                                          strides=(1, 2, 2, 1),
                                                          activation='relu')


        conv_up_3 = tflearn.layers.conv.conv_2d_transpose(incoming=conv_up_2,
                                                          nb_filter=1,
                                                          filter_size=3,
                                                          output_shape=[
                                                              self.i_dim[0],
                                                              self.i_dim[1]],
                                                          strides=(1, 2, 2, 1),
                                                          activation='relu')


        return img, embedding, conv_up_3


    def embed(self, img):
        return self.sess.run(self.embedding, feed_dict={self.img: img})

    def train(self, img):
        err, _, emb = self.sess.run([self.cost, self.optimize, self.embedding],
                                feed_dict={
            self.img:img})

        return err, emb

    def reconstruct(self, img):
        return self.sess.run(self.reconstruction, feed_dict={self.img: img})

class Vencoder:
    pass