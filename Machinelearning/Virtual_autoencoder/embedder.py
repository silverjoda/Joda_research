import tensorflow as tf
import tflearn

class Autoencoder:
    def __init__(self, i_dim, e_dim):

        self.i_dim = i_dim
        self.e_dim = e_dim

        self.g = tf.Graph()

        with self.g.as_default():

            self.create_embedder_network()

            self.learning_rate = 0.0005

            # Loss function
            self.cost = tflearn.mean_square(self.X, self.reconstruction)

            # Optimization Op
            self.optimize = tf.train.AdamOptimizer(self.learning_rate).minimize(
                self.cost)

            self.init = tf.global_variables_initializer()

        config = tf.ConfigProto(
            device_count={'GPU': 0}
        )

        self.sess = tf.Session(graph=self.g, config=config)
        self.sess.run(self.init)

    def create_embedder_network(self):

        # Input image
        self.X = tflearn.input_data(shape=[None,
                                        self.i_dim[0],
                                        self.i_dim[1],
                                        1])

        # Xavier initializer
        conv_init = tflearn.initializations.xavier()

        # First convolutional layer
        conv1 = tflearn.conv_2d(self.X, 16, 3, regularizer="L2",
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

        self.embedding = conv3

        # Bottle neck layer
        bottleneck = tflearn.dropout(conv3, keep_prob=0.7)

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

        self.reconstruction = conv_up_3


    def embed(self, img):
        return self.sess.run(self.embedding, feed_dict={self.X: img})


    def train(self, imgs):
        err, _ = self.sess.run([self.cost, self.optimize],
                                feed_dict={self.X : imgs})

        return err

    def reconstruct(self, img):
        return self.sess.run(self.reconstruction, feed_dict={self.X : img})

class Vencoder:
    def __init__(self, i_dim, e_dim):

        self.i_dim = i_dim
        self.e_dim = e_dim

        self.g = tf.Graph()

        with self.g.as_default():

            self.create_embedder_network()

            self.learning_rate = 0.0005

            # Loss function
            self.cost = tflearn.mean_square(self.X, self.reconstruction)

            # Optimization Op
            self.optimize = tf.train.AdamOptimizer(self.learning_rate).minimize(
                self.cost)

            self.init = tf.global_variables_initializer()

        config = tf.ConfigProto(
            device_count={'GPU': 0}
        )

        self.sess = tf.Session(graph=self.g, config=config)
        self.sess.run(self.init)

    def create_embedder_network(self):

        # Input image
        self.X = tflearn.input_data(shape=[None,
                                        self.i_dim[0],
                                        self.i_dim[1],
                                        1])

        # Xavier initializer
        conv_init = tflearn.initializations.xavier()

        # First convolutional layer
        conv1 = tflearn.conv_2d(self.X, 16, 3, regularizer="L2",
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
        self.embedding = conv3


        self.reconstruction = tf.gradients(ys=self.embedding,
                                           xs=self.X)


    def embed(self, img):
        return self.sess.run(self.embedding, feed_dict={self.X: img})


    def train(self, imgs):
        err, _ = self.sess.run([self.cost, self.optimize],
                                feed_dict={self.X : imgs})

        return err

    def reconstruct(self, img):
        return self.sess.run(self.reconstruction, feed_dict={self.X : img})