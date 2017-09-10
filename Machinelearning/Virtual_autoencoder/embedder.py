import tensorflow as tf
import tflearn as tfl

class Autoencoder:
    def __init__(self, i_dim, e_dim):

        self.i_dim = i_dim
        self.e_dim = e_dim

        self.g = tf.Graph()

        with self.g.as_default():

            self.create_embedder_network()

            self.learning_rate = 5e-4

            # Loss function
            self.cost = tfl.mean_square(self.X, self.reconstruction)

            # Optimization Op
            self.optimize = tf.train.AdamOptimizer(self.learning_rate).minimize(
                self.cost)

            self.init = tf.global_variables_initializer()

        config = tf.ConfigProto(
            device_count={'GPU': 1}
        )

        self.sess = tf.Session(graph=self.g, config=config)
        self.sess.run(self.init)

    def create_embedder_network(self):

        # Input image
        self.X = tfl.input_data(shape=[None,
                                        self.i_dim[0],
                                        self.i_dim[1],
                                        1])

        # Xavier initializer
        conv_init = tfl.initializations.xavier()

        # First convolutional layer
        conv1 = tfl.conv_2d(self.X, 32, 3, regularizer="L2",
                                activation='relu', weights_init=conv_init,
                                strides=(1, 1, 1, 1), padding='same')
        conv1 = tfl.max_pool_2d(conv1,
                                    kernel_size=(2, 2),
                                    strides=(1, 2, 2, 1))

        # Second convolutional layer
        conv2 = tfl.conv_2d(conv1, 32, 3, regularizer="L2",
                                activation='relu', weights_init=conv_init,
                                strides=(1, 1, 1, 1), padding='same')
        conv2 = tfl.max_pool_2d(conv2,
                                    kernel_size=(2, 2),
                                    strides=(1, 2, 2, 1))

        # third convolutional layer
        conv3 = tfl.conv_2d(conv2, 3, 3, regularizer="L2",
                                activation='relu', weights_init=conv_init,
                                strides=(1, 1, 1, 1), padding='same')
        conv3 = tfl.max_pool_2d(conv3,
                                    kernel_size=(2, 2),
                                    strides=(1, 2, 2, 1))

        self.embedding = conv3


        conv_up_1 = tfl.conv_2d_transpose(incoming=conv3,
                                          nb_filter=32,
                                          filter_size=3,
                                          output_shape=[int(self.i_dim[0]/4),
                                                        int(self.i_dim[1]/4)],
                                          strides=(1,2,2,1),
                                          activation='tanh',
                                          weights_init=conv_init)


        conv_up_2 = tfl.conv_2d_transpose(incoming=conv_up_1,
                                                          nb_filter=32,
                                                          filter_size=3,
                                                          output_shape=[
                                                          int(self.i_dim[0] / 2),
                                                          int(self.i_dim[1] / 2)],
                                                          strides=(1, 2, 2, 1),
                                                          activation='tanh',
                                                          weights_init=conv_init)


        conv_up_3 = tfl.conv_2d_transpose(incoming=conv_up_2,
                                                          nb_filter=1,
                                                          filter_size=3,
                                                          output_shape=[
                                                              int(self.i_dim[0]),
                                                              int(self.i_dim[1])],
                                                          strides=(1, 2, 2, 1),
                                                          activation='tanh',
                                                          weights_init=conv_init)

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

            self.learning_rate = 5e-4

            # Loss function
            self.cost = tfl.mean_square(self.X, self.reconstruction)
            self.upconv_cost = tfl.mean_square(self.X, self.upconv_reconstruction)

            # Optimization Op
            self.optimize = tf.train.AdamOptimizer(self.learning_rate).minimize(
                self.cost)

            self.optimize_upconv = tf.train.AdamOptimizer(self.learning_rate).minimize(
                self.upconv_cost, var_list=self.upconv_vars)

            self.init = tf.global_variables_initializer()

        config = tf.ConfigProto(
            device_count={'GPU': 1}
        )

        self.sess = tf.Session(graph=self.g, config=config)
        self.sess.run(self.init)

    def create_embedder_network(self):

        with tf.variable_scope("encoder"):

            # Input image
            self.X = tfl.input_data(shape=[None,
                                            self.i_dim[0],
                                            self.i_dim[1],
                                            1])

            # Xavier initializer
            conv_init = tfl.initializations.xavier()

            # First convolutional layer
            conv1 = tfl.conv_2d(self.X, 32, 3, regularizer="L2",
                                    activation='relu', weights_init=conv_init,
                                    strides=(1, 1, 1, 1), padding='valid')
            conv1 = tfl.max_pool_2d(conv1,
                                        kernel_size=(2,2),
                                        strides=(1,2,2,1))

            # Second convolutional layer
            conv2 = tfl.conv_2d(conv1, 32, 3, regularizer="L2",
                                    activation='relu', weights_init=conv_init,
                                    strides=(1, 1, 1, 1), padding='same')
            conv2 = tfl.max_pool_2d(conv2,
                                        kernel_size=(2, 2),
                                        strides=(1, 2, 2, 1))


            # third convolutional layer
            conv3 = tfl.conv_2d(conv2, 3, 3, regularizer="L2",
                                    activation='relu', weights_init=conv_init,
                                    strides=(1, 1, 1, 1), padding='same')
            conv3 = tfl.max_pool_2d(conv3,
                                        kernel_size=(2, 2),
                                        strides=(1, 2, 2, 1))

        with tf.variable_scope("decoder"):
            conv_up_1 = tfl.conv_2d_transpose(incoming=conv3,
                                              nb_filter=32,
                                              filter_size=3,
                                              output_shape=[
                                                  int(self.i_dim[0] / 4),
                                                  int(self.i_dim[1] / 4)],
                                              strides=(1, 2, 2, 1),
                                              activation='tanh',
                                              weights_init=conv_init)

            conv_up_2 = tfl.conv_2d_transpose(incoming=conv_up_1,
                                              nb_filter=32,
                                              filter_size=3,
                                              output_shape=[
                                                  int(self.i_dim[0] / 2),
                                                  int(self.i_dim[1] / 2)],
                                              strides=(1, 2, 2, 1),
                                              activation='tanh',
                                              weights_init=conv_init)

            conv_up_3 = tfl.conv_2d_transpose(incoming=conv_up_2,
                                              nb_filter=1,
                                              filter_size=3,
                                              output_shape=[
                                                  int(self.i_dim[0]),
                                                  int(self.i_dim[1])],
                                              strides=(1, 2, 2, 1),
                                              activation='tanh',
                                              weights_init=conv_init)

        self.upconv_vars = tf.get_collection(tf.GraphKeys.VARIABLES, scope='decoder')

        # Upconv decoder
        self.upconv_reconstruction = conv_up_3

        # Bottle neck layer
        self.embedding = conv3

        self.reconstruction = tf.gradients(ys=self.embedding, xs=self.X)


    def embed(self, img):
        return self.sess.run(self.embedding, feed_dict={self.X: img})


    def train(self, imgs):
        err, _ = self.sess.run([self.cost, self.optimize],
                                feed_dict={self.X : imgs})

        return err

    def train_upconv_decoder(self, imgs):
        err, _ = self.sess.run([self.upconv_cost, self.optimize_upconv],
                               feed_dict={self.X: imgs})

        return err


    def reconstruct(self, img):
        return self.sess.run(self.reconstruction, feed_dict={self.X : img})

    def upconv_reconstruct(self, img):
        return self.sess.run(self.upconv_reconstruction, feed_dict={self.X: img})