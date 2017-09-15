import tensorflow as tf
import tflearn as tfl

class VAE:

    def __init__(self, in_dim, z_dim):
        self.in_dim = in_dim
        self.z_dim = z_dim

        self.g = tf.Graph()
        with self.g.as_default():
            self.lr_ph = tf.placeholder(tf.float32)
            self._encoder()
            self._decoder()
            self._objective()
            self._init_op = tf.global_variables_initializer()

        self.sess = tf.Session(self.g)
        self.sess.run(self._init_op)


    def _encoder(self):
        self.X = tf.placeholder(tf.float32, [None] + self.in_dim, 'X')
        w_init = tfl.initializations.xavier()
        l1 = tfl.conv_2d(self.X, 32, (3,3), (1,1), 'same', activation='relu', weights_init=w_init)
        l1 = tfl.max_pool_2d(l1, [2, 2], strides=2)
        l2 = tfl.conv_2d(l1, 32, (3, 3), (1, 1), 'same', activation='relu', weights_init=w_init)
        l2 = tfl.max_pool_2d(l2, [2, 2], strides=2)
        l3 = tfl.conv_2d(l2, 32, (3, 3), (1, 1), 'same', activation='relu', weights_init=w_init)
        l3 = tfl.max_pool_2d(l3, [2, 2], strides=2)
        flattened = tfl.flatten(l3, 'flattened')

        self.mu = tf.layers.dense(flattened, self.z_dim, w_init)
        self.log_sig = tf.layers.dense(flattened, self.z_dim, w_init)

        eps = tf.random_normal((-1, self.z_dim), name='epsilon')
        self.z = self.mu + tf.multiply(eps, tf.exp(self.log_sig))


    def _decoder(self):

        deconv_1 = tfl.conv_2d_transpose(self.z,
                                         nb_filter=128,
                                         filter_size=3,
                                         output_shape=(3,3),
                                         strides=1,
                                         padding='same')

        deconv_2 = tfl.conv_2d_transpose(deconv_1,
                                         nb_filter=64,
                                         filter_size=3,
                                         output_shape=(12, 12),
                                         strides=1,
                                         padding='same')

        deconv_3 = tfl.conv_2d_transpose(deconv_2,
                                         nb_filter=32,
                                         filter_size=3,
                                         output_shape=(28, 28),
                                         strides=1,
                                         padding='same')

        self.reconstruction = deconv_3


    def _objective(self):
        self.N = tf.contrib.distributions.Normal(tf.zeros((self.z_dim)),
                                                 tf.ones((self.z_dim)))
        self.Q_dist = tf.contrib.distributions.Normal(self.mu,
                                                      self.log_sig)

        self.latent_loss = tf.contrib.distributions.kl(self.Q_dist, self.N)
        self.reconstruction_loss = tfl.mean_square(self.X, self.reconstruction)
        self.optim = tf.train.AdamOptimizer(self.lr_ph).minimize(
            self.reconstruction_loss + self.latent_loss)

    def train(self, X):
        pass

    def embed(self, X):
        pass

    def sample_random(self, X):
        pass

    def sample(self, z):
        pass