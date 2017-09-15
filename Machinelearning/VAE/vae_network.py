import tensorflow as tf
import tflearn as tfl

class VAE:

    def __init__(self, in_dim, z_dim):
        self.in_dim = in_dim
        self.z_dim = z_dim

        self.g = tf.Graph()
        with self.g.as_default():
            self.lr_ph = tf.placeholder(tf.float32)
            self.z_rnd_ph = tf.placeholder(tf.float32, shape=(-1, self.z_dim))
            self._encoder()
            self.trn_reconstruction = self._decoder(self.X, reuse=False)
            self.tst_reconstruction = self._decoder(self.z_rnd_ph, reuse=True)
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


    def _decoder(self, z, reuse=False):
        with tf.variable_scope("decoder", reuse=reuse):
            deconv_1 = tfl.conv_2d_transpose(z,
                                             nb_filter=128,
                                             filter_size=3,
                                             output_shape=(3,3),
                                             strides=1,
                                             padding='same',
                                             activation='relu')

            deconv_2 = tfl.conv_2d_transpose(deconv_1,
                                             nb_filter=64,
                                             filter_size=3,
                                             output_shape=(12, 12),
                                             strides=1,
                                             padding='same',
                                             activation='relu')

            deconv_3 = tfl.conv_2d_transpose(deconv_2,
                                             nb_filter=32,
                                             filter_size=3,
                                             output_shape=(28, 28),
                                             strides=1,
                                             padding='same',
                                             activation='relu')

        return deconv_3


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
        fetches = [self.optim, self.reconstruction_loss, self.latent_loss]
        _, mse, kl_loss = self.sess.run(fetches, feed_dict={self.X : X})

        return mse, kl_loss


    def embed(self, X):
        return self.sess.run(self.z, feed_dict={self.X : X})

    def sample(self, z):
        return self.sess.run(self.tst_reconstruction,
                             feed_dict={self.z_rnd_ph : z})