import tensorflow as tf
import tflearn as tfl

class VAE:

    def __init__(self, in_dim, z_dim, lr):
        self.in_dim = in_dim
        self.z_dim = z_dim
        self.lr = lr

        self.g = tf.Graph()
        with self.g.as_default():
            self.lr_ph = tf.placeholder(tf.float32)
            self.z_rnd_ph = tf.placeholder(tf.float32, shape=(None, self.z_dim))
            self._encoder()
            self.trn_recon = self._decoder(self.z, reuse=False)
            self.tst_recon = self._decoder(self.z_rnd_ph, reuse=True)
            self._objective()
            self._init_op = tf.global_variables_initializer()

        self.sess = tf.Session(graph=self.g)
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

        self.mu = tfl.fully_connected(flattened, self.z_dim, weights_init=w_init)
        self.log_sig = tfl.fully_connected(flattened, self.z_dim, weights_init=w_init)

        eps = tf.random_normal(tf.shape(self.mu), name='epsilon')
        self.z = self.mu + tf.multiply(eps, tf.exp(self.log_sig / 2))


    def _decoder(self, z, reuse=False):
        flat_conv = tf.reshape(z, (-1, 1, 1, self.z_dim))
        with tf.variable_scope("decoder", reuse=reuse):
            w_init = tf.contrib.layers.xavier_initializer()
            deconv_1 = tf.layers.conv2d_transpose(inputs=flat_conv,
                                                  filters=128,
                                                  kernel_size=[3,3],
                                                  strides=(1, 1),
                                                  padding='valid',
                                                  activation=tf.nn.relu,
                                                  kernel_initializer=w_init)

            deconv_2 = tf.layers.conv2d_transpose(inputs=deconv_1,
                                                  filters=64,
                                                  kernel_size=[3, 3],
                                                  strides=(2, 2),
                                                  padding='valid',
                                                  activation=tf.nn.relu,
                                                  kernel_initializer=w_init)

            deconv_3 = tf.layers.conv2d_transpose(inputs=deconv_2,
                                                  filters=32,
                                                  kernel_size=[3, 3],
                                                  strides=(2, 2),
                                                  padding='valid',
                                                  activation=tf.nn.relu,
                                                  kernel_initializer=w_init)

            deconv_4 = tf.layers.conv2d_transpose(inputs=deconv_3,
                                                  filters=1,
                                                  kernel_size=[3, 3],
                                                  strides=(2, 2),
                                                  padding='valid',
                                                  activation=tf.nn.relu,
                                                  kernel_initializer=w_init)

            deconv_4_legit = deconv_4[:,1:29,1:29,:]

        return deconv_4_legit


    def _objective(self):

        # self.N = tf.contrib.distributions.Normal(tf.zeros((self.z_dim)),
        #                                          tf.ones((self.z_dim)))
        # self.Q_dist = tf.contrib.distributions.Normal(self.mu,
        #                                               self.log_sig)
        #
        # self.latent_loss = tf.contrib.distributions.kl(self.Q_dist, self.N)

        self.latent_loss = -.5 * tf.reduce_sum(
            1. + self.log_sig - tf.pow(self.mu, 2) - tf.exp(self.log_sig))
        self.reconstruction_loss = tfl.mean_square(self.X, self.trn_recon)
        total_loss = tf.reduce_mean(self.reconstruction_loss + self.latent_loss)
        self.optim = tf.train.AdamOptimizer(self.lr).minimize(total_loss)


    def train(self, X):
        fetches = [self.optim, self.reconstruction_loss, self.latent_loss]
        _, mse, kl_loss = self.sess.run(fetches, feed_dict={self.X : X})

        return mse, kl_loss


    def embed(self, X):
        return self.sess.run(self.z, feed_dict={self.X : X})


    def sample(self, z):
        return self.sess.run(self.tst_recon,
                             feed_dict={self.z_rnd_ph : z})