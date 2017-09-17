import tensorflow as tf
import tflearn as tfl
import numpy as np

class GAN:
    def __init__(self, in_dim, z_dim, lr, mnistdataset):
        self.in_dim = in_dim
        self.z_dim = z_dim
        self.lr = lr
        self.mnistdataset = mnistdataset

        self.g = tf.Graph()

        with self.g.as_default():

            with tf.name_scope("placeholders"):
                self.X = tf.placeholder(dtype=tf.float32, shape=[None, 28, 28, 1],
                                        name='data_input')
                self.Z = tf.placeholder(dtype=tf.float32, shape=[None, self.z_dim],
                                        name='entropy')

            self.G = self._generator(self.Z)
            self.D_real = self._discriminator(self.X)
            self.D_fake = self._discriminator(self.G, reuse=True)

            self.D_loss = -tf.reduce_mean(tf.log(self.D_real) + tf.log(1 - self.D_fake))
            self.G_loss = -tf.reduce_mean(tf.log(self.D_fake)
                                          # + tf.log(1 - self.D_real)
                                          )

            self.D_vars = tf.get_collection(key=tf.GraphKeys.GLOBAL_VARIABLES,
                                       scope='discriminator')
            self.G_vars = tf.get_collection(key=tf.GraphKeys.GLOBAL_VARIABLES,
                                       scope='generator')

            self.D_optim = tf.train.AdamOptimizer(self.lr).minimize(self.D_loss,
                                                                    var_list=self.D_vars)
            self.G_optim = tf.train.AdamOptimizer(self.lr).minimize(self.G_loss,
                                                                    var_list=self.G_vars)

            tf.summary.image('generations', self.G)
            tf.summary.scalar('d_loss', self.D_loss)
            tf.summary.scalar('g_loss', self.G_loss)

            self.merged = tf.summary.merge_all()
            self._init_op = tf.global_variables_initializer()

        self.writer = tf.summary.FileWriter('/tmp/tensorboard/gan/1',
                                            graph=self.g)

        self.sess = tf.Session(graph=self.g)
        self.sess.run(self._init_op)


    def _generator(self, Z):
        z_rs = tf.reshape(Z, (-1, 1, 1, self.z_dim))
        with tf.variable_scope('generator'):
            w_init = tf.contrib.layers.xavier_initializer()
            deconv_1 = tf.layers.conv2d_transpose(inputs=z_rs,
                                                  filters=256,
                                                  kernel_size=[7, 7],
                                                  strides=(1, 1),
                                                  padding='valid',
                                                  activation=tfl.leaky_relu,
                                                  kernel_initializer=w_init)
            deconv_1 = tfl.batch_normalization(deconv_1)


            deconv_2 = tf.layers.conv2d_transpose(inputs=deconv_1,
                                                  filters=64,
                                                  kernel_size=[3, 3],
                                                  strides=(2, 2),
                                                  padding='same',
                                                  activation=tfl.leaky_relu,
                                                  kernel_initializer=w_init)
            deconv_2 = tfl.batch_normalization(deconv_2)

            deconv_3 = tf.layers.conv2d_transpose(inputs=deconv_2,
                                                  filters=1,
                                                  kernel_size=[3, 3],
                                                  strides=(2, 2),
                                                  padding='same',
                                                  activation=tfl.leaky_relu,
                                                  kernel_initializer=w_init)


        return deconv_3


    def _discriminator(self, X, reuse=False):
        X_noisy = tf.random_uniform(shape=tf.shape(X), maxval=0.3)
        with tf.variable_scope("discriminator", reuse=reuse):
            w_init = tfl.initializations.xavier()
            l1 = tfl.conv_2d(X, 32, (3, 3), (1, 1), 'same',
                             activation='leaky_relu', weights_init=w_init)
            l1 = tfl.batch_normalization(l1)
            l1 = tfl.avg_pool_2d(l1, [2, 2], strides=2)

            l2 = tfl.conv_2d(l1, 64, (3, 3), (1, 1), 'same', activation='leaky_relu',
                             weights_init=w_init)
            l2 = tfl.batch_normalization(l2)
            l2 = tfl.avg_pool_2d(l2, [2, 2], strides=2)

            l3 = tfl.conv_2d(l2, 64, (3, 3), (1, 1), 'same', activation='leaky_relu',
                             weights_init=w_init)
            l3 = tfl.batch_normalization(l3)
            l3 = tfl.avg_pool_2d(l3, [2, 2], strides=2)
            flattened = tfl.flatten(l3, 'flattened')

            dropped = tfl.dropout(flattened, keep_prob=0.7)

            fc1 = tfl.fully_connected(dropped, 32,
                                      weights_init=w_init,
                                      activation='leaky_relu')

            p_data = tfl.fully_connected(fc1, 1,
                                          weights_init=w_init,
                                          activation='sigmoid')

        return p_data


    def train_vanilla(self, batchsize):
        X = self._getBatch(batchsize)
        d_loss, _ = self._train_D(X)
        X = self._getBatch(batchsize)
        _, g_loss = self._train_G(X)

        return d_loss, g_loss

    def train_balanced(self, batchsize, lam):

        while True:
            X = self._getBatch(batchsize)
            d_loss, g_loss = self._train_D(X)

            if d_loss < lam*g_loss: break

        while True:
            X = self._getBatch(batchsize)
            d_loss, g_loss = self._train_G(X)

            if d_loss > lam*g_loss: break

        return d_loss, g_loss

    def _train_D(self, X):
        z = np.random.randn(len(X), self.z_dim)
        _, d_loss, g_loss = self.sess.run([self.D_optim, self.D_loss, self.G_loss],
                                  feed_dict={self.Z: z, self.X: X})

        return d_loss, g_loss

    def _train_G(self, X):
        z = np.random.randn(len(X), self.z_dim)
        _, d_loss, g_loss = self.sess.run([self.G_optim, self.D_loss, self.G_loss],
                                  feed_dict={self.Z: z, self.X : X})

        return d_loss, g_loss

    def summarize(self, X):
        z = np.random.randn(len(X), self.z_dim)
        summary = self.sess.run(self.merged, feed_dict={self.X : X, self.Z : z})
        self.writer.add_summary(summary)

    def generate(self, z):
        return self.sess.run(self.G, feed_dict={self.Z : z})

    def _getBatch(self, batchsize):
        X, _ = self.mnistdataset.next_batch(batchsize)
        X_tf = np.reshape(X, [batchsize, 28, 28, 1])
        return X_tf

    def _makeNoise(self, dims):
        return np.random.randn(*dims)
