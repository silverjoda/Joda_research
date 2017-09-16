import tensorflow as tf
import tflearn as tfl

class GAN:
    def __init__(self, in_dim, z_dim, lr):
        self.in_dim = in_dim
        self.z_dim = z_dim
        self.lr = lr

        with tf.name_scope("placeholders"):
            self.X = tf.placeholder(dtype=tf.float32, shape=[None, 28, 28, 1],
                                    name='data_input')
            self.Z = tf.placeholder(dtype=tf.float32, shape=[None, self.z_dim],
                                    name='entropy')


    def _generator(self, z):
        z_rs = tf.reshape(z, (-1, 1, 1, self.z_dim))
        with tf.variable_scope('generator'):
            w_init = tf.contrib.layers.xavier_initializer()
            deconv_1 = tf.layers.conv2d_transpose(inputs=z_rs,
                                                  filters=128,
                                                  kernel_size=[7, 7],
                                                  strides=(1, 1),
                                                  padding='valid',
                                                  activation=tf.nn.relu,
                                                  kernel_initializer=w_init)

            deconv_2 = tf.layers.conv2d_transpose(inputs=deconv_1,
                                                  filters=32,
                                                  kernel_size=[3, 3],
                                                  strides=(2, 2),
                                                  padding='same',
                                                  activation=tf.nn.relu,
                                                  kernel_initializer=w_init)

            deconv_3 = tf.layers.conv2d_transpose(inputs=deconv_2,
                                                  filters=1,
                                                  kernel_size=[3, 3],
                                                  strides=(2, 2),
                                                  padding='same',
                                                  activation=tf.nn.relu,
                                                  kernel_initializer=w_init)

        return deconv_3


    def _discriminator(self, X, reuse):
        with tf.variable_scope("discriminator", reuse=reuse):
            w_init = tfl.initializations.xavier()
            l1 = tfl.conv_2d(X, 32, (3, 3), (1, 1), 'same',
                             activation='relu', weights_init=w_init)
            l1 = tfl.max_pool_2d(l1, [2, 2], strides=2)
            l2 = tfl.conv_2d(l1, 32, (3, 3), (1, 1), 'same', activation='relu',
                             weights_init=w_init)
            l2 = tfl.max_pool_2d(l2, [2, 2], strides=2)
            l3 = tfl.conv_2d(l2, 32, (3, 3), (1, 1), 'same', activation='relu',
                             weights_init=w_init)
            l3 = tfl.max_pool_2d(l3, [2, 2], strides=2)
            flattened = tfl.flatten(l3, 'flattened')

            p_data = tfl.fully_connected(flattened, 1,
                                          weights_init=w_init,
                                          activation='sigmoid')

        return p_data


    def train(self):
        pass


    def generate(self, z):
        pass
