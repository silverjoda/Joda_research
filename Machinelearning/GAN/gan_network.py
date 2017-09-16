import tensorflow as tf
import tflearn as tfl

class GAN:
    def __init__(self, in_dim, z_dim, lr):
        self.in_dim = in_dim
        self.z_dim = z_dim
        self.lr = lr

        with tf.name_scope("placeholders"):
            pass



    def _generator(self):

        pass


    def _discriminator(self):
        pass


    def train(self):
        pass


    def generate(self, z):
        pass
