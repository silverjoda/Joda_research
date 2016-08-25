import tensorflow as tf

class tfmodel:

    def __init__(self):
        self.x = tf.constant(3)
        self.y = tf.constant(8)

        self.z = self.x + self.y

