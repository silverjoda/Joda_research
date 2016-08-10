import tensorflow as tf
import numpy as np

class FStransformer:

    def __init__(self):
        self.X = tf.placeholder("float", [None, 256, 256, 3])
        self.p_keep_conv = tf.placeholder("float")
        self.params = self.loadparamsFromVGG()
        self.Yvec = self.model(self.X, self.params, self.p_keep_conv)

        # Start session
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)
        self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
        self.sess.run(tf.initialize_all_variables())

    def model(self, X, params, p_keep_conv):

        weights = params[0]
        biases = params[1]

        # Z1
        l1 = tf.nn.relu(tf.nn.conv2d(X, weights["w1"], [1, 1, 1, 1], 'SAME') + biases["b1"])
        l2 = tf.nn.relu(tf.nn.conv2d(l1, weights["w2"], [1, 1, 1, 1], 'SAME') + biases["b2"])

        l2 = tf.nn.max_pool(l2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

        # Z2
        l3 = tf.nn.relu(tf.nn.conv2d(l2, weights["w3"], [1, 1, 1, 1], 'SAME') + biases["b3"])
        l4 = tf.nn.relu(tf.nn.conv2d(l3, weights["w4"], [1, 1, 1, 1], 'SAME') + biases["b4"])

        l4 = tf.nn.max_pool(l4, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

        # Z3
        l5 = tf.nn.relu(tf.nn.conv2d(l4, weights["w5"], [1, 1, 1, 1], 'SAME') + biases["b5"])
        l6 = tf.nn.relu(tf.nn.conv2d(l5, weights["w6"], [1, 1, 1, 1], 'SAME') + biases["b6"])
        l7 = tf.nn.relu(tf.nn.conv2d(l6, weights["w7"], [1, 1, 1, 1], 'SAME') + biases["b7"])  # tap l

        l7_p = tf.nn.max_pool(l7, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

        # Z4
        l8 = tf.nn.relu(tf.nn.conv2d(l7_p, weights["w8"], [1, 1, 1, 1], 'SAME') + biases["b8"])
        l9 = tf.nn.relu(tf.nn.conv2d(l8, weights["w9"], [1, 1, 1, 1], 'SAME') + biases["b9"])
        l10 = tf.nn.relu(tf.nn.conv2d(l9, weights["w10"], [1, 1, 1, 1], 'SAME') + biases["b10"])  # tap m

        # Added _p suffix to differentiate between pooled and unpooled layer
        l10_p = tf.nn.max_pool(l10, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

        # Z5
        l11 = tf.nn.relu(tf.nn.conv2d(l10_p, weights["w11"], [1, 1, 1, 1], 'SAME') + biases["b11"])
        l12 = tf.nn.relu(tf.nn.conv2d(l11, weights["w12"], [1, 1, 1, 1], 'SAME') + biases["b12"])
        l13 = tf.nn.relu(tf.nn.conv2d(l12, weights["w13"], [1, 1, 1, 1], 'SAME') + biases["b13"])  # tap u

        l13_p = tf.nn.max_pool(l13, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

        fc = tf.reshape(l13_p, shape=[512 * 8 * 8])
        fc = tf.matmul(fc, weights["wfc"]) + biases["bfc"]

        return fc

    def loadparamsFromVGG(self):

        weights = {}
        biases = {}

        weights["w1"] = tf.Variable(np.load('VGGweights/VGG16/conv1_1_w.npy'),trainable=False)
        weights["w2"] = tf.Variable(np.load('VGGweights/VGG16/conv1_2_w.npy'),trainable=False)

        weights["w3"] = tf.Variable(np.load('VGGweights/VGG16/conv2_1_w.npy'),trainable=False)
        weights["w4"] = tf.Variable(np.load('VGGweights/VGG16/conv2_2_w.npy'),trainable=False)

        weights["w5"] = tf.Variable(np.load('VGGweights/VGG16/conv3_1_w.npy'),trainable=False)
        weights["w6"] = tf.Variable(np.load('VGGweights/VGG16/conv3_2_w.npy'),trainable=False)
        weights["w7"] = tf.Variable(np.load('VGGweights/VGG16/conv3_3_w.npy'),trainable=False)

        weights["w8"] = tf.Variable(np.load('VGGweights/VGG16/conv4_1_w.npy'),trainable=False)
        weights["w9"] = tf.Variable(np.load('VGGweights/VGG16/conv4_2_w.npy'),trainable=False)
        weights["w10"] = tf.Variable(np.load('VGGweights/VGG16/conv4_3_w.npy'),trainable=False)

        weights["w11"] = tf.Variable(np.load('VGGweights/VGG16/conv5_1_w.npy'),trainable=False)
        weights["w12"] = tf.Variable(np.load('VGGweights/VGG16/conv5_2_w.npy'),trainable=False)
        weights["w13"] = tf.Variable(np.load('VGGweights/VGG16/conv5_3_w.npy'),trainable=False)

        weights["wfc"] = tf.Variable(np.load('VGGweights/VGG16/fc1_w.npy'),trainable=False)


        # Biases ==================================================================
        biases["b1"] = tf.Variable(np.load('VGGweights/VGG16/conv1_1_b.npy'),trainable=False)
        biases["b2"] = tf.Variable(np.load('VGGweights/VGG16/conv1_2_b.npy'),trainable=False)

        biases["b3"] = tf.Variable(np.load('VGGweights/VGG16/conv2_1_b.npy'),trainable=False)
        biases["b4"] = tf.Variable(np.load('VGGweights/VGG16/conv2_2_b.npy'),trainable=False)

        biases["b5"] = tf.Variable(np.load('VGGweights/VGG16/conv3_1_b.npy'),trainable=False)
        biases["b6"] = tf.Variable(np.load('VGGweights/VGG16/conv3_2_b.npy'),trainable=False)
        biases["b7"] = tf.Variable(np.load('VGGweights/VGG16/conv3_3_b.npy'),trainable=False)

        biases["b8"] = tf.Variable(np.load('VGGweights/VGG16/conv4_1_b.npy'),trainable=False)
        biases["b9"] = tf.Variable(np.load('VGGweights/VGG16/conv4_2_b.npy'),trainable=False)
        biases["b10"] = tf.Variable(np.load('VGGweights/VGG16/conv4_3_b.npy'),trainable=False)

        biases["b11"] = tf.Variable(np.load('VGGweights/VGG16/conv5_1_b.npy'),trainable=False)
        biases["b12"] = tf.Variable(np.load('VGGweights/VGG16/conv5_2_b.npy'),trainable=False)
        biases["b13"] = tf.Variable(np.load('VGGweights/VGG16/conv5_3_b.npy'),trainable=False)

        biases["bfc"] = tf.Variable(np.load('VGGweights/VGG16/fc1_b.npy'),trainable=False)

        return weights,biases

    def transform(self,img):
        return self.sess.run(self.Yvec, feed_dict={self.X : np.expand_dims(img, 0), self.p_keep_conv : 1})
