import numpy as np

from keras.datasets import mnist
import tensorflow as tf

# TODO: create batching function
# TODO: Add tensorflow naming, scope and datalogging ERRRYWHERE!!!!

def onehot(labels):
    labels_arr = np.array(labels)
    onehotlabs = np.zeros((len(labels), np.max(labels) + 1))
    onehotlabs[np.arange(len(labels)), labels_arr] = 1
    return onehotlabs

# =========== PREPARE DATA ========================
# load MNIST data
(X_trn, t_trn), (X_tst, t_tst) = mnist.load_data()

# proportions
P = [0.005, 0.01, 0.1, 1.0]

batch_size = 100

# extract 0 to 5 data
X_trn05 = X_trn[t_trn < 6] # (36017, 28, 28)
t_trn05 = t_trn[t_trn < 6] # (36017,)
X_tst05 = X_tst[t_tst < 6] # (6031, 28, 28)
t_tst05 = t_tst[t_tst < 6] # (6031,)

# extract 0 to 6 data (dictionary using the proportion as a key)
X_trn69 = {}
t_trn69 = {}
n_69_samples = np.sum(t_trn[t_trn >= 6])  # the number of all 6-9 samples

for p in P:
    n_samples = batch_size * (int(n_69_samples * p) / batch_size)  # round to the batch size
    X_trn69[p] = X_trn[t_trn >= 6][:n_samples]
    t_trn69[p] = t_trn[t_trn >= 6][:n_samples] - 6

X_tst69 = X_tst[t_tst >= 6] # (3969, 28, 28)
t_tst69 = t_tst[t_tst >= 6] - 6 # (3969,)

# Turn labels into onehot
t_trn05 = onehot(t_trn05)
t_tst05 = onehot(t_tst05)

# convert to floats having range [0, 1]
def convert_to_float(array):
    return array.astype('float32') / 255


# Expand last dimension to get 4d tensor and convert to float
X_trn05 = np.expand_dims(convert_to_float(X_trn05), axis=3)
X_tst05 = np.expand_dims(convert_to_float(X_tst05), axis=3)

for p in P:
    X_trn69[p] = convert_to_float(np.expand_dims(X_trn69[p], axis=3))
X_tst69 = convert_to_float(np.expand_dims(X_tst69, axis=3))

for p in P:
    X_trn69[p] = convert_to_float(X_trn69[p])

# ======================================================

# ==================== Make weights ====================
# Weights ___________________________________________________
w_conv1 = tf.Variable(tf.random_normal(shape=[3, 3, 1, 32],
                                       stddev=0.05))
w_conv2 = tf.Variable(tf.random_normal(shape=[3, 3, 32, 32],
                                       stddev=0.05))

w_fc_05 = tf.Variable(tf.random_normal(shape=[14 * 14 * 32, 6],
                                       stddev=0.05))

w_fc_69 = tf.Variable(tf.random_normal(shape=[14 * 14 * 32, 4],
                                       stddev=0.05))

# Biases ___________________________________________
b_conv1 = tf.Variable(tf.constant(0., shape=[32]))
b_conv2 = tf.Variable(tf.constant(0., shape=[32]))
b_fc_05 = tf.Variable(tf.constant(0., shape=[1]))
b_fc_69 = tf.Variable(tf.constant(0., shape=[1]))

# Make a list of only the fc weights for the transfer learning
fc_weights_69_TL = [w_fc_69, b_fc_69]

# ==================== Make network ====================
X = tf.placeholder(dtype=tf.float32, shape=[-1,28,28,1])
Y_05 = tf.placeholder(dtype=tf.float32, shape=[-1,6])
Y_69 = tf.placeholder(dtype=tf.float32, shape=[-1,4])

# Convolutional layers
l_conv1 = tf.nn.relu(tf.nn.conv2d(X, w_conv1, [1,1,1,1], 'SAME') + b_conv1)
l_conv2 = tf.nn.relu(tf.nn.conv2d(l_conv1, w_conv2, [1,1,1,1], 'SAME') + b_conv1)

# Max pooling layer
mp_layer = tf.nn.max_pool(l_conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                          padding='SAME')
# Reshape into vector
fc_reshape = tf.reshape(mp_layer, shape=(-1, 14*14))

# Fully connected layers 05 datasets
l_fc_05 = tf.nn.relu(tf.matmul(fc_reshape, w_fc_05) + b_fc_05)
#out_05 = tf.nn.softmax(l_fc_05)

# Fully connected layers 69 datasets
l_fc_69 = tf.nn.relu(tf.matmul(fc_reshape, w_fc_69) + b_fc_69)
#out_69 = tf.nn.softmax(l_fc_69)

# Loss functions
CE_05 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(l_fc_05, Y_05))
CE_69 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(l_fc_69, Y_69))

# Training ===================================================

# OPTIMIZERS :



# Plan and the respective optimizers:
# 1) Train network on 69 dataset n times
raw_69_optim = tf.train.AdamOptimizer(0.01).minimize(CE_69)

# 2) Train network on 05 dataset n times
raw_05_optim = tf.train.AdamOptimizer(0.01).minimize(CE_05)

# 3) Train network on 69 with conv layers from the 05 dataset n times
TL_69_optim_freeze_conv = tf.train.AdamOptimizer(0.01).minimize(CE_69,
                                                    var_list=fc_weights_69_TL)

# 4) Train network on 69 with conv layers from 05 without freezing n times
TL_69_optim_finetune = tf.train.AdamOptimizer(0.01).minimize(CE_69)

# Execution
n_rep = 3

# 1) Train network on 69 dataset n_rep times ===============

sess = tf.Session() # Make tensorflow session
sess.run(tf.initialize_all_variables()) # Initialize all variables

for i in range(n_rep):
    pass




# Make tensorflow session
sess = tf.Session()





