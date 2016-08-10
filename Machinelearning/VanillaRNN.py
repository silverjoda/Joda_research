import numpy as np
import tensorflow as tf

# training dataset generation
int2binary = {}
binary_dim = 8

largest_number = pow(2, binary_dim)
binary = np.unpackbits(
    np.array([range(largest_number)], dtype=np.uint8).T, axis=1)

for i in range(largest_number):
    int2binary[i] = binary[i]

# input variables
alpha = 0.05
input_dim = 2
hidden_dim = 16
output_dim = 1
batchsize = 32

iters = 10000

# Input placeholder for single binary pair input
X = tf.placeholder(dtype=tf.float32, shape=[None, binary_dim, input_dim])

# Output label
Y = tf.placeholder(dtype=tf.float32, shape=[None, binary_dim])

# Output weights
w_out = tf.Variable(tf.random_normal(shape=[hidden_dim, output_dim],
                                     stddev=0.1))

b_out = tf.Variable(tf.constant(0, dtype=tf.float32))

# Single RNN cell
basic_cell = tf.nn.rnn_cell.BasicRNNCell(num_units=hidden_dim)

# Symbolic outputs of the whole unraveled RNN network
output, _ = tf.nn.dynamic_rnn(cell=basic_cell,
                              inputs=X,
                              dtype=tf.float32)


output = tf.reshape(output, [-1, hidden_dim])
output = tf.nn.tanh(tf.matmul(output, w_out) + b_out)
output = tf.reshape(output, [-1, binary_dim])

cost = tf.reduce_mean(tf.square(Y - output))

# # List of tensors with shape (batch_size, hidden_dim)
# unpacked_hidden = tf.unpack(tf.transpose(output, (1,0,2)))
#
# # List of ready outputs
# unpacked_output = [tf.nn.tanh(tf.matmul(single_output, w_out) + b_out)
#                    for single_output in unpacked_hidden]
#
# # Pack output into single output vector
# packed_output = tf.squeeze(tf.transpose(tf.pack(unpacked_output),(1, 0, 2)),[2])
#
# # Define simple square difference cost function
# cost = tf.reduce_mean(tf.square(Y - packed_output))

# Define optimizer
optimizer = tf.train.GradientDescentOptimizer(alpha)

# Training step
optim_step = optimizer.minimize(cost)

sess = tf.Session()
sess.run(tf.initialize_all_variables())

# Merge tensorflow summaries
merged = tf.merge_all_summaries()

# Create tensorboard graph
writer = tf.train.SummaryWriter('/tmp/rec_tensorboard_logs/', sess.graph)


# training logic
for j in range(iters):

    # generate a simple addition problem (a + b = c)

    a_ints = np.random.randint(0, largest_number / 2, batchsize)  # int version
    a = np.array([int2binary[a_int] for a_int in a_ints], dtype=np.uint8)
    a = np.expand_dims(a, axis=(2))

    b_ints = np.random.randint(0, largest_number / 2, batchsize)  # int version
    b = np.array([int2binary[b_int] for b_int in b_ints], dtype=np.uint8)
    b = np.expand_dims(b, axis=(2))

    ab = np.concatenate((a, b), axis=2)

    # true answer
    c_ints = a_ints + b_ints
    c = np.array([int2binary[c_int] for c_int in c_ints], dtype=np.uint8)

    # Run a train step
    sess.run(optim_step, feed_dict={X: ab, Y: c})

    # print out progress
    if (j % (iters / 10) == 0):

        # Run a train step
        run_results, pred = sess.run([cost, output], feed_dict={X: ab, Y: c})

        d = np.round(pred[0]).astype(np.uint8)

        print "Error:" + str(run_results)
        print "Pred: {}".format(d)
        print "True:" + str(c[0])
        out = 0
        for index, x in enumerate(reversed(d)):
            out += x * pow(2, index)
        print str(a_ints[0]) + " + " + str(b_ints[0]) + " = " + str(out)
        print "------------"
