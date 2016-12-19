import numpy as np
import tensorflow as tf
from keras.datasets import mnist
from copy import deepcopy
import matplotlib.pyplot as plt


def onehot(labels):
    labels_arr = np.array(labels)
    onehotlabs = np.zeros((len(labels), np.max(labels) + 1))
    onehotlabs[np.arange(len(labels)), labels_arr] = 1
    return onehotlabs

def getMNISTData():
    # load MNIST data
    (X_trn, t_trn), (X_tst, t_tst) = mnist.load_data()

    # proportions
    P = [0.005, 0.01, 0.1, 1.0]

    batch_size = 100

    # extract 0 to 5 data
    X_trn05 = X_trn[t_trn < 6]  # (36017, 28, 28)
    t_trn05 = t_trn[t_trn < 6]  # (36017,)
    X_tst05 = X_tst[t_tst < 6]  # (6031, 28, 28)
    t_tst05 = t_tst[t_tst < 6]  # (6031,)

    # extract 0 to 6 data (dictionary using the proportion as a key)
    X_trn69 = {}
    t_trn69 = {}
    n_69_samples = len(t_trn[t_trn >= 6])  # the number of all 6-9 samples

    for p in P:
        n_samples = batch_size * (
        int(n_69_samples * p) / batch_size)  # round to the batch size
        X_trn69[p] = X_trn[t_trn >= 6][:n_samples]
        t_trn69[p] = onehot(t_trn[t_trn >= 6][:n_samples] - 6)

    X_tst69 = X_tst[t_tst >= 6]  # (3969, 28, 28)
    t_tst69 = t_tst[t_tst >= 6] - 6  # (3969,)

    # Turn labels into onehot
    t_trn05 = onehot(t_trn05)
    t_tst05 = onehot(t_tst05)
    t_tst69 = onehot(t_tst69)

    # convert to floats having range [0, 1]
    def convert_to_float(array):
        return array.astype('float32') / 255

    # Expand last dimension to get 4d tensor and convert to float
    X_trn05 = np.expand_dims(convert_to_float(X_trn05), axis=3)
    X_tst05 = np.expand_dims(convert_to_float(X_tst05), axis=3)

    for p in P:
        X_trn69[p] = convert_to_float(np.expand_dims(X_trn69[p], axis=3))
    X_tst69 = convert_to_float(np.expand_dims(X_tst69, axis=3))

    return {'X_trn05' : X_trn05, 'X_tst05' : X_tst05,
            'X_trn69' : X_trn69, 'X_tst69' : X_tst69,
            't_trn05' : t_trn05, 't_tst05' : t_tst05,
            't_trn69' : t_trn69, 't_tst69' : t_tst69
            }

def getMNISTbatch(X,Y,n):

    if len(X) < n or len(Y) < n:
        print 'WARNING : Batchsize of {} is greater than dataset of size {}'.\
            format(n, len(X))

        return X, Y

    # Make index vector
    indexes = np.arange(len(X))

    # Shuffle vector
    np.random.shuffle(indexes)

    return (deepcopy(X[indexes[:n]]), deepcopy(Y[indexes[:n]]))


def main():

    # Get data
    MNISTdatadict = getMNISTData()

    # ==================== Make weights ====================

    # Weights ___________________________________________________
    w_conv1 = tf.get_variable(name="w_conv1",
                              shape=[3, 3, 1, 32],
                              initializer=tf.contrib.layers.xavier_initializer_conv2d())

    w_conv2 = tf.get_variable(name="w_conv2",
                              shape=[3, 3, 32, 32],
                              initializer=tf.contrib.layers.xavier_initializer_conv2d())

    w_fc_rs = tf.get_variable(name="w_fc_rs",
                              shape=[14 * 14 * 32, 128],
                              initializer=tf.contrib.layers.xavier_initializer())

    w_fc_05 = tf.get_variable(name="w_fc_05",
                              shape=[128, 6],
                              initializer=tf.contrib.layers.xavier_initializer())

    w_fc_69 = tf.get_variable(name="w_fc_69",
                              shape=[128, 4],
                              initializer=tf.contrib.layers.xavier_initializer())


    # Biases ___________________________________________
    b_conv1 = tf.Variable(tf.constant(0., shape=[32]))
    b_conv2 = tf.Variable(tf.constant(0., shape=[32]))
    b_fc_rs = tf.Variable(tf.constant(0., shape=[1]))
    b_fc_05 = tf.Variable(tf.constant(0., shape=[1]))
    b_fc_69 = tf.Variable(tf.constant(0., shape=[1]))

    # Make a list of only the fc weights for the transfer learning
    fc_weights_69_TL = [w_fc_rs, w_fc_69, b_fc_rs, b_fc_69]

    # Make list of variables that will be reinitialized even during pretraining
    reinit_vars = [w_fc_rs, w_fc_05, w_fc_69, b_fc_rs, b_fc_05, b_fc_69]

    # ==================== Make network ====================

    X = tf.placeholder(dtype=tf.float32, shape=[None, 28, 28, 1])
    Y_05 = tf.placeholder(dtype=tf.float32, shape=[None, 6])
    Y_69 = tf.placeholder(dtype=tf.float32, shape=[None, 4])

    #Convolutional layers
    l_conv1 = tf.nn.relu(
        tf.nn.conv2d(X, w_conv1, [1, 1, 1, 1], 'SAME') + b_conv1)
    l_conv2 = tf.nn.relu(
        tf.nn.conv2d(l_conv1, w_conv2, [1, 1, 1, 1], 'SAME') + b_conv2)

    # Max pooling layer
    mp_layer = tf.nn.max_pool(l_conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                              padding='SAME')
    # Reshape into vector
    fc_reshape = tf.reshape(mp_layer, shape=(-1, 14 * 14 * 32))

    # First fully connected layers
    l_fc_rs =  tf.nn.relu(tf.matmul(fc_reshape, w_fc_rs) + b_fc_rs)

    # Fully connected layers 05 datasets
    l_fc_05 = tf.matmul(l_fc_rs, w_fc_05) + b_fc_05

    # Fully connected layers 69 datasets
    l_fc_69 = tf.matmul(l_fc_rs, w_fc_69) + b_fc_69

    # Loss functions
    CE_05 = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(l_fc_05, Y_05))
    CE_69 = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(l_fc_69, Y_69))

    predict_05 = tf.argmax(tf.nn.softmax(l_fc_05), 1)
    predict_69 = tf.argmax(tf.nn.softmax(l_fc_69), 1)

    # Accuracy operation
    ACC_05 = tf.reduce_mean(tf.cast(tf.equal(predict_05, tf.argmax(Y_05, 1)),
                                    tf.float32))

    ACC_69 = tf.reduce_mean(tf.cast(tf.equal(predict_69, tf.argmax(Y_69, 1)),
                                    tf.float32))



    # Training ===================================================

    # OPTIMIZERS :

    # Plan and the respective optimizers:
    # 1) Train network on 69 dataset n times
    raw_69_optim = tf.train.AdamOptimizer(0.001).minimize(CE_69)

    # 2) Train network on 05 dataset n times
    raw_05_optim = tf.train.AdamOptimizer(0.001).minimize(CE_05)

    # 3) Train network on 69 with conv layers from the 05 dataset n times
    TL_69_optim_freeze_conv = tf.train.AdamOptimizer(0.001).minimize(CE_69,
                                                                    var_list=fc_weights_69_TL)

    # 4) Train network on 69 dataset n times with finetune
    finetune_69_optim = tf.train.AdamOptimizer(0.0003).minimize(CE_69)

    # Execution
    n_rep = 1
    batchSize = 100
    n_iters = 200
    every_n_samples = 1000
    every_n_iters = every_n_samples/batchSize

    # 1) Train network on 69 dataset n_rep times ===============

    sess = tf.Session()  # Make tensorflow session
    sess.run(tf.initialize_all_variables())  # Initialize all variables

    # proportions
    P = [0.005, 0.01, 0.1, 1.0]

    # Test accuracy matrix : (p, rep, n_iters/batchSize) ,
    test_acc_mat = np.zeros((len(P), n_rep, n_iters/every_n_iters))

    # Test data
    X_tst = MNISTdatadict['X_tst69']
    t_tst = MNISTdatadict['t_tst69']


    # === Run the training for all 4 dataset proportions of the 69 ====
    # =================================================================
    for pi,p in enumerate(P):

        print 'Testing for proportion: {}'.format(p)

        # Data
        X_data = MNISTdatadict['X_trn69'][p]
        Y_data = MNISTdatadict['t_trn69'][p]

        # Run training n_repetition times
        for r in range(n_rep):
            print 'Starting rep: {}'.format(r + 1)
            # Reset variables
            sess.run(tf.initialize_all_variables())

            tst_ctr = 0
            for i in range(n_iters + 1):
                batch = getMNISTbatch(X_data,Y_data,batchSize)

                sess.run(raw_69_optim, feed_dict={X: batch[0],Y_69: batch[1]})

                # Perform test accuracy
                if i > 0 and (i % every_n_iters == 0):
                    acc = sess.run(ACC_69, feed_dict={X: X_tst, Y_69: t_tst})
                    print 'Test accuracy: {}'.format(acc)

                    # Add accuracy to our log
                    test_acc_mat[pi, r, tst_ctr] = acc
                    tst_ctr += 1


    # === DO the 05 pretraining ======================================
    # ================================================================

    # Reset all the variables
    sess.run(tf.initialize_all_variables())

    # Data
    X_data = MNISTdatadict['X_trn05']
    Y_data = MNISTdatadict['t_trn05']
    X_tst = MNISTdatadict['X_tst05']
    t_tst = MNISTdatadict['t_tst05']

    print "Starting 05 training. "


    # Train the 05 network
    # Run training n_repetition times
    for i in range(n_iters + 1):
        batch = getMNISTbatch(X_data, Y_data, batchSize)
        sess.run(raw_05_optim, feed_dict={X : batch[0], Y_05: batch[1]})

        # Perform test accuracy
        if i > 0 and i % 1000 == 0:
            acc = sess.run(ACC_05, feed_dict={X : X_tst, Y_05: t_tst})
            print 'Pretraining with 05 net: Iter: {} Test accuracy: {}'\
                .format(i, acc)


    # === Do the 69 with pretrained conv layers(FROZEN) ======================
    # ================================================================

    print "Starting 69 training with pretrained conv layers frozen"

    # Test data
    X_tst = MNISTdatadict['X_tst69']
    t_tst = MNISTdatadict['t_tst69']

    # Test accuracy matrix : (p, rep, n_iters/batchSize) ,
    test_acc_mat_pt = np.zeros((len(P), n_rep, n_iters/every_n_iters))

    for pi, p in enumerate(P):

        print 'Testing for proportion: {}'.format(p)

        # Data
        X_data = MNISTdatadict['X_trn69'][p]
        Y_data = MNISTdatadict['t_trn69'][p]

        # Run training n_repetition times
        for r in range(n_rep):
            print 'Starting rep: {}'.format(r + 1)
            # Reset variables
            sess.run(tf.initialize_variables(reinit_vars))

            tst_ctr = 0
            for i in range(n_iters + 1):
                batch = getMNISTbatch(X_data, Y_data, batchSize)

                sess.run(TL_69_optim_freeze_conv, feed_dict={X: batch[0], Y_69: batch[
                    1]})

                # Perform test accuracy
                if i > 0 and (i % every_n_iters == 0):
                    acc = sess.run(ACC_69, feed_dict={X: X_tst, Y_69: t_tst})
                    print 'Test accuracy: {}'.format(acc)

                    # Add accuracy to our log
                    test_acc_mat_pt[pi, r, tst_ctr] = acc
                    tst_ctr += 1

    # === Do the 69 with pretrained conv layers (FINETUNED) ==========
    # ================================================================



    print "Starting 69 training with pretrained conv layers, non-frozen"

    # Test data
    X_tst = MNISTdatadict['X_tst69']
    t_tst = MNISTdatadict['t_tst69']

    # Test accuracy matrix : (p, rep, n_iters/batchSize) ,
    test_acc_mat_pt_finetune = np.zeros(
        (len(P), n_rep, n_iters / every_n_iters))


    for pi, p in enumerate(P):

        print 'Testing for proportion: {}'.format(p)

        # Data
        X_data = MNISTdatadict['X_trn69'][p]
        Y_data = MNISTdatadict['t_trn69'][p]

        # Run training n_repetition times
        for r in range(n_rep):
            print 'Starting rep: {}'.format(r + 1)
            # Reset variables
            sess.run(tf.initialize_variables(reinit_vars))

            tst_ctr = 0
            for i in range(n_iters + 1):
                batch = getMNISTbatch(X_data, Y_data, batchSize)

                sess.run(finetune_69_optim,
                         feed_dict={X: batch[0], Y_69: batch[
                             1]})

                # Perform test accuracy
                if i > 0 and (i % every_n_iters == 0):
                    acc = sess.run(ACC_69, feed_dict={X: X_tst,
                                                      Y_69: t_tst})
                    print 'Test accuracy: {}'.format(acc)

                    # Add accuracy to our log
                    test_acc_mat_pt_finetune[pi, r, tst_ctr] = acc
                    tst_ctr += 1

    # Close session because we are going to modify the network
    sess.close()

    # ==== Modify the network with dropout ========================
    # =============================================================



    # Dropout from the conv features
    fc_reshape_DO = tf.nn.dropout(fc_reshape, keep_prob=0.5)

    l_fc_rs = tf.nn.relu(tf.matmul(fc_reshape_DO, w_fc_rs) + b_fc_rs)

    fc_rs_DO = tf.nn.dropout(l_fc_rs, keep_prob=0.5)

    # Add dropout to the 69 fc layer
    l_fc_69 = tf.matmul(fc_rs_DO, w_fc_69) + b_fc_69

    CE_69 = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(l_fc_69, Y_69))

    # 4) Train network on 69 with conv layers from 05 without freezing n times
    TL_69_optim_DO = tf.train.AdamOptimizer(0.001).minimize(CE_69)

    # Make new session
    sess = tf.Session()

    # === Run the training for all 4 dataset proportions of the 69 ====
    # =================================================================
    # Test accuracy matrix : (p, rep, n_iters/batchSize) ,
    test_acc_mat_do = np.zeros((len(P), n_rep, n_iters/every_n_iters))

    print "Starting raw 69 training with dropout"

    for pi, p in enumerate(P):

        print 'Testing for proportion: {}'.format(p)

        # Data
        X_data = MNISTdatadict['X_trn69'][p]
        Y_data = MNISTdatadict['t_trn69'][p]

        # Run training n_repetition times
        for r in range(n_rep):
            print 'Starting rep: {}'.format(r + 1)
            # Reset variables
            sess.run(tf.initialize_all_variables())

            tst_ctr = 0
            for i in range(n_iters + 1):
                batch = getMNISTbatch(X_data, Y_data, batchSize)

                sess.run(TL_69_optim_DO, feed_dict={X: batch[0], Y_69: batch[1]})

                # Perform test accuracy
                if i > 0 and (i % every_n_iters == 0):
                    acc = sess.run(ACC_69, feed_dict={X: X_tst, Y_69: t_tst})
                    print 'Test accuracy: {}'.format(acc)

                    # Add accuracy to our log
                    test_acc_mat_do[pi, r, tst_ctr] = acc
                    tst_ctr += 1



    # ====== Do a bunch of plotting ==================================
    print "Plotting test errors: "

    # test_acc_mat    - 69 raw
    # test_acc_mat_pt - pretrained errors on 69
    # test_acc_mat_do - 69 with dropout

    # Plot the 69 raw training
    plot_acc_matrix(test_acc_mat, "Raw 69 training")

    # Plot the 69 training with pretrained layers
    plot_acc_matrix(test_acc_mat_pt, "69 training with pretraining")

    # Plot the 69 training with pretrained layers (FINETUNED)
    plot_acc_matrix(test_acc_mat_pt_finetune, "69 training with pretraining ("
                                              "Finetuned)")

    # Plot the 69 raw training with dropout
    plot_acc_matrix(test_acc_mat_do, "Raw 69 training with dropout")

def plot_acc_matrix(test_acc_matrix, title):

    # Averaged test errors
    mat = np.mean(test_acc_matrix, axis=1)

    # Plot vector
    t = np.arange(mat.shape[1])

    # Make figure
    fig = plt.figure()

    # red dashes, blue squares and green triangles
    plt.plot(t, mat[0], 'r-', t, mat[1], 'b-', t, mat[2], 'g-', t, mat[3],
             'y-')

    plt.title(title)
    plt.xlabel("Iters x 10")
    plt.ylabel("Test accuracy")
    plt.legend(["p = 0.005","p = 0.01","p = 0.1","p = 1"])
    plt.ylim(0,1.7)

    plt.savefig("{}.png".format(title.replace(" ", "")), dpi=100)


if __name__ == "__main__":
    main()


