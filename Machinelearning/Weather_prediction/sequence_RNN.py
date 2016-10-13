import tensorflow as tf

class FFNetwork:



    def __init__(self,
                 input_dim,
                 output_dim,
                 n_training_steps,
                 n_prediction_steps,
                 config):

        # Amount of variables that will be fed into the network at each time
        # step. This is the same as the input dimension of the network input.
        # All variables are assumed to be normalized to (-1,1)
        self.input_dim = input_dim

        # Amount of steps that we will be classifying. These time steps do not
        # necessarily correspond with the input time steps. We only classify
        # one output variable (e.g. temperature).
        self.output_dim = output_dim

        # Amount of time steps we want to feed to the network before expecting
        # it to produce a classification of the future n time steps
        self.n_training_steps = n_training_steps

        # Amount of steps that Y will be predicting
        self.n_prediction_steps = n_prediction_steps

        self.dropout_keep_p = config["dropout_keep_p"]
        self.learning_rate = config["learning_rate"]
        self.horizon_decay = config["horizon_decay"]
        self.l2_loss = config["l2_alpha"]

        # Set dropout percentage to zero if we are visualizing
        if config["mode"] == 'vis':
            self.dropout_keep_p = 1.

        # Intialize the classifier (forecaster) network
        self._init_FFNN()

        # Define an optimizer
        self.optimizer = tf.train.AdamOptimizer(
            learning_rate=self.learning_rate)

        # Train step
        self.train_step = self.optimizer.minimize(self.cost)



    def _init_FFNN(self):
        # Input placeholder for single binary pair input
        self.X = tf.placeholder(dtype=tf.float32, shape=[None,
                                                         self.n_training_steps,
                                                         self.input_dim])

        # Output label
        self.Y = tf.placeholder(dtype=tf.float32, shape=[None,
                                                         self.n_prediction_steps,
                                                         self.output_dim])

        # Weights ___________________________________________________
        self.w_l1 = tf.Variable(tf.random_normal(shape=[1, 7, self.input_dim,
                                                        16], stddev=0.05))
        self.w_l2 = tf.Variable(tf.random_normal(shape=[1, 7, 16, 32],
                                                 stddev=0.05))
        self.w_l3 = tf.Variable(tf.random_normal(shape=[1, 5, 32, 32],
                                                 stddev=0.05))
        self.w_l4 = tf.Variable(tf.random_normal(shape=[1, 5, 32, 32],
                                                 stddev=0.05))
        self.w_l5 = tf.Variable(tf.random_normal(shape=[1, 5, 32, 1],
                                                 stddev=0.05))

        # Biases ___________________________________________
        self.b_l1 = tf.Variable(tf.constant(0., shape=[16]))
        self.b_l2 = tf.Variable(tf.constant(0., shape=[32]))
        self.b_l3 = tf.Variable(tf.constant(0., shape=[32]))
        self.b_l4 = tf.Variable(tf.constant(0., shape=[32]))
        self.b_l5 = tf.Variable(tf.constant(0., shape=[1]))

        # Layers _____________________________________________
        self.l1 = tf.nn.tanh(tf.nn.conv2d(tf.expand_dims(self.X, dim=[1]),
                               self.w_l1,
                               strides = [1, 1, 3, 1],
                               padding='SAME') + self.b_l1)

        self.l1 = tf.nn.dropout(self.l1, keep_prob=self.dropout_keep_p)

        self.l2 = tf.nn.tanh(tf.nn.conv2d(self.l1,
                               self.w_l2,
                               strides=[1, 1, 2, 1],
                               padding='VALID') + self.b_l2)

        self.l2 = tf.nn.dropout(self.l2, keep_prob=self.dropout_keep_p)

        self.l3 = tf.nn.tanh(tf.nn.conv2d(self.l2,
                               self.w_l3,
                               strides=[1, 1, 2, 1],
                               padding='SAME') + self.b_l3)

        self.l3 = tf.nn.dropout(self.l3, keep_prob=self.dropout_keep_p)

        self.l4 = tf.nn.tanh(tf.nn.conv2d(self.l3,
                               self.w_l4,
                               strides=[1, 1, 1, 1],
                               padding='VALID') + self.b_l4)

        self.l4 = tf.nn.dropout(self.l4, keep_prob=self.dropout_keep_p)

        self.l5 = tf.nn.conv2d(self.l4,
                               self.w_l5,
                               strides=[1, 1, 1, 1],
                               padding='VALID') + self.b_l5

        self.l5 = tf.squeeze(self.l5, squeeze_dims=[1])

        # print tf.expand_dims(self.X, dim=[1]).get_shape()
        # print self.l5.get_shape()
        # print self.Y.get_shape()
        # exit()

        self.lossfun = tf.reduce_mean(tf.square(self.l5 - self.Y))

        self.cost = self.lossfun + self.l2_loss * (tf.nn.l2_loss(self.w_l1) +
                                                   tf.nn.l2_loss(self.w_l2) +
                                                   tf.nn.l2_loss(self.w_l3) +
                                                   tf.nn.l2_loss(self.w_l4) +
                                                   tf.nn.l2_loss(self.w_l5))

        self.predict = self.l5

class WeatherNetwork:
    """
    Class of a recurrent neural network in TensorFlow which is built to predict
    immediate weather
    """
    def __init__(self,
                 input_dim,
                 output_dim,
                 n_training_steps,
                 n_prediction_steps,
                 config):

        # Amount of variables that will be fed into the network at each time
        # step. This is the same as the input dimension of the network input.
        # All variables are assumed to be normalized to (-1,1)
        self.input_dim = input_dim

        # Amount of steps that we will be classifying. These time steps do not
        # necessarily correspond with the input time steps. We only classify
        # one output variable (e.g. temperature).
        self.output_dim = output_dim

        # Amount of time steps we want to feed to the network before expecting
        # it to produce a classification of the future n time steps
        self.n_training_steps = n_training_steps

        # Amount of steps that Y will be predicting
        self.n_prediction_steps = n_prediction_steps

        self.hidden_dim = config["hidden_dim"]
        self.dropout_keep_p = config["dropout_keep_p"]
        self.n_layers = config["n_layers"]
        self.learning_rate = config["learning_rate"]
        self.horizon_decay = config["horizon_decay"]
        self.l2_loss = config["l2_alpha"]

        # Set dropout percentage to zero if we are visualizing
        if config["mode"] == 'vis':
            self.dropout_keep_p = 1.

        # Intialize the classifier (forecaster) network
        self._init_RNN()

        # Define an optimizer
        self.optimizer = tf.train.AdamOptimizer(
            learning_rate=self.learning_rate)

        # Train step
        self.train_step = self.optimizer.minimize(self.cost)

    def _init_RNN(self):
        # Input placeholder for single binary pair input
        self.X = tf.placeholder(dtype=tf.float32,shape=[None,
                                                        self.n_training_steps,
                                                        self.input_dim])

        # Output label
        self.Y = tf.placeholder(dtype=tf.float32, shape=[None,
                                                         self.n_prediction_steps,
                                                         self.output_dim])

        # Output weights
        self.w_out = tf.Variable(tf.random_normal(shape=[self.hidden_dim,
                                                         self.output_dim],
                                                  stddev=0.1))

        # Output biases
        self.b_out = tf.Variable(tf.constant(0, dtype=tf.float32))

        # Single RNN cell (LSTM)
        self.basic_cell = tf.nn.rnn_cell.BasicLSTMCell(self.hidden_dim,
                                                       state_is_tuple=True)

        # Add dropout
        self.basic_cell = tf.nn.rnn_cell.DropoutWrapper(self.basic_cell,
                                        output_keep_prob=self.dropout_keep_p)

        # Make the network multi-layer
        self.basic_cell = tf.nn.rnn_cell.MultiRNNCell([self.basic_cell] *
                                                      self.n_layers,
                                                      state_is_tuple=True)

        # Symbolic hidden outputs of the whole unraveled RNN network
        self.hidden_outputs, _ = tf.nn.dynamic_rnn(cell=self.basic_cell,
                                            inputs=self.X,
                                            dtype=tf.float32)

        # Transpose so that we have [steps, batch, hiddendim]
        self.hidden_outputs = tf.transpose(self.hidden_outputs, [1, 0, 2])

        # Take only the last n_prediction_steps outputs
        self.hidden_predictions = tf.gather(params=self.hidden_outputs,
                           indices=range(self.n_training_steps -
                                         self.n_prediction_steps,
                                         self.n_training_steps))

        # Unpack the predictions into a list of length *n_prediction_steps*
        # and shapes [batch, output_dim]
        self.unpacked_predictions = tf.unpack(self.hidden_predictions)
        self.unpacked_predictions_o = [(tf.matmul(up, self.w_out) +
            self.b_out) for up in self.unpacked_predictions]

        # Prediction operation
        self.packed_predictions = tf.pack(self.unpacked_predictions_o)
        self.predict = tf.transpose(self.packed_predictions,
                                               [1, 0, 2])

        self.cost = 0

        self.unpacked_Y = tf.unpack(tf.transpose(self.Y, [1, 0, 2]))

        for i in range(self.n_prediction_steps):
            self.cost += (self.horizon_decay ** i) * tf.square(
                self.unpacked_predictions_o[i] - self.unpacked_Y[i])

        self.cost = tf.reduce_mean(self.cost) + self.l2_loss * tf.nn.l2_loss(
            self.w_out)

        self.lossfun = tf.reduce_mean(self.cost)


