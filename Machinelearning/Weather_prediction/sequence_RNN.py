import tensorflow as tf

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
        self.unpacked_predictions = [tf.nn.tanh(tf.matmul(p, self.w_out) +
            self.b_out) for p in self.unpacked_predictions]

        # Prediction operation
        self.packed_predictions = tf.pack(self.unpacked_predictions)
        self.predict = tf.transpose(self.packed_predictions,
                                               [1, 0, 2])

        self.cost = 0

        self.unpacked_Y = tf.unpack(tf.transpose(self.Y, [1, 0, 2]))

        for i in range(self.n_prediction_steps):
            self.cost += (self.horizon_decay ** i) * tf.square(
                self.unpacked_predictions[i] - self.unpacked_Y[i])

        self.cost = tf.reduce_mean(self.cost) + self.l2_loss * tf.nn.l2_loss(
            self.w_out)

        self.lossfun = tf.reduce_mean(self.cost)


