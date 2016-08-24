import tensorflow as tf

# TODO: Check all the shapes on the RNN and correctly transpose if necessary


class WeatherNetwork:
    """
    Class of a recurrent neural network in TensorFlow which is built to predict
    immediate weather
    """
    def __init__(self, input_dim, output_dim, n_time_steps, config):

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
        self.n_time_steps = n_time_steps

        self.hidden_dim = config["hidden_dim"]
        self.dropout_keep_p = config["dropout_keep_p"]
        self.n_layers = config["n_layers"]
        self.learning_rate = config["learning_rate"]

        # Intialize the classifier (forecaster) network
        self.init_RNN()

        # Define an optimizer
        self.optimizer = tf.train.AdamOptimizer(
            learning_rate=self.learning_rate)

        # Train step
        self.train_step = self.optimizer.minimize(self.cost)

    def init_RNN(self):
        # Input placeholder for single binary pair input
        self.X = tf.placeholder(dtype=tf.float32,
                           shape=[None, self.n_time_steps, self.input_dim])

        # Output label
        self.Y = tf.placeholder(dtype=tf.float32, shape=[None, self.output_dim])

        # Output weights
        self.w_out = tf.Variable(tf.random_normal(shape=[self.hidden_dim,
                                                         self.output_dim],
                                                  stddev=0.1))

        # Output biases
        self.b_out = tf.Variable(tf.constant(0, dtype=tf.float32))

        # Single RNN cell (LSTM)
        self.basic_cell = tf.nn.rnn_cell.BasicLSTMCell(self.hidden_dim)

        # Add dropout
        self.basic_cell = tf.nn.rnn_cell.DropoutWrapper(self.basic_cell,
                                        output_keep_prob=self.dropout_keep_p)

        # Make the network multi-layer
        self.basic_cell = tf.nn.rnn_cell.MultiRNNCell([self.basic_cell] *
                                                      self.n_layers)

        # Symbolic hidden outputs of the whole unraveled RNN network
        self.hidden_outputs, _ = tf.nn.dynamic_rnn(cell=self.basic_cell,
                                            inputs=self.X,
                                            dtype=tf.float32)

        self.hidden_outputs = tf.transpose(self.hidden_outputs, [1, 0, 2])

        # Take only the last hidden output
        self.hidden_output = tf.gather(params=self.hidden_outputs,
                           indices=int(self.hidden_outputs.get_shape()[0]) - 1)

        self.prediction = tf.nn.tanh(tf.matmul(self.hidden_output, self.w_out) +
                                 self.b_out)

        self.cost = tf.square(self.prediction, self.Y)

