

"""
Script for training a sequence network on weather data
"""

from weather_dataset_server import DatasetServer, INPUT_DIM
from sequence_RNN import WeatherNetwork

import numpy as np
import argparse
import json
import tensorflow as tf
import os

N_TIME_STEPS = 216

INDEX_OF_OUTSIDE_TEMP = 5

def main():
    # Make input argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', required=True, type=str,
                        help='string: path to sqlite3 database containing '
                             'archived data')
    parser.add_argument('--output_path', required=True, type=str,
                        help='string: path where the output will be saved')

    # Parse arguments
    args = parser.parse_args()

    # Read config file
    with open('weather_prediction_config.json') as data_file:
        config = json.load(data_file)

    # Make the results folder in storage if doesn't exist
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    print 'Loading dataserver'
    # Create data server object
    dataserver = DatasetServer(args.input_path)

    print 'Loading Network'
    # Create sequence RNN object
    seq_RNN = WeatherNetwork(input_dim = INPUT_DIM,
                             output_dim= 1,
                             n_training_steps= N_TIME_STEPS,
                             n_prediction_steps = 72,
                             config=config)

    print 'Training network'
    # Train the network
    train_network(dataserver, seq_RNN, config)

def train_network(dataserver, RNN, config):

    # Amount of iterations
    iters = config['learning_iterations']

    # Start tensorflow session
    session = tf.Session()

    # Initialize all variables
    session.run(tf.initialize_all_variables())


    # Train
    for i in range(iters):

        # Get data
        batch = dataserver.get_training_batch(0, config["batchsize"])

        _, batch_err = session.run([RNN.train_step, RNN.cost],
                    feed_dict = {RNN.X : batch[:, : N_TIME_STEPS],
                                 RNN.Y : batch[:, N_TIME_STEPS :,
                                         [INDEX_OF_OUTSIDE_TEMP]] })

        print 'MSE on training batch: {}'.format(batch_err /
                                                 config["batchsize"])

        if i % config["iters_per_eval"] == 0 and i > 0:
            # Get data
            batch = dataserver.get_validation_batch(0, config["batchsize"])

            validation_err = session.run([RNN.cost],
                              feed_dict={RNN.X: batch[:, : N_TIME_STEPS],
                                         RNN.Y: batch[:, N_TIME_STEPS:,
                                                [INDEX_OF_OUTSIDE_TEMP]] })

            print 'MSE on validation batch: {}'.format(validation_err)

if __name__ == '__main__':
    main()