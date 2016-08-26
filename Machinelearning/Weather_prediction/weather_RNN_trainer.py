

"""
Script for training a sequence network on weather data
"""

from weather_dataset_server import (DatasetServer,
                                    INPUT_DIM,
                                    DATASET_MEAN,
                                    DATASET_STD)

from sequence_RNN import WeatherNetwork

import numpy as np
import argparse
import json
import tensorflow as tf
import os
import matplotlib.pyplot as plt

N_TIME_STEPS = 216

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

    if config["mode"] == 'train':
        print 'Training network'
        # Train the network
        train_network(dataserver, seq_RNN, config, args.output_path)
    if config["mode"] == 'vis':
        print 'Visualizing network'
        visualize_prediction(dataserver, seq_RNN, config, args.output_path)


def visualize_prediction(dataserver, RNN, config, output_path):
    # Start tensorflow session
    session = tf.Session()

    # Initialize all variables
    session.run(tf.initialize_all_variables())

    # Weight saver
    saver = tf.train.Saver()

    # Path where the checkpoint will be saved to / loaded from
    model_path = os.path.join(output_path, "model.ckpt")

    # Load from checkpoint if exists
    if os.path.exists(model_path):
        saver.restore(session, model_path)
        print 'Checkpoint model.ckpt for prediction restored'
    else:
        'Model path not found, exiting'
        return

    # Make n visualizations
    for i in range(10):
        # Get data
        batch = dataserver.get_validation_batch(0, 1)

        # Get predictions from the network
        predictions = session.run([RNN.predict],
                                   feed_dict={RNN.X: batch[:, : N_TIME_STEPS]})

        # Plot
        x = np.arange(288)
        plt.plot(x, batch[0, :, 0]*DATASET_STD[0] + DATASET_MEAN[0])
        plt.plot(x, np.array([0]*216 + list(predictions[0][0, :,0]),
                             dtype=np.float32) * DATASET_STD[0] +
                 DATASET_MEAN[0] )
        plt.show()


def train_network(dataserver, RNN, config, output_path):

    # Amount of iterations
    iters = config['learning_iterations']

    # Start tensorflow session
    session = tf.Session()

    # Initialize all variables
    session.run(tf.initialize_all_variables())

    # Weight saver
    saver = tf.train.Saver()

    # Path where the checkpoint will be saved to / loaded from
    model_path = os.path.join(output_path, "model.ckpt")

    # Load from checkpoint if exists
    if os.path.exists(model_path) and config["load_from_checkpoint"]:
        saver.restore(session, model_path)
        print 'Checkpoint model.ckpt restored'

    # Train
    for i in range(iters):

        # Get data
        batch = dataserver.get_training_batch(0, config["batchsize"])

        _, batch_err = session.run([RNN.train_step, RNN.lossfun],
                    feed_dict = {RNN.X : batch[:, : N_TIME_STEPS],
                                 RNN.Y : batch[:, N_TIME_STEPS :,
                                         [0]] })

        print '[Iteration: {} / {}] MSE on training batch: {}'.format(
            i, iters, batch_err / config["batchsize"])

        if i % config["iters_per_eval"] == 0 and i > 0:
            # Get data
            batch = dataserver.get_validation_batch(0, config["batchsize"])

            validation_err = session.run([RNN.lossfun],
                              feed_dict={RNN.X: batch[:, : N_TIME_STEPS],
                                         RNN.Y: batch[:, N_TIME_STEPS:,
                                                [0]] })

            print 'VALIDATION: MSE on validation batch: {}'.format(
                validation_err[0] / config["batchsize"])


        if i % config["iters_per_checkpoint"] == 0:
            print 'Saving checkpoint'
            saver.save(session, model_path)


if __name__ == '__main__':
    main()