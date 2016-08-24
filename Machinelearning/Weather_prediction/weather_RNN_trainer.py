

"""
Script for training a sequence network on weather data
"""

from weather_dataset_server import DatasetServer
from sequence_RNN import WeatherNetwork

import numpy as np
import argparse
import json

def main():
    # Make input argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', required=True, type=str,
                        help='string: path to sqlite3 database containing '
                             'archived data')
    parser.add_argument('--model_path', required=True, type=str,
                        help='string: path where model ckpt will be saved')

    # Parse arguments
    args = parser.parse_args()

    # Read config file
    config_file = 'weather_prediction_config.json'
    config = json.load(config_file)

    # Create data server object
    dataserver = DatasetServer(args.input_path)

    # Create sequence RNN object
    seq_RNN = WeatherNetwork(input_dim = 20,
                             output_dim= 1,
                             n_time_steps= 12 * 18,
                             config=config)

    # Train the network
    train_network(dataserver, seq_RNN)

def train_network(dataserver, RNN):
    pass


if __name__ == '__main__':
    main()