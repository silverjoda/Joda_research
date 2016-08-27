import argparse
from os.path import join

import numpy as np
from skimage.io import imread
from skimage.transform import resize

from Machinelearning.VGG_net.VGG_feature_extractor import (
    VGG_feature_extractor, IMAGENET_MEAN)

from sklearn.neighbors import KNeighborsClassifier

def main():

    # Make input argument parser
    parser = argparse.ArgumentParser()

    parser.add_argument('--input_path', required=True, type=str,
                        help='string: path to faces folder')
    parser.add_argument('--model_path', required=True, type=str,
                        help='string: path to model weights')

    # Parse arguments
    args = parser.parse_args()

    # Get our sets
    training_set, test_set = get_dataset(args.input_path)

    # Make feature extractor network
    extractor_network = VGG_feature_extractor(args.model_path)

    # Transform sets into feature vectors
    training_set_fv = []

    # Extract training set features
    for t in training_set:
        training_set_fv.append(extract_features(t, extractor_network))

    # Extract test set features
    test_set_fv = extract_features(test_set, extractor_network)

    # Make classifier
    knn_classifier = KNeighborsClassifier(n_neighbors=3)

    # Fit data

    # Classify test data

    # Save test images to results folder for visual evaluation

def extract_features(set, network):
    """
    Turn given set of images into feature vectors using the network
    Parameters
    ----------
    set: list, ndarray of images
    network obj, feature extractor network

    Returns: list, features
    -------

    """

    feature_set = []

    for s in set:
        s -= np.array(IMAGENET_MEAN, np.float32)
        feature_set.append(network.VGG_extract_features(s))

    return feature_set


def get_all_images_from_dir(path):
    """
    Read all images from directory and resize them to (224,224,3)
    Parameters
    ----------
    path: str, string

    Returns list of images
    -------

    """

    img_list = []
    file_list = os.listdir(path)

    for f in file_list:

        # Read image
        img = imread(f)

        # Turn to float32
        img = img.astype(np.float32)

        # Resize image
        img = resize(img, (224,224,3))

        # Add image to list
        img_list.append(img)

    return img_list


def get_dataset(path):
    """
    Returns the training and test set lists.
    Parameters
    ----------
    path: str, path to dataset

    Returns training and test set lists. Training set is list of lists of
    images in their categories and. Test set is simple list.
    -------

    """

    # Paths
    test_path = join(path, 'test')
    train_path = join(path, 'train')

    # Get all classes directories
    training_class_paths = os.listdir(train_path)

    training_set = []

    for c in training_class_paths:
        training_set.append(get_all_images_from_dir(join(train_path, c)))

    test_set = get_all_images_from_dir(test_path)

    return training_set, test_set

if __name__ == '__main__':
    main()