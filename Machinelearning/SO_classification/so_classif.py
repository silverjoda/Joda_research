from scipy.io import loadmat
import numpy as np
from perceptrons import *

def makeData(PATH):
    """

    Parameters
    ----------
    PATH: str, path to .mat file

    Returns [dict,dict], with keys 'X','Y','img'
    -------

    """
    mat_data = loadmat(PATH)
    TrnData = mat_data['TrnData']
    TstData = mat_data['TstData']

    # Training and test data dictionaries
    TrnDataDict = {}
    TstDataDict = {}

    vars = ['X','Y','img']

    for v in vars:
        TrnDataDict[v] = TrnData[v][0].T
        TstDataDict[v] = TstData[v][0].T

    return [TrnDataDict,TstDataDict]


def main():

    # .mat data file path
    MAT_DATA_PATH = '/home/shagas/Data/SW/Joda_research/Machinelearning/' \
                    'SO_classification/data/matlab/ocr_names.mat'

    # Get data in a proper format
    TrnDataDict, TstDataDict = makeData(MAT_DATA_PATH)

    n_classes = 26
    n_features = TrnDataDict['X'][0].shape[0]

    # Make perceptron which classifies individual letters
    cl_perc_unary = CharWisePerceptron(n_features, n_classes)
    cl_perc_unary.fit(TrnDataDict['X'], TrnDataDict['Y'], 10000)
    err = cl_perc_unary.evaluate(TstDataDict['X'], TstDataDict['Y'])

    print 'Single char error on charwise perceptron: {}'.format(err)



if __name__ == "__main__":
    main()