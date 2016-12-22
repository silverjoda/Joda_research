from scipy.io import loadmat
import numpy as np


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




if __name__ == "__main__":
    main()