from scipy.io import loadmat
import numpy as np
from convnets import *
import os.path

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


def trainandEvalCharwiseConvnet(m, n, n_classes, TrnDataDict, TstDataDict):

    # Make Convnet classifier which classifies individual letters
    cl_conv = CharWiseConvnet(m, n, n_classes)


    if os.path.isfile('convnet.tfl'):
        cl_conv.model.load('convnet.tfl')
        print 'Loaded pretrained model. '
    else:
        print "Training model: "
        cl_conv.fit(TrnDataDict['img'], TrnDataDict['Y'],
                    TstDataDict['img'], TstDataDict['Y'])



    char_conv_errors = cl_conv.evaluate(TstDataDict['img'], TstDataDict['Y'])

    print 'Conv net charwise evaluation: S_acc: {}, C_acc: {}'.format(
        char_conv_errors[0], char_conv_errors[1])

    return cl_conv

def main():

    # .mat data file path
    MAT_DATA_PATH = '/home/shagas/Data/SW/Joda_research/Machinelearning/' \
                    'SO_classification/data/matlab/ocr_names.mat'

    # Get data in a proper format
    TrnDataDict, TstDataDict = makeData(MAT_DATA_PATH)

    # Required parameters
    n_classes = 26
    m = 16
    n = 8

    # Convnet
    cl_conv = trainandEvalCharwiseConvnet(m, n, n_classes, TrnDataDict,
                                        TstDataDict)


    cl_conv.makedataset(TrnDataDict['img'], TrnDataDict['Y'],
                        TstDataDict['img'], TstDataDict['Y'])


if __name__ == "__main__":
    main()
