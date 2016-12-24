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

def makeConvData(PATH):
    dict = np.load(PATH).item()

    TrnDataDict = dict['trn']
    TstDataDict = dict['tst']

    return TrnDataDict, TstDataDict

def trainandEvalTask1(n_features, n_classes, TrnDataDict, TstDataDict):
    # Make perceptron which classifies individual letters
    cl_perc_unary = CharWisePerceptron(n_features, n_classes)
    cl_perc_unary.fit(TrnDataDict['X'], TrnDataDict['Y'], 1000)
    cl_perc_unary_errs = cl_perc_unary.evaluate(TstDataDict['X'], TstDataDict[
        'Y'])

    print 'Charwise perceptron evaluation: S_acc: {}, C_acc: {}'.format(
        cl_perc_unary_errs[0], cl_perc_unary_errs[1])

    return cl_perc_unary_errs

def trainandEvalTask2(n_features, n_classes, TrnDataDict, TstDataDict):
    # Make perceptron which classifies individual letters
    cl_perc_struct = StructuredPerceptron(n_features, n_classes)
    cl_perc_struct.fit(TrnDataDict['X'], TrnDataDict['Y'], 1000)

    exit()

    cl_perc_struct_errs = cl_perc_struct.evaluate(TstDataDict['X'], TstDataDict[
        'Y'])

    print 'Structured perceptron evaluation: S_acc: {}, C_acc: {}'.format(
        cl_perc_struct_errs[0], cl_perc_struct_errs[1])

    return cl_perc_struct_errs

def trainandEvalTask3(n_features, n_classes, TrnDataDict, TstDataDict):
    # Make perceptron which classifies individual letters
    cl_perc_seq= SeqPerceptron(n_features, n_classes)
    cl_perc_seq.fit(TrnDataDict['X'], TrnDataDict['Y'], 1000)

    cl_perc_seq_errs = cl_perc_seq.evaluate(TstDataDict['X'], TstDataDict[
        'Y'])

    print 'Structured perceptron evaluation: S_acc: {}, C_acc: {}'.format(
        cl_perc_seq_errs[0], cl_perc_seq_errs[1])

    return cl_perc_seq_errs

def trainandEvalKNN(n_features, n_classes, TrnDataDict, TstDataDict):
    # Make perceptron which classifies individual letters
    cl_knn = KNN(n_features, n_classes)
    cl_knn.fit(TrnDataDict['X'], TrnDataDict['Y'])

    cl_knn_errs = cl_knn.evaluate(TstDataDict['X'], TstDataDict[
        'Y'])

    print 'KNN  evaluation: S_acc: {}, C_acc: {}'.format(
        cl_knn_errs[0], cl_knn_errs[1])

    return cl_knn_errs

def trainandEvalCharwiseConvnet(n_features, n_classes, TrnDataDict, TstDataDict):
    # Make perceptron which classifies individual letters
    cl_net = SeqPerceptron(n_features, n_classes)
    cl_net.fit(TrnDataDict['X'], TrnDataDict['Y'], 1000)

    cl_perc_seq_errs = cl_net.evaluate(TstDataDict['X'], TstDataDict[
        'Y'])

    print 'Structured perceptron evaluation: S_acc: {}, C_acc: {}'.format(
        cl_perc_seq_errs[0], cl_perc_seq_errs[1])

    return cl_perc_seq_errs


def main():

    # .mat data file path
    MAT_DATA_PATH = '/home/shagas/Data/SW/Joda_research/Machinelearning/' \
                    'SO_classification/data/matlab/ocr_names.mat'

    CONV_DATA_PATH = 'convfeatures.npy'

    # Get data in a proper format
    #TrnDataDict, TstDataDict = makeData(MAT_DATA_PATH)
    TrnDataDict, TstDataDict = makeConvData(CONV_DATA_PATH)

    # Required parameters
    n_classes = 26
    n_features = TrnDataDict['X'][0].shape[0]

    # Peceptrons
    #trainandEvalTask1(n_features, n_classes, TrnDataDict, TstDataDict)
    #trainandEvalTask2(n_features, n_classes, TrnDataDict, TstDataDict)
    #trainandEvalTask3(n_features, n_classes, TrnDataDict, TstDataDict)

    trainandEvalKNN(n_features, n_classes, TrnDataDict, TstDataDict)


if __name__ == "__main__":
    main()
