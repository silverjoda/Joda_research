
# Reads face images from classes and makes a matrix set which is then saved


from os import listdir
from os.path import isfile, join

from skimage import io

sourcePath = '/home/shagas/Data/SW/Machinelearning/FaceDetection/Datasets/faces/'

""" Make ANN object """
import VGGTransformer
network = VGGTransformer.FStransformer()

classvectors = []

for i in range(10):

    """ Get train and test images """
    trainfoldername = sourcePath + "c" + str(i) + "/Train/extractedfaces/"
    rawtrainfiles = [f for f in listdir(trainfoldername) if isfile(join(trainfoldername, f))]
    trainimages = [io.imread(join(trainfoldername, im)) for im in rawtrainfiles]

    testfoldername = sourcePath + "c" + str(i) + "/Test/extractedfaces/"
    rawtestfiles = [f for f in listdir(testfoldername) if isfile(join(testfoldername, f))]
    testimages = [io.imread(join(testfoldername, im)) for im in rawtestfiles]

    """ Lists in which we will store training and test vectors """
    trainvectors = []
    testvectors = []

    """ Perform classification """
    for img in trainimages:
        trainvectors.append(network.transform(img))

    for img in testimages:
        testvectors.append(network.transform(img))

    """ Append to main list """
    classvectors.append([trainvectors,testvectors])

print "Done. "

