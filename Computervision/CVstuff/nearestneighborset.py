import cv2
import sys
import time
from os import listdir
from os.path import isfile, join
from skimage import io

cascPath = 'haarcascade_frontalface_default.xml'
sourcePath = '/home/shagas/Data/SW/Machinelearning/FaceDetection/Datasets/faces/'
labelPath = '/home/shagas/Data/SW/Machinelearning/FaceDetection/Datasets/faces/extractedfaces/'

faceCascade = cv2.CascadeClassifier(cascPath)

print "Reading filenames..."
# Read all images into memory
onlyfiles = [f for f in listdir(sourcePath) if isfile(join(sourcePath, f))]
print "Found " + str(len(onlyfiles)) + " images."

print "Starting detection..."

