# Reads images from one single folder and saves extracted faces to another folder.

import cv2
import sys
import time
from os import listdir
from os.path import isfile, join
from skimage import io

cascPath = 'haarcascade_frontalface_default.xml'
sourcePath = '/home/shagas/Data/SW/Machinelearning/FaceDetection/Datasets/faces/unlabeled/'
labelPath = '/home/shagas/Data/SW/Machinelearning/FaceDetection/Datasets/faces/extractedfaces/'

faceCascade = cv2.CascadeClassifier(cascPath)

print "Reading filenames..."
# Read all images into memory
onlyfiles = [f for f in listdir(sourcePath) if isfile(join(sourcePath, f))]
print "Found " + str(len(onlyfiles)) + " images."

print "Starting detection..."

# Image index
i = 0

for file in onlyfiles:

    # Get image
    frame = io.imread(sourcePath + file)

    # Convert to GS
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Perform face detection
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(150, 150),
        flags=cv2.cv.CV_HAAR_SCALE_IMAGE
    )

    if not len(faces) == 1:
        continue

    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        face = frame[y:y+h, x:x+w]

    io.imsave(labelPath + "img" + str(i) + ".jpg", face)

    i += 1

print "Done. "