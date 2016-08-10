
# Reads raw face images and saves extracted faces.

import cv2
import sys
import time
from os import listdir
from os.path import isfile, join
import os
from skimage import io

cascPath = 'haarcascade_frontalface_default.xml'
sourcePath = '/home/shagas/Data/SW/Machinelearning/FaceDetection/Datasets/faces/'

faceCascade = cv2.CascadeClassifier(cascPath)


for i in range(10):

    # Get images
    trainfoldername = sourcePath + "c" + str(i) + "/Train/"
    rawtrainfiles = [f for f in listdir(trainfoldername) if isfile(join(trainfoldername, f))]
    rawtrainimages = [io.imread(join(trainfoldername,im)) for im in rawtrainfiles]


    testfoldername = sourcePath + "c" + str(i) + "/Test/"
    rawtestfiles = [f for f in listdir(testfoldername) if isfile(join(testfoldername, f))]
    rawtestimages = [io.imread(join(testfoldername,im)) for im in rawtestfiles]

    for imgidx,img in enumerate(rawtrainimages):
        # Convert to GS
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

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
            face = img[y:y+h, x:x+w]

            if not os.path.exists(trainfoldername + "extractedfaces"):
                os.makedirs(trainfoldername + "extractedfaces")

            io.imsave(trainfoldername + "extractedfaces/face" + str(imgidx) + ".jpg", face)

    for imgidx, img in enumerate(rawtestimages):
        # Convert to GS
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

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
            face = img[y:y + h, x:x + w]

            if not os.path.exists(testfoldername + "extractedfaces"):
                os.makedirs(testfoldername + "extractedfaces")

            io.imsave(testfoldername + "extractedfaces/face" + str(imgidx) + ".jpg", face)


print "Done. "

