import tensorflow as tf
import draft_tf_class
import os


path = '/home/shagas/Data/SW/Joda_storage/MachineLearning/FaceDetection' \
       '/Datasets/FamousPeople/p3/'

idx = 1
while True:

    src = os.path.join(path,'dic{}.jpg'.format(idx))
    dst = os.path.join(path,'face{}.jpg'.format(idx))

    if os.path.exists(src):
        os.rename(src,dst)
    else:
        break

    idx += 1

