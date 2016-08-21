import numpy as np
import scipy.io
import os

input_path = '/home/shagas/Data/SW/Joda_storage/VGGweights/' \
          'imagenet-vgg-verydeep-16.mat'
output_path = '/home/shagas/Data/SW/Joda_storage/VGGweights/checker/'

layers16 = (
        'conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1',

        'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2',

        'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3',
        'relu3_3', 'pool3',

        'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3',
        'relu4_3', 'pool4',

        'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3',
        'relu5_3',

        'pool5', 'fc6', 'relu6', 'fc7', 'relu7', 'fc8', 'prob'
    )



layers19 = (
    'conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1',

    'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2',

    'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3',
    'relu3_3', 'conv3_4', 'relu3_4', 'pool3',

    'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3',
    'relu4_3', 'conv4_4', 'relu4_4', 'pool4',

    'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3',
    'relu5_3', 'conv5_4', 'relu5_4'
)

data = scipy.io.loadmat(input_path)
weights = data['layers'][0]

for i, name in enumerate(layers16):
    kind = name[:4]
    if kind == 'conv':
        kernels, bias = weights[i][0][0][0][0]
        # matconvnet: weights are [width, height, in_channels, out_channels]
        # tensorflow: weights are [height, width, in_channels, out_channels]
        kernels = np.transpose(kernels, (1, 0, 2, 3))
        #bias = bias.reshape(-1)

        print kernels.shape
        #print bias.shape
        #np.save(os.path.join(output_path, '{}_w'.format(name)), kernels)
        #np.save(os.path.join(output_path, '{}_b'.format(name)), bias)

    kind = name[:2]
    if kind == 'fc':
        w, b = weights[i][0][0][0][0]


        print w.shape

        # np.save(os.path.join(output_path, '{}_w'.format(name)), kernels)
        # np.save(os.path.join(output_path, '{}_b'.format(name)), bias)