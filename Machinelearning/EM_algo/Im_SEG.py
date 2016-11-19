import numpy as np
from PIL import Image
from sklearn.cluster import KMeans

# Image housekeeping #################################################

img_path = 'data/'

# Amount of images in the dataset
n_images = 50

# List which holds the read images in raw form
images = []  # [n_images, 3, m, n]
images_seg = []


# Read all images --
for i in range(n_images):
    # Create image file name
    img_idx = str(i).zfill(2)
    img_name = "{}hand_{}.png".format(img_path, img_idx)
    img_seg_name = "{}hand_{}_seg.png".format(img_path, img_idx)

    # Open image -------------
    img = Image.open(img_name).convert("RGB")
    img_arr = np.array(img, dtype=np.float32) / 255.0
    images.append(img_arr)

    # Open segmentation ------
    img_seg = Image.open(img_seg_name).convert("RGB")
    img_seg_arr = np.array(img_seg, dtype=np.float32) / 255.0
    images_seg.append(img_arr)

# Read the model initialization image
model_init_img = Image.open("{}model_init.png".format(img_path)).convert("RGB")
model_init_img_arr = np.array(model_init_img, dtype=np.float32) / 255.0

# Reshape images into 2 dimensional vector with shape (3, n_images*m*n)
images_arr = np.array(images, dtype=np.float32)
images_arr = np.transpose(images_arr, (3,1,2,0))
images_arr = np.reshape(images_arr,
                        (images_arr.shape[0], np.prod(images_arr.shape[1:])))


# Segment the image using K-means (2 clusters). ########################

# Amount of clusters
n_clusters = 2

# Make classifier
KMeans_cl = KMeans(n_clusters=2, n_init=3)

# Fit the data
KMeans_cl.fit(images_arr)


