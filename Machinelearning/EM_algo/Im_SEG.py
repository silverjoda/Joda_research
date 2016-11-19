import numpy as np
from PIL import Image
from sklearn.cluster import KMeans
from sklearn import mixture

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

# Image matrix of shape (n_images, m, n, 3)
images_mat = np.array(images, dtype=np.float32)
images_seg_mat = np.array(images_seg, dtype=np.float32)

# Read the model initialization image
model_init_img = Image.open("{}model_init.png".format(img_path)).convert('L')
model_init_img_mat = np.array(model_init_img, dtype=np.float32) / 255.0

# Find image shapes
img_shape = [images_mat.shape[1],images_mat.shape[2]]
model_pic_shape = [model_init_img_mat.shape[0],model_init_img_mat.shape[1]]

assert img_shape == model_pic_shape

# Find indeces where image is foreground and background
foregnd_idxs = np.reshape(model_init_img_mat, np.prod(img_shape)) > 0.7
backgnd_idxs = np.reshape(model_init_img_mat, np.prod(img_shape)) < 0.2

# Find mean RGB pixel vectors for background and foreground
flattened_images = np.reshape(np.transpose(images_mat,(1,2,0,3)),(np.prod(images_mat.shape[0:2]),images_mat.shape[2],images_mat.shape[3]))

mean_foregnd_vec = np.mean(flattened_images[:,foregnd_idxs], keepdims=2)
mean_backgnd_vec = np.mean(flattened_images[:,backgnd_idxs], keepdims=2)


print mean_foregnd_vec
print mean_backgnd_vec

exit()

# Segment the image using K-means (2 clusters). ########################

# Reshape images into 2 dimensional vector with shape (n_images*m*n, 3)
images_trsp = np.transpose(images_mat, (3, 1, 2, 0))
images_arr = np.reshape(images_trsp,
                        (images_trsp.shape[0], np.prod(images_trsp.shape[1:])))

# Vector with shape (n_images*m*n, 3)
images_arr = np.transpose(images_arr)

# Amount of clusters
n_clusters = 2

# Make classifier
KMeans_cl = KMeans(n_clusters=2, n_init=1, n_jobs=-1)

# Fit the data
KMeans_cl.fit(images_arr)


# Segment the image using Gaussian mixture (2 components) ####################

# Reshape images into 2 dimensional vector with shape (n_images*m*n, 3)
images_trsp = np.transpose(images_mat, (3, 1, 2, 0))
images_arr = np.reshape(images_trsp,
                        (images_trsp.shape[0], np.prod(images_trsp.shape[1:])))

# Vector with shape (n_images*m*n, 3)
images_arr = np.transpose(images_arr)

# Amount of components
n_components = 2

# Make classifier
gmm = mixture.GMM(n_components=2)

# Fit the data
gmm.fit(images_arr)
