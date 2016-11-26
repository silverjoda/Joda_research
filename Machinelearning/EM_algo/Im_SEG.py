import numpy as np
from PIL import Image
from sklearn.cluster import KMeans
from sklearn import mixture
import matplotlib.pyplot as plt


class EM_classif:
    def __init__(self):
        pass

    def fit(self, images, shape_model):
        pass

    def evaluate(self, images, gt):
        pass

    def predict(self):
        pass


def load_images(path, n):
    # List which holds the read images in raw form
    images = []  # [n_images, 3, m, n]
    images_seg = []

    #  =========== Read all images ==============

    for i in range(n):
        # Create image file name
        img_idx = str(i).zfill(2)
        img_name = "{}hand_{}.png".format(path, img_idx)
        img_seg_name = "{}hand_{}_seg.png".format(path, img_idx)

        # Open image -------------
        img = Image.open(img_name).convert("RGB")
        img_arr = np.array(img, dtype=np.float32) / 255.0
        images.append(img_arr)

        # Open segmentation ------
        img_seg = Image.open(img_seg_name).convert('L')
        img_seg_arr = np.array(img_seg, dtype=np.float32) / 255.0
        images_seg.append(img_seg_arr)

    # Read the model initialization image
    model_init_img = Image.open("{}model_init.png".format(path)).convert(
        'L')
    model_init_img_mat = np.array(model_init_img, dtype=np.float32) / 255.0

    # Image matrix of shape (n_images, m, n, c)
    images_mat = np.array(images, dtype=np.float32)
    images_seg_mat = np.array(images_seg, dtype=np.float32)

    #  ==========================================

    return [images_mat, images_seg_mat, model_init_img_mat]

def get_FG_BG_RGB_means(image):

    assert len(image.shape) == 3, "Image must be RGB"
    m,n,c = image.shape

    # Compute mean along the first row of the image
    fg_mean = np.mean(image[:m,0:1,:], axis=(0,1))

    # compute mean in a small middle square of the image
    bg_mean = np.mean(image[2*m/5:3*m/5, 2*n/5:3*n/5, :], axis=(0, 1))

    assert len(fg_mean) == 3, "Mean error! Len is not 3"
    assert len(bg_mean) == 3, "Mean error! Len is not 3"

    return fg_mean, bg_mean

def segment_by_k_means(image):

    assert len(image.shape) == 3, "Image must be RGB"
    m, n, c = image.shape

    # Get means
    init_mean = get_FG_BG_RGB_means(image)

    pixels_arr = np.reshape(image, (m*n, c))

    # Make classifier
    KMeans_cl = KMeans(n_clusters=2, n_init=3, init=np.array(init_mean))

    # Fit the data
    KMeans_cl.fit(pixels_arr)

    # Make the segmentation on the image and return it
    prediction = KMeans_cl.predict(pixels_arr)

    # Reshape pixels back into image
    pred_img = np.reshape(prediction, (m ,n))

    return pred_img

def segment_by_GMM(image):
    assert len(image.shape) == 3, "Image must be RGB"
    m, n, c = image.shape

    # Get means
    init_mean = get_FG_BG_RGB_means(image)

    pixels_arr = np.reshape(image, (m * n, c))

    # Make classifier
    gmm = mixture.GMM(n_components=2, n_init=1)

    # Fit the data
    gmm.fit(pixels_arr)

    # Make the segmentation on the image and return it
    prediction = gmm.predict(pixels_arr)

    # Reshape pixels back into image
    pred_img = np.reshape(prediction, (m, n))

    return pred_img

def plot_img_and_seg(image, seg, gt):
    assert len(image.shape) == 3, "Failed to plot, image is not RGB!"
    assert len(seg.shape) == len(gt.shape) == 2 , "Failed to plot, " \
                                                  "Segmentation is not B/W!"

    f = plt.figure()
    f.add_subplot(1, 3, 1)
    plt.imshow(image)
    f.add_subplot(1, 3, 2)
    plt.imshow(seg, cmap='gray')
    f.add_subplot(1, 3, 3)
    plt.imshow(gt, cmap='gray')

    plt.show()

def main():

    # Path to images
    img_path = 'data/'

    # Amount of images in the dataset
    n_images = 50

    # Load images
    images_mat, images_seg_mat, model_init_img = load_images(img_path, n_images)

    # Random image index
    rnd_idx = np.random.randint(0,49)

    # Make custom EM classifier
    em_cl = EM_classif()

    # Train the classifier
    em_cl.fit(images_mat, model_init_img)

    # Predict single image
    em_seg = em_cl.predict(images_mat[rnd_idx])

    # Plot image
    plot_img_and_seg(images_mat[rnd_idx], em_seg, images_seg_mat[rnd_idx])





if __name__ == "__main__":
    main()