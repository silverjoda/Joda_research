import numpy as np
from PIL import Image
from sklearn.cluster import KMeans
from sklearn import mixture






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
        img_seg = Image.open(img_seg_name).convert("RGB")
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
    fg_mean = np.mean(image[:m,0,:], axis=(0,1))

    # compute mean in a small middle square of the image
    bg_mean = np.mean(image[2*m/5:3*m/5, 2*n/5:3*n/5, :], axis=(0, 1))

    assert len(fg_mean) == 3, "Mean error! Len is not 3"
    assert len(bg_mean) == 3, "Mean error! Len is not 3"

    return fg_mean, bg_mean

def segment_by_k_means(image):

    # Make classifier
    KMeans_cl = KMeans(n_clusters=2, n_init=1, n_jobs=-1)

    # Fit the data
    KMeans_cl.fit(image)

    # Make the segmentation on the image and return it
    return KMeans_cl.predict(image)

def segment_by_GMM(image):

    # Make classifier
    gmm = mixture.GMM(n_components=2)

    # Fit the data
    gmm.fit(image)

    # Make the segmentation on the image and return it
    gmm.predict(image)

def plot_img_and_seg(image, seg, gt):
    assert len(image.shape) == 3, "Failed to plot, image is not RGB!"
    assert len(seg.shape) == len(gt.shape) == 2 , "Failed to plot, " \
                                                  "Segmentation is not B/W!"



def main():

    # Path to images
    img_path = 'data/'

    # Amount of images in the dataset
    n_images = 50

    # Load images
    images_mat, images_seg_mat, model_init_img = load_images(img_path, n_images)




if __name__ == "__main__":
    main()