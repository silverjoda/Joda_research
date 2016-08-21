
"""
Description:
This script maps given input (latitude, longitude) coordinate pairs into
points on a mercator projection of earth.
"""

from skimage.io import imread
import matplotlib.pyplot as plt


def main():
    # Input coordinates in degrees that we wish to show on our map image
    input_coordinates = (0., 0.)

    # Name of image file
    fname = '/home/...'

    # Read image as rgb
    earth_mercator_img = imread(fname)

    # Pixel coordinates of origin (equator & prime meridian intersection)
    origin = (500,500)

    # Amount of pixels on the image that correspond to one meridian
    meridian_pixel_length = 30

    # Amount of pixels on the image that correspond to the FIRST parallel above
    # or below the equator
    first_parallel_pixel_length = 35




    # Plot the image
    plt.imshow(earth_mercator_img)
    plt.show()


if __name__ == 'main':
    main()





