import math

import numpy as np


def compute(image):
    img = np.array(image)
    mag = []
    theta = []
    for i in range(image.height):
        magnitudeArray = []
        angleArray = []
        for j in range(image.width):
            # Condition for axis 0
            if j - 1 <= 0 or j + 1 >= image.width:
                """
                    Ignoring the egdes
                """
                Gx=0
                """
                    Previously for edges i was subtracting with 0
                """
                # if j - 1 <= 0:
                #     # Condition if first element
                #     Gx = img[i][j + 1] - 0
                # elif j + 1 >= len(img[0]):
                #     Gx = 0 - img[i][j - 1]
            # Condition for first element
            else:
                # Applying mask <-1, 0, 1>
                Gx = int(img[i][j + 1]) - int(img[i][j - 1])

            # Condition for axis 1
            if i - 1 <= 0 or i + 1 >= image.height:
                """
                    Ignoring the egdes
                """
                Gy=0
                """
                    Previously for edges i was subtracting with 0
                """
                # if i - 1 <= 0:
                #     Gy = 0 - img[i + 1][j]
                # elif i + 1 >= 128:
                #     Gy = img[i - 1][j] - 0
            else:
                # Applying mask <-1, 0, 1> transpose
                Gy = int(img[i - 1][j]) - int(img[i + 1][j])

            # Calculating magnitude
            magnitude = math.sqrt(pow(Gx, 2) + pow(Gy, 2))
            mag.append(round(magnitude, 9))

            # Calculating angle
            if Gx == 0:
                angle = np.degrees(0.0)
            else:
                angle = np.degrees(360 + np.arctan2(Gy, Gx)) % 360
            theta.append(round(angle, 9))
        # mag.append(magnitudeArray)
        # theta.append(angleArray)

    """
        The above calculation will give the magnitude and angle for each of the pixels in 300*100 grid
        Now we will save them in 9 bins
    """
    bin_size = 40
    hog_bins = [0] * (360 // bin_size) # Create an array of 9 bin size
    for index in range(0, len(mag)):
        angle = theta[index]
        magnitude = mag[index]
        lower_bin = math.floor(angle/bin_size)
        upper_bin = lower_bin+1

        mag_lower_bin = (upper_bin-angle/bin_size)*magnitude
        mag_upper_bin = magnitude - mag_lower_bin

        hog_bins[lower_bin] += mag_lower_bin
        hog_bins[upper_bin % len(hog_bins)] += mag_upper_bin #the upper index should be cycled back to 0 ig the index is more than 9

    return hog_bins


def calculate_hog(image):
    if image.mode != 'RGB':
        return "Image not an RGB!! Cannot Process"

    newsize = (300, 100)
    image = image.resize(newsize)

    # Convert the image to grayscale
    gray_img = image.convert('L')

    # Partition the image into a 10x10 grid
    num_rows, num_cols = 10, 10
    image_height = gray_img.height
    image_width = gray_img.width

    tile_width = image_width // 10
    tile_height = image_height // 10
    tiles = [gray_img.crop(box=(w, h, w + tile_width, h + tile_height))
             for h in range(0, image_height, tile_height)
             for w in range(0, image_width, tile_width)]

    results = []
    for tile in tiles:
        results.append(compute(tile.getchannel(0)))

    # Return as a unified flattened vector
    return np.array(results).flatten()
