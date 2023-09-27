import numpy as np
from scipy.stats import skew


def compute(image):
    channel_data = np.asarray(image).flatten()
    mean = np.mean(channel_data)
    std_dev = np.std(channel_data)

    # Calculate the skew expression
    skew = np.cbrt(np.sum((channel_data - mean) ** 3) / len(channel_data))

    # # Results
    # print("Mean :", mean)
    # print("Standard Deviation :", std_dev)
    # print("Skewness :", skew)

    return mean, std_dev, skew


# Colour Movements
def calculate_color_moments(image):

    if image.mode != 'RGB':
        return "Image not an RGB!! Cannot Process"

    newsize = (300, 100)
    image = image.resize(newsize)

    # Partition the image into a 10x10 grid
    num_rows, num_cols = 10, 10
    image_height = image.height
    image_width = image.width

    # row_indices = np.linspace(0, height, num_rows + 1, dtype=int)
    # col_indices = np.linspace(0, width, num_cols + 1, dtype=int)
    #
    # # an array to store the color moments for each grid cell
    # color_moments = np.zeros((num_rows, num_cols, 3, 3))

    tile_width = image_width // 10
    tile_height = image_height // 10
    tiles = [image.crop(box=(w, h, w + tile_width, h + tile_height))
             for h in range(0, image_height, tile_height)
             for w in range(0, image_width, tile_width)]

    results = []
    for tile in tiles:
        results.append(compute(tile.getchannel(0)) +
                       compute(tile.getchannel(1)) +
                       compute(tile.getchannel(2)))

    # Return as a unified flattened vector
    return np.array(results).flatten()
