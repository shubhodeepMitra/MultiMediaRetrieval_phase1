import numpy as np
from numpy import argsort
from torchvision import datasets, models, transforms
from scipy.spatial.distance import cosine
from scipy.stats import pearsonr
from scipy.spatial import distance_matrix
from scipy.spatial.distance import cityblock, cosine, euclidean
import matplotlib.pyplot as plt


data_dir = '/Users/shubhodeepmitra/Documents/dev/cse_515/phase1/dataset'

# Caltech 101 dataset
caltech101_dataset = datasets.Caltech101(
    root=data_dir,
    download=True,
)


def res_show(result_img_index, distance, target_id, k):
    result = []
    target_image, _ = caltech101_dataset[target_id]
    for id in result_img_index:
        image, _ = caltech101_dataset[id]
        result.append(image)

    # Plot the target image and k closest images
    plt.figure(figsize=(15, 5))
    plt.subplot(1, k + 1, 1)
    plt.title(f"Target Image\n: {target_id}")
    plt.imshow(target_image)
    plt.axis('off')

    for i, image in enumerate(result):
        plt.subplot(1, k + 1, i + 2)
        plt.title(f"Dist {round(distance[i], 4)}\nImageID: {result_img_index[i]}")
        plt.imshow(image)
        plt.axis('off')

    plt.tight_layout()
    plt.show()

def res_show_all(result_img_index, distance, feature_name, target_id, k):
    result = []
    target_image, _ = caltech101_dataset[target_id]

    for row in range(len(result_img_index)):
        # Create a subplot for the row of images
        plt.subplot(len(result_img_index), k + 1, row * (k + 1) + 1)
        plt.title(f"{feature_name[row]}\nTarget Image\n: {target_id}")
        plt.imshow(target_image)
        plt.axis('off')

        for i, (imageid) in enumerate(result_img_index[row]):
            image, _ = caltech101_dataset[imageid]
            plt.subplot(len(result_img_index), k + 1, row * (k + 1) + i + 2)
            plt.title(f"Dist {round(distance[row][i], 4)}\nImageID: {result_img_index[row][i]}")
            plt.imshow(image)
            plt.axis('off')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.suptitle("Caltech 101 Images", fontsize=16)
    plt.show()

    pass


def compute_similarity(search, target, k, distance_func):
    distance = []
    target = target.flatten()

    for row in search:
        dist = 0
        if distance_func == "pearson":
            dist, _ = pearsonr(row[1:], target[1:])
        elif distance_func == "manhattan":
            dist = cityblock(row[1:], target[1:])
        elif distance_func == 'cosine':
            dist = cosine(row[1:], target[1:])
        elif distance_func == 'euclidean':
            dist = euclidean(row[1:], target[1:])
            dist = 1 / (1 + dist)

        distance.append(dist)

    res_index = np.argsort(distance, axis=0)[-k-3:][::-1]
    res_index = res_index[1:k+1]
    distance.sort(reverse=True)
    distance = distance[1:k+1]
    #print(distance)
    result_img_index = search[res_index, 0]
    result_img_index = result_img_index.astype(np.int64)

    print("Image IDs")
    print(result_img_index)

    # res_show(result_img_index, distance, int(target[0]), k)

    return result_img_index, distance



if __name__ == "__main__":
    input_id = int(input("Enter Image ID: "))
    img, label = caltech101_dataset[input_id]
    if img.mode != 'RGB':
        print("Image not an RGB!! Cannot Process")
        exit(0)

    k = int(input("Enter the number of similar images(k):"))

    moments = np.load('../results/caltech101_moments.npy')
    hog = np.load('../results/caltech101_hog.npy')
    avgpool = np.load('../results/caltech101_resnet_avgpool_1024.npy')
    layer3 = np.load('../results/caltech101_resnet_layer3_1024.npy')
    fc = np.load('../results/caltech101_resnet_fc_1000.npy')

    # fetch the input row for each feature
    target_moment_row = moments[moments[:, 0] == input_id]
    target_hog_row = hog[hog[:, 0] == input_id]
    target_avg_row = avgpool[avgpool[:, 0] == input_id]
    target_l3_row = layer3[layer3[:, 0] == input_id]
    target_fc_row = fc[fc[:, 0] == input_id]

    # search_space = [moments, hog, avgpool, layer3, fc]
    # target_space = [target_moment_row, target_hog_row, target_avg_row, target_l3_row, target_fc_row]
    #
    # results = []
    # for feature_index in range(0, len(search_space)):
    #     results.append(compute_similarity(search_space[feature_index], target_space[feature_index], k, "pearson"))

    distance_all = []
    result_index_all = []

    print("{} Closest Image Id for Color Moment Feature".format(k))
    index, dist = compute_similarity(moments, target_moment_row, k, "euclidean")
    for i in range(len(index)):
        print(f'{index[i]}:{dist[i]}', end=' ')
    result_index_all.append(index)
    distance_all.append(dist)

    print("{} Closest Image Index for HOG Feature".format(k))
    index, dist = compute_similarity(hog, target_hog_row, k, "pearson")
    for i in range(len(index)):
        print(f'{index[i]}:{dist[i]}', end=' ')
    result_index_all.append(index)
    distance_all.append(dist)

    print("\n{} Closest Image Index for AVGPOOL Feature".format(k))
    index, dist = compute_similarity(avgpool, target_avg_row, k, "pearson")
    for i in range(len(index)):
        print(f'{index[i]}:{dist[i]}', end=' ')
    result_index_all.append(index)
    distance_all.append(dist)

    print("\n{} Closest Image Index for L3 Feature".format(k))
    index, dist = compute_similarity(layer3, target_l3_row, k, "pearson")
    for i in range(len(index)):
        print(f'{index[i]}:{dist[i]}', end=' ')
    result_index_all.append(index)
    distance_all.append(dist)

    print("\n{} Closest Image Index for FC Feature".format(k))
    index, dist = compute_similarity(fc, target_fc_row, k, "pearson")
    for i in range(len(index)):
        print(f'{index[i]}:{dist[i]}', end=' ')
    result_index_all.append(index)
    distance_all.append(dist)

    print("\n")

    """
        Display the results
    """
    feature_name = ["Moments", "HOG", "AVGPOOL", "Layer3", "FC"]
    res_show_all(result_index_all, distance_all, feature_name, input_id, k)
