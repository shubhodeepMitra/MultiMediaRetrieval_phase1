import concurrent.futures
import time
import torch
import torchvision
from torchvision import datasets, models, transforms
import numpy as np

import color_moments
import hog
import resnet_layer
import multiprocessing
from torchvision.models import ResNet50_Weights


print("Torchvision Version: ", torchvision.__version__)

data_dir = '/Users/shubhodeepmitra/Documents/dev/cse_515/phase1/dataset'

# Caltech 101 dataset
caltech101_dataset = datasets.Caltech101(
    root=data_dir,
    download=True,
)

# original model
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model = models.resnet50(weights=ResNet50_Weights.DEFAULT)
model = model.to(device)

# Set the model in evaluation mode
model.eval()

# Define a function to process a single image
def process_image(image_ID, caltech101_dataset):
    img, label = caltech101_dataset[image_ID]
    if img.mode != 'RGB':
        return image_ID, None

    moments_output = color_moments.calculate_color_moments(img)
    hog_output = hog.calculate_hog(img)
    avgpool_output = resnet_layer.calculate_resnet_layers(model, img, 0).numpy().flatten()
    layer3_output = resnet_layer.calculate_resnet_layers(model, img, 1).numpy().flatten()
    fc_output = resnet_layer.calculate_resnet_layers(model, img, 2).numpy().flatten()

    results = {
        'moments_output': moments_output,
        'hog_output': hog_output,
        'avgpool_output': avgpool_output,
        'layer3_output': layer3_output,
        'fc_output': fc_output,
    }

    return image_ID, results


if __name__ == "__main__":
    num_processes = multiprocessing.cpu_count() // 2  # Number of CPU cores available

    # Split the workload into chunks for parallel processing
    chunk_size = len(caltech101_dataset) // num_processes
    image_chunks = [range(i * chunk_size, (i + 1) * chunk_size) for i in range(num_processes)]

    start_time = time.time()

    # Create a ProcessPoolExecutor
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_processes) as executor:
        futures = []
        for image_ID in range(len(caltech101_dataset)):
            # Submit a task for each image
            future = executor.submit(process_image, image_ID, caltech101_dataset)
            futures.append(future)

        moments_list = []
        hog_list = []
        avgpool_list = []
        layer3_list = []
        fc_list = []

        # Collect the results as they become available
        for future in concurrent.futures.as_completed(futures):
            image_ID, results = future.result()
            if results is None:
                continue
            if results['moments_output'] is not None:
                moments_output = np.insert(results['moments_output'], 0, image_ID)
                moments_list.append(moments_output)
            if results['hog_output'] is not None:
                hog_output = np.insert(results['hog_output'], 0, image_ID)
                hog_list.append(hog_output)
            if results['avgpool_output'] is not None:
                avgpool_output = np.insert(results['avgpool_output'], 0, image_ID)
                avgpool_list.append(avgpool_output)
            if results['layer3_output'] is not None:
                layer3_output = np.insert(results['layer3_output'], 0, image_ID)
                layer3_list.append(layer3_output)
            if results['fc_output'] is not None:
                fc_output = np.insert(results['fc_output'], 0, image_ID)
                fc_list.append(fc_output)

    print("Time Taken: ", time.time()-start_time)

    # Save the numpy arrays to files
    if len(moments_list) > 0:
        np.save('../results/caltech101_moments.npy', np.array(moments_list))

    if len(hog_list) > 0:
        np.save('../results/caltech101_hog.npy', np.array(hog_list))

    if len(avgpool_list) > 0:
        np.save('../results/caltech101_resnet_avgpool_1024.npy', np.array(avgpool_list))

    if len(layer3_list) > 0:
        np.save('../results/caltech101_resnet_layer3_1024.npy', np.array(layer3_list))

    if len(fc_list) > 0:
        np.save('../results/caltech101_resnet_fc_1000.npy', np.array(fc_list))
