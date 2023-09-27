import random
import torch
import torchvision
from torchvision import datasets, models, transforms 
from pathlib import Path
import os
from IPython.display import display, Image
import numpy as np
from torchvision.models import ResNet50_Weights
from torchvision.io import read_image, ImageReadMode


import color_moments
import hog
import resnet_layer

print("Torchvision Version: ", torchvision.__version__)

data_dir = '/Users/shubhodeepmitra/Documents/dev/cse_515/phase1/dataset'


# Caltech 101 dataset
caltech101_dataset = datasets.Caltech101(
    root=data_dir,
    download=True,
    target_type='annotation'
)

# original model
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model = models.resnet50(weights=ResNet50_Weights.DEFAULT)
model = model.to(device)

# Set the model in evaluation mode
model.eval()

# data_loader = torch.utils.data.DataLoader(caltech101_dataset,batch_size=4, shuffle=True, num_workers=8)

"""
    Task 1:
    Take input from the user
    1. Image ID
    2. Feature Descriptor
"""

input_id = int(input("Enter Image ID: "))
descriptor = int(input("Enter the feature Descriptor: \n"
                   "1. Color Moments\n"
                   "2. HOG\n"
                   "3. ResNet50 AVGPOOL\n"
                   "4. ResNet50 Layer3\n"
                   "5. ResNet50 FC\n"))


img, label = caltech101_dataset[input_id]

if descriptor == 1:
    print("Color Moments Feature Descriptor:")
    color_moments_output = color_moments.calculate_color_moments(img)
    print("Color Moments Shape:", color_moments_output.shape)
    print(color_moments_output)
elif descriptor == 2:
    print("HOG Feature Descriptor:")
    hog_output = hog.calculate_hog(img)
    print("HOG Shape:", hog_output.shape)
    print(hog_output)
elif descriptor == 3:
    print("ResNet-AvgPool-1024 Feature Descriptor:")
    avgpool_output = resnet_layer.calculate_resnet_layers(model, img, 0).numpy().flatten()
    print("ResNet-AvgPool-1024 Output Shape:", avgpool_output.shape)
    print(avgpool_output)
elif descriptor == 4:
    print("ResNet-Layer3-1024 Feature Descriptor:")
    layer3_output = resnet_layer.calculate_resnet_layers(model, img, 1).numpy().flatten()
    print("ResNet-Layer3-1024 Output Shape:", layer3_output.shape)
    print(layer3_output)
elif descriptor == 5:
    print("ResNet-FC-1000 Feature Descriptor:")
    fc_output = resnet_layer.calculate_resnet_layers(model, img, 2).numpy().flatten()
    print("ResNet-FC-1000 Output Shape:", fc_output.shape)
    print(fc_output)
else:
    print("ERRR!! Input choice not recognised")

display(img)





