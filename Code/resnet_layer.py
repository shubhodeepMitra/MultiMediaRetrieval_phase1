import torch
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import numpy as np

# Resize the image to 224x224
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])


def calculate_resnet_layers(model, image, descriptor_layer):
    image = preprocess(image).unsqueeze(0)

    # Define a hook to capture the output of the layer as provided by the input "name"
    #  Documentation
    #  https://web.stanford.edu/~nanbhas/blog/forward-hooks-pytorch/
    activation = {}

    def get_activation(name):
        def hook(model, input, output):
            activation[name] = output

        return hook

    # Register hooks for avgpool, L3 and FC layers
    avgpool_hook = model.avgpool.register_forward_hook(get_activation('avgpool'))
    layer3_hook = model.layer3.register_forward_hook(get_activation('layer3'))
    layerfc_hook = model.fc.register_forward_hook(get_activation('fc'))

    # Forward pass through the model
    with torch.no_grad():
        model(image)

    if descriptor_layer == 0:
        avgpool_output = activation['avgpool']
        # Reduce the number of dimensions to 1024 by averaging two consecutive entries
        avgpool_output = avgpool_output.view(2048, -1)  # Reshape to 2048x1
        avgpool_output = torch.mean(avgpool_output.view(2, -1, 1024), dim=0)
        avgpool_hook.remove()
        return avgpool_output

    if descriptor_layer == 1:
        # Extract the activation from the hook (layer3 output)
        layer3_output = activation['layer3']
        # Convert the 1024x14x14 tensor to a 1024-dimensional vector by averaging each 14x14 slice
        layer3_output = torch.mean(layer3_output, dim=(2, 3))
        layer3_hook.remove()
        return layer3_output

    if descriptor_layer == 2:
        # Extract the activation from the hook (fc layer output)
        fc_output = activation['fc']
        layerfc_hook.remove()
        return fc_output
