import numpy as np
import matplotlib.pyplot as plt
import argparse
import json
from PIL import Image
from matplotlib.ticker import FormatStrFormatter
import pandas as pd
from helpers import NormalizeInverse
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from collections import OrderedDict

# Argument parsing
parser = argparse.ArgumentParser ()
parser.add_argument ('--image_path', default='flowers/train/1/image_06735.jpg', help = 'Path to image.', type = str)
parser.add_argument('--checkpoint', default='checkpoint.pth', help='Point to checkpoint file as str.', type=str)
parser.add_argument ('--top_k', default = 5, help = 'Top K most likely classes.', type = int)
parser.add_argument ('--category_names' , default = 'cat_to_name.json', help = 'Mapping of categories to real names.', type = str)
parser.add_argument ('--device', default = 'cpu', help = "Option to use between GPU or CPU. Optional", type = str)

# creating object parser and setting parameters
commands = parser.parse_args()
path_to_image = commands.image_path
saved_model_path = commands.checkpoint
k_top = commands.top_k
json_location = commands.category_names

## logic starts here
with open(json_location, 'r') as f:
    cat_to_name = json.load(f)
if commands.device == 'GPU':
    device = 'cuda'
else:
    device = 'cpu'

def load_checkpoint(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model = models.inception_v3(pretrained=True)
    model.aux_logits=False
    fc = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(2048, 102,bias=True)),
                                                                ]))
    model.fc = fc
    model.load_state_dict(checkpoint['state_dict'], strict=False)
    for param in model.parameters():
        param.requires_grad = False
    model.eval()
    return model

reloaded_model = load_checkpoint(saved_model_path)

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    # TODO: Process a PIL image for use in a PyTorch model
    # Converting image to (Python Imaging Library)PIL image using image file path
    pil_im = Image.open(f'{image}')
    transformations = transforms.Compose([transforms.Resize(320),
                                      transforms.CenterCrop(299),
                                      transforms.ToTensor(),
                                      transforms.Normalize(mean,
                                                           std)])
    # Apply transforms to the PIL image
    pil_transformed = transformations(pil_im)
    
    # Converting to Numpy array 
    image_data = np.array(pil_transformed)
    
    return image_data

def imshow(image, ax=None, title=None):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()
    image = torch.Tensor(image)
    # Undo preprocessing
    image = inv_tensor(image)
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.permute(1,2,0)
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    
def predict(image_path, model, topk, device):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    # TODO: Implement the code to predict the class from an image file
    model.to(f'{device}:0')
    img_torch = process_image(image_path)
    img_torch = torch.from_numpy(img_torch).unsqueeze_(0)
    img_torch = img_torch.float()
    model.eval()
    with torch.no_grad():
        output = model.forward(img_torch)
        model.train()
    probability = F.softmax(output.data,dim=1)
    probabilities, classes = probability.topk(topk)
    class_mapped = []
    for curr_class in classes[0]:
        class_idx = train_data.classes[curr_class]
        class_mapped.append(cat_to_name[class_idx])
    return probabilities[0].numpy(), class_mapped 

def sanity_check(image_path, model, topk, device):
    probabilities, top_classes = predict(image_path, model, topk, device)
    image = process_image(image_path)
    fig = plt.figure(figsize=(15,10))
    ax = fig.add_subplot(1,2,1)
    imshow(image, ax=plt)
    plt.title(f"Predicted label: {top_classes[0]}");
    ax = fig.add_subplot(1,2,2)
    plt.barh(top_classes, probabilities*100)
    fig.tight_layout(w_pad=5)

sanity_check(path_to_image, reloaded_model, k_top, device)