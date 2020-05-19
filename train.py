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
parser = argparse.ArgumentParser (description = "Parser of training script")

parser.add_argument ('--data_root', default ='flowers',help = 'Provide data directory. Mandatory argument', type = str)
parser.add_argument ('--save_dir', default='checkpoint1.pth', help = 'Provide saving directory. Optional argument', type = str)
parser.add_argument ('--arch', default='inception' ,help = 'inception or resnet', type = str)
parser.add_argument ('--lr', default=0.001, help = 'Learning rate, default value 0.001', type = float)
parser.add_argument ('--hidden_units', default=2048, help = 'Hidden units in Classifier. Default value is 100', type = int)
parser.add_argument ('--epochs', default=1, help = 'Number of epochs', type = int)
parser.add_argument ('--device', default = 'cpu', help = "Option to use between GPU or CPU. ", type = str)

# creating object parser and setting parameters
commands = parser.parse_args()
data_dir = commands.data_root
saving_model_path = commands.save_dir
architecture = commands.arch
learning_rate = commands.lr
hidden = commands.hidden_units
Epochs = commands.epochs
if commands.device == 'GPU':
    device = 'cuda'
else:
    device = 'cpu'
## logic starts here
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'

# Define your transforms for the training, validation, and testing sets
# Since we will use inception network we will resize this to 229 x 229 and mean [0.5, 0.5, 0.5]
batch_size = 64
mean = [0.5, 0.5, 0.5]
std = [0.229, 0.224, 0.225]
train_transforms = transforms.Compose([transforms.RandomRotation(45),
                                       transforms.RandomResizedCrop(299),
                                       transforms.RandomVerticalFlip(),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize(mean,
                                                            std)])

valid_transforms = transforms.Compose([transforms.Resize(320),
                                      transforms.CenterCrop(299),
                                      transforms.ToTensor(),
                                      transforms.Normalize(mean,
                                                           std)])

test_transforms = transforms.Compose([transforms.Resize(320),
                                      transforms.CenterCrop(299),
                                      transforms.ToTensor(),
                                      transforms.Normalize(mean,
                                                           std)])

# TODO: Load the datasets with ImageFolder
train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
valid_data = datasets.ImageFolder(valid_dir, transform=valid_transforms)
test_data = datasets.ImageFolder(test_dir, transform=test_transforms)

# TODO: Using the image datasets and the trainforms, define the dataloaders
trainloader = torch.utils.data.DataLoader(train_data, batch_size=256, shuffle=True)
validloader = torch.utils.data.DataLoader(valid_data, batch_size=batch_size)
testloader = torch.utils.data.DataLoader(test_data, batch_size=batch_size)


if architecture == 'resnet':
    model = models.resnet101(pretrained=True)
    model.aux_logits=False
    for param in model.parameters():
        param.requires_grad = False

    # reclacing the last fully connected layer with a new sequence
    fc = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(2048, hidden,bias=True)),
                          ('relu1', nn.ReLU ()),
                          ('dropout1', nn.Dropout (0.7)),
                          ('fc2', nn.Linear (hidden, 102))]))

    model.fc = fc
else:
    model = models.inception_v3(pretrained=True)
    model.aux_logits=False
    for param in model.parameters():
        param.requires_grad = False

    # reclacing the last fully connected layer with a new sequence
    fc = nn.Sequential(OrderedDict([
                              ('dropout1', nn.Dropout (0.7)),
                              ('fc1', nn.Linear(2048, hidden,bias=True)),
                              ('relu1', nn.ReLU ()),
                              ('fc2', nn.Linear (hidden, 102))]))


    model.fc = fc
    

## training the model new classifiers weights
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=learning_rate)


model.to(device)
epochs=Epochs
steps = 0
print_every=6
valid_loss_min = np.Inf # track change in validation loss
for epoch in range(epochs):
    running_loss = 0
    
    for images, labels in trainloader:
        steps += 1
        # Move input and label tensors to the GPU
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model.forward(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if steps % print_every == 0:
            accuracy=0
            model.eval()
            with torch.no_grad(): 
                valid_loss = 0
                stepValid = 0
                for valid_images, valid_labels in validloader:
                    stepValid+=1
                    valid_images, valid_labels = valid_images.to(device), valid_labels.to(device)
                    logits = model.forward(valid_images)
                    loss = criterion(logits, valid_labels)
                    valid_loss += loss.item()
                    _,topClasses=logits.topk(1, dim=1)
                    equals=topClasses==valid_labels.view(*topClasses.shape)
                    accuracy+=torch.mean(equals.type(torch.FloatTensor)) 
                print(f"Epoch {epoch+1}/{epochs}.. "
                f"Train loss: {running_loss/print_every:.4f}.. "
                f"Valid loss: {valid_loss/len(validloader):.4f}.. "
                f"valid accuracy: {accuracy/len(validloader):.4f}")
                # save model if validation loss has decreased and is at least 0.8
                curr_valid_loss = valid_loss/len(validloader) 
                curr_accuracy = accuracy/len(validloader)
                if curr_valid_loss <= valid_loss_min and curr_accuracy>0.8:
                    print('Validation loss decreased ({:.4f} --> {:.4f}).  Saving model ...'.format(
                    valid_loss_min,
                    curr_valid_loss))
                    torch.save(model.state_dict(), 'myModel2.pt')
                    # Save the checkpoint 
                    checkpoint = {'state_dict': model.state_dict(),
                                'fc': model.fc,
                                'class_to_idx': train_data.class_to_idx,
                                'opt_state': optimizer.state_dict,
                                'num_epochs': epochs}

                    torch.save(checkpoint, saving_model_path)
                    valid_loss_min = curr_valid_loss
                model.train()
                running_loss = 0