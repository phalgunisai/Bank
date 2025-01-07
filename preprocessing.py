import torch
from torch.utils.data import DataLoader
from torchvision import models, datasets, transforms
import numpy as np
import os
import time

# Define the path relative to the root of the repository
path = os.path.join(os.getcwd(), 'classification')  # 'classification' is directly under the repo
path_folder = ['train', 'val']

# Define data transformations for training and validation
transformers = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
}

# Check if the folders exist, otherwise throw an error
if not all(os.path.exists(os.path.join(path, folder)) for folder in path_folder):
    raise FileNotFoundError(f"One of the folders {path_folder} is missing in the directory {path}")

# Load the dataset
image_dataset = {x: datasets.ImageFolder(os.path.join(path, x), transformers[x]) for x in path_folder}

# Prepare dataloaders
image_dataloaders = {x: DataLoader(image_dataset[x], batch_size=64, num_workers=os.cpu_count(), shuffle=True) for x in path_folder}

# Get the size of each dataset (train/val)
dataset_size = {x: len(image_dataset[x]) for x in path_folder}

# Get class names
class_names = image_dataset['train'].classes

# Set device to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
