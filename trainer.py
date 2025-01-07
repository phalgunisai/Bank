from torch import optim
import torch.nn as nn
import torchvision
import preprocessing as pp
import train
import torch
import numpy as np

# Load the ResNet-18 model
model_conv = torchvision.models.resnet18(pretrained=True)

# Freeze all layers except the final fully connected (fc) layer
for param in model_conv.parameters():
    param.requires_grad = False

for param in model_conv.fc.parameters():
    param.requires_grad = True

# Define the new network head and attach it to the model
num_ftrs = model_conv.fc.in_features

headModel = nn.Sequential(
    nn.Linear(num_ftrs, 512),
    nn.ReLU(),
    nn.Dropout(0.25),
    nn.Linear(512, 256),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(256, 128),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(128, 64),
    nn.ReLU(),
    nn.Linear(64, 7)  # Assuming 7 classes for denominations
)

model_conv.fc = headModel

# Move the model to the appropriate device
model_conv = model_conv.to(pp.device)

# Define the loss function and optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer_conv = optim.Adam(model_conv.fc.parameters(), lr=0.001)

# Train the model
model_ft = train.train(
    model_conv,
    optimizer_conv,
    loss_fn,
    pp.device,
    pp.image_dataloaders,
    dataset_size=pp.dataset_size,
    epochs=1000
)

# Feature extraction function
def extract_features(model, dataloader, device):
    model.eval()
    features_list = []
    labels_list = []

    with torch.no_grad():
        for inputs, labels in dataloader['train']:
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Get features from the penultimate layer
            outputs = model(inputs)
            features_list.extend(outputs.cpu().numpy())
            labels_list.extend(labels.cpu().numpy())

    return np.array(features_list), np.array(labels_list)

# Extract features using the trained model
features, labels = extract_features(model_ft, pp.image_dataloaders, pp.device)

# Save the extracted features and labels
np.save('features.npy', features)
np.save('labels.npy', labels)

print("Feature extraction complete. Features and labels saved.")