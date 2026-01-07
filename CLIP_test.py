import requests
import torch

from PIL import Image

from torchvision.datasets import Caltech101

import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split

import torchvision.datasets as datasets
import torchvision.transforms as transforms

from transformers import AutoProcessor, AutoModel, pipeline, CLIPModel, CLIPProcessor

from tqdm import tqdm
import wandb


"""
OBJECTIVES:
- Take pre-trained CLIP model that can understand images
- Freezes CLIP model and uses it as a feature extractor -> convert every image into embedding/layer
- Train new neural network to map embeddings to the 101 class labels in the Caltech dataset

"""


# lightning imports
import os
from torch import optim, utils, Tensor
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor

processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
target_size = processor.image_processor.crop_size["height"]
print(target_size) # 224

# preprocessing pipeline for CLIP images

img_transforms = transforms.Compose([
    transforms.Resize(target_size),
    transforms.CenterCrop(target_size),
    transforms.Lambda(lambda img: img.convert("RGB")),
    transforms.ToTensor(),
    transforms.Normalize(
        mean = processor.image_processor.image_mean,
        std = processor.image_processor.image_std
    )
])

dataset = Caltech101(root = './data', download = True, transform = img_transforms) # loads as tensor

# check # categories
# print(dataset.__len__())
num_categories = len(dataset.categories)
print(num_categories) # should be 101

# split into training/validation/testing data (70, 15, 15)
size = len(dataset)
training_size = int(0.7 * size)
validation_size = int(0.15 * size)
testing_size = size - training_size - validation_size

# random split
train_set, val_set, test_set = random_split(dataset, [training_size, validation_size, testing_size])

train_loader = DataLoader(train_set, batch_size = 32, shuffle = True)
val_loader = DataLoader(val_set, batch_size = 32, shuffle = False)
test_loader = DataLoader(test_set, batch_size = 32, shuffle = False)

class NN(nn.Module):
    def __init__(self, input_size, num_classes):
        super(),__init__()
        self.fulCon1 = nn.Linear(input_size, 50)
        self.fulCon2 = nn.Linear(50, num_classes)

    def forward(self, x):
        x = F.relu(self.fulCon1(x)) # relu replaces all negative numbers with 0
        x = self.fulCon2(x)
        return x # logits
    
LEARNING_RATE = 0.001
NUM_EPOCHS = 3

# set up

DEVICE = "cuda" if torch.cuda.is_available() else "cpu" # run on fast GPU

# use w&b
wandb.login()
wandb.init(
    project="CLIP-Caltech101-Manual-Loop",
    config={
        "learning_rate": LEARNING_RATE,
        "batch_size": BATCH_SIZE,
        "num_epochs": NUM_EPOCHS,
        "architecture": "CLIP_Embeddings + 2-Layer_NN",
    }
)

clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(DEVICE)
clip_model.requires_grad_(False) # FREEZING CLIP MODEL
clip_model.eval() 

# embedding size outputted = 512
model = NN(input_size=512, num_classes=num_categories).to(DEVICE)

# CrossEntropyLoss: loss function for multi-class classification
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE) # optimization, will NOT update clip_model

print("Starting training...")
for epoch in range(NUM_EPOCHS):
    model.train()

    for batch_idx, (data, targets) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}")):
        data = data.to(device=DEVICE)
        targets = targets.to(device=DEVICE)

        with torch.no_grad():
            embeddings = clip_model.get_image_features(pixel_values=data)

        scores = model(embeddings)
        loss = criterion(scores, targets)

        wandb.log({"train_loss": loss.item()})

        optimizer.zero_grad()
        loss.backward() # back propagation

        optimizer.step()
    

def check_accuracy(loader, nn_model, clip_feature_extractor):
    num_correct = 0
    num_samples = 0
    nn_model.eval() 
    with torch.no_grad():
        for x, y in tqdm(loader, desc="Checking accuracy"):
            x = x.to(device=DEVICE)
            y = y.to(device=DEVICE)

            embeddings = clip_feature_extractor.get_image_features(pixel_values=x)

            scores = nn_model(embeddings)
            predictions = scores.max(1)[1]

            num_correct += (predictions == y).sum() # counts number of True values

            num_samples += predictions.size(0) # add batch size to total count

    nn_model.train()
    accuracy = (num_correct / num_samples) * 100
    return accuracy

print("Checking accuracy...")
train_acc = check_accuracy(train_loader, model, clip_model)
test_acc = check_accuracy(test_loader, model, clip_model)

print(f"Accuracy on training set: {train_acc:.2f}%")
print(f"Accuracy on test set: {test_acc:.2f}%")

wandb.log({
    "final_train_accuracy": train_acc,
    "final_test_accuracy": test_acc
})

wandb.finish()