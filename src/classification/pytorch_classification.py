import torch
import torch.nn as nn
import numpy as np
import torchvision
from torchvision import transforms, datasets, models
from torchvision.datasets import ImageFolder
import matplotlib.pyplot as plt
import os
from time import time
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from tqdm import tqdm

import warnings
warnings.filterwarnings("ignore", category=UserWarning) 


class Image_classification:
    def __init__(self):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f"The process is running in {self.device}")

    def data_transform(self, train_data_dir, val_data_dir, img_size):
        # Transform the data
        transform_data = {
            "train": ImageFolder(root= train_data_dir,
                                    transform= transforms.Compose([
                                    transforms.Resize(img_size),
                                    transforms.RandomResizedCrop(640),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor()])
            ),
            "val": ImageFolder(root= val_data_dir,
                                    transform=transforms.Compose([
                                    transforms.Resize(img_size),
                                    transforms.CenterCrop(640),
                                    transforms.ToTensor()])
            )
        }

        return transform_data

    def data_loader(self, data_transform, batch_size):
        
        # Load the data
        data_loader = {
            "train": DataLoader(data_transform["train"], batch_size=batch_size, shuffle=True, num_workers=4),
            "val": DataLoader(data_transform["val"], batch_size=batch_size, shuffle=True, num_workers=4)
        }
        return data_loader


    def build_model(self, model, transform_data, freeze_layer = True):

        
        model = models.get_model(model, pretrained=True)

        classes = transform_data["train"].classes

        # Freeze the layers
        if freeze_layer:
            for param in model.parameters():
                param.required_grad = False

        no_of_features = model.fc.in_features
        model.fc = nn.Linear(no_of_features, len(classes)) #incoming and outgoing features
        model = model.to(self.device)

        return model

    def train_loop(self, model, data_loader, loss_fn, optimizer, epochs):

        for epoch in range(epochs):
            print(f"Epoch {epoch}/{epochs-1}")
            print('*'*50)

            for phase in ["train", "val"]:
                training = phase == "train"

                if training:
                    model.train()
                else:
                    model.eval()

                running_loss = 0.0
                running_accuracy = 0.0

                with tqdm(data_loader[phase]) as tepoch:
                    for x,y in tepoch:
                        x = x.to(self.device)
                        y = y.to(self.device)

                        optimizer.zero_grad()

                        with torch.set_grad_enabled(training):
                            y_pred = model(x)
                            loss = loss_fn(y_pred, y)
                            
                            if training:
                                loss.backward()
                                optimizer.step()
                            
                            # Calculate the accuracy
                            train_pred = torch.max(y_pred, dim=1).indices
                            running_loss += loss.item() * x.size(0)
                            running_accuracy += torch.sum(train_pred == y.data)

                    epoch_loss = running_loss / len(data_loader[phase].dataset) #len(data_loader[phase])             
                    epoch_accuracy = running_accuracy.double() / len(data_loader[phase].dataset) #len(data_loader[phase])

                    print(f"Epoch: {epoch}, {phase} Loss: {epoch_loss}, Accuracy: {epoch_accuracy}")

        return model

                

if __name__=='__main__':
    train_data_dir = "/home/shyam/Documents/CV suite/data/train"
    val_data_dir = "/home/shyam/Documents/CV suite/data/test"
    img_size = 640,640
    batch_size = 12
    epochs = 10
    model = "resnet18"
    freeze_layer = True


    image_classification = Image_classification()
    data_transform = image_classification.data_transform(train_data_dir, val_data_dir, img_size)
    print("Data transformed")

    data_loader = image_classification.data_loader(data_transform, batch_size)
    print("Data Loader created")

    model = image_classification.build_model(model, data_transform, freeze_layer)
    print("Model built successfully")

    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    model = image_classification.train_loop(model, data_loader, loss_fn, optimizer, epochs)
    print("Model trained successfully")

