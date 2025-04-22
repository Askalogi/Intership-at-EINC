import numpy as np
import norse.torch as norse
import torch
import torchvision.transforms as transforms
import torch.nn as nn
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from class_dataset import CustomSinDataset
import torch.utils.data.dataloader
from norse.torch.functional.encode import constant_current_lif_encode  # rate encoding
from torch.utils.data import random_split

# conv2-> lif -> maxpool2d -> 2LI layers

#! DEVICE

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#! HPYERPARAPAMETERAS


#!TRANSFORM
transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Grayscale(),
        transforms.Normalize((0.5,), (0.5,)),
    ]
)

dataset = CustomSinDataset(root="./custom_dataset", transform=transform)


#! SPLIT THE DATA TO TRAIN AND TEST
train_size = int(len(dataset) * prcnt_of_train)
test_size = len(dataset) - train_size
test_size
train_size

train_dataset, test_dataset = random_split(dataset, [train_size, test_size])


#! DEFINE THE DATA LOADER

train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True
)

test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=batch_size, shuffle=True
)


#! MODEL
class LI2Model(nn.Module):
    def __init__(self, input_size, output_size, num_steps):
        super(LI2Model, self).__init__()

        self.num_steps = num_steps

        #!LAYERS:
        # Classic conv layer
        self.conv = nn.Conv2d(1, 2, 4, 1)
        # LIF Layer
        self.lif = norse.LIFCell(p=norse.LIFParameters())
        # Classical Pooling layer
        self.pool = nn.MaxPool2d(2, 1)
        # LI 1 layer
        self.li1 = norse.LICell(p=norse.LIFParameters())
        # LI 2 layer
        self.li2 = norse.LICell(p=norse.LIFParameters())
        # linear or fcl
        self.fcl = nn.Linear(2 * 6 * 6, output_size)
