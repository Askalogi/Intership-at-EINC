import numpy as np
import norse.torch as norse
import torch
import torch.optim.adam
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
input_size = 16 * 16
num_classes = 2
num_steps = 300
epochs = 10
batch_size = 20
learning_rate = 0.005

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

    def forward(self, x):
        batch_size = x.shape[0]  # [B,C,H,W]
        x_encoded = constant_current_lif_encode(
            x, p=norse.LIFParameters(), seq_length=self.num_steps
        )
        # after the line above the x now has temporal dimension [T,B,C,H,W]
        mem_record = []
        spk_record = []
        counter = 0
        mem = None

        for step in range(self.num_steps):
            x_curr = x_encoded[step]

            #! Going through the layers

            # conv layer
            convout = self.conv(x_curr)
            # lif layer
            spk, mem = self.lif(convout, mem)
            # pooling layer
            pooled_mem = self.pool(mem)
            pooled_spk = self.pool(spk)
            # first LI layer (for membrance )
            mem2 = self.li1(pooled_mem)
            # second LI layer (for spikes )
            spk2 = self.li2(pooled_spk)
            # flatten and then pass through full connected layer
            flat_spk = spk2.view(batch_size, -1)
            flat_mem = mem2.view(batch_size, -1)

            out_mem = self.fcl(flat_mem)
            out_spk = self.fcl(flat_spk)
            # store the results :

            spk_record.append(out_spk)
            mem_record.append(out_mem)

        spk_out = torch.stack(spk_record, dim=0).sum(0)

        return spk_record, spk_out, mem_record


#! CREATING THE MODEL
model = LI2Model(input_size=input_size, output_size=num_classes, num_steps=num_steps)

#! CREATING THE OPTIMIZER
optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate)
#! CREATING THE CRITERION (loss)
criterion = nn.CrossEntropyLoss()


