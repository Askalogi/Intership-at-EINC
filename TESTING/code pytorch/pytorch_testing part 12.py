import torch
import torch.nn as nn
import numpy as np

#! ACTIVATION FUNCITONS


# 1st way to create nn modules


class NeuralNet2(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet2, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)  # Linear Layer
        self.relu = nn.ReLU()  # Activation Funciton
        # Another Linear layer
        self.linear2 = nn.Linear(hidden_size, num_classes)
        self.sigmoid = nn.Sigmoid()  # Activation Function
        # ? --> nn.ReLU / nn.LeakyReLU / nn.Sigmoid / nn.Softmax / nn.TanH

    def forward(self, x):  # here we apply the layers
        out = self.linear1(x)
        out = self.relu(out)
        out = self.linear2(out)
        out = self.sigmoid(out)
        return out

# second way we can do this is (Use activation functions directly in the forward pass)


class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(NeuralNet, self).__init__
        # We initialize only 2 layers
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # And here through the torch build in methods we pply the required activation funcitons after each layer

        out = torch.relu(self.linear1(x))
        out = torch.sigmoid(self.linear2(out))
        # ? --> torch.relu / torch.softmax / torch.leakyrelu / torch.sigmoid / torch.tanh
        # ? Somtimes they are not avaliable so we must first
        # ? import torch.nn.functional as F and then do F.relu and F.sigmoid etc...
        return out
