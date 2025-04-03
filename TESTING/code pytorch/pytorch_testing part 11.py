import torch
import numpy as np
import torch.nn as nn

#! Softmax and class entropy

# SOFTMAX FUNCTIONS --> S(yi) = exp(yi)/(Σexp(yi))


def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)


x = np.array([2.0, 1.0, 0.1])
outputs = softmax(x)

print("Softmax Ouput κανονικοποιημενη πιθανοτητα :", outputs)

x = torch.tensor([2.0, 1.0, 0.1])
outputs = torch.softmax(x, dim=0)
print("TOrch softmax function is ", outputs)

# Cross Entropy Loss
# D (y`, y) = - 1/N * Σ (yi* log(y`i)  where y' is predicted and y is actual value

# Examples
# Y = [1,0,0] and the predicted is Y' = [0.7, 0.2, 0.1] Its an OKAY prediction
# So the Cross Entropy Loss is equal to D (Y',Y) = 0,35 small diviation or low cross entropy loss

# Y = [1,0,0] and the predicted is Y' = [0.1, 0.3, 0.6] Its NOT GOOD
# So the Cross Entropy Loss is equal to D (Y',Y) = 2,3 HUGE diviation or high cross entropy loss

# NOTE to Y prepei na einai one hot encoded ara ayto pou theloume na kanoume classification na einai 1 kai ta allaclasses 0
# NOTE opws kai prepei kai to Y' na einai pithanotites kanonikopoihmenes apo SOFTMAX


def cross_entropy(actual, predicted):
    loss = - np.sum(actual * np.log(predicted))
    return loss


Y = np.array([1.0, 0.0, 0.0])


y_pred_good = np.array([0.8, 0.1, 0.1])

y_pred_bad = np.array([0.1, 0.4, 0.5])

l1_good = cross_entropy(Y, y_pred_good)
l2_bad = cross_entropy(Y, y_pred_bad)

print(f"A good prediction has a cross entropy loss of : {l1_good:.4f}")
print(f"A bad prediction has a cross entropy loss of : {l2_bad:.4f}")

# nn.CrossEntropyLoss alerady applies nn.LogSoftMax + nn.NLLLoss(negative log likelihood loss)

loss = nn.CrossEntropyLoss()

Y = torch.tensor([0])
# n of samples * n of classes = 1x3
Y_pred_g = torch.tensor([[2.0, 1.0, 0.1]])  # note raw values not softmaxed
y_pred_b = torch.tensor([[0.4, 2.3, 1.2]])

print(f"A GOOD cross entropy loss is {loss(Y_pred_g, Y).item()}")
print(f"A BAD cross entropy loss is {loss(y_pred_b, Y).item()}")

# How we can CHOOSE the highest propability

_, predictions1 = torch.max(Y_pred_g, 1)
_, predictions2 = torch.max(y_pred_b, 1)
# this one from pred_g chooses the FIRST value of the tensor
print(predictions1)
# this one from the pred_b chooses the SECOND value of the tensor
print(predictions2)

# What if we had 3 Samples on the classes so
# 3 Samples
# NOTE if you want to have a good prediction then you need to think of it as indecies
# NOTE in the tensor below classes are like this ([THIRD, FIRST, SECOND])
Y = torch.tensor([2, 0, 1])

# The size of the predictions MUST be n samples * n classes = 3x3
# NOTE SO HERE if we want to have good or bad prediction we must think the class as a single matrix of the big matrix

Y_pred_nice = torch.tensor([[0.3, 1.0, 2.0],
                            [2.0, 1.0, 0.1],
                            [1.0, 3.0, 0.4]])

Y_pred_poop = torch.tensor([[3.0, 1.0, 0.6],
                            [0.9, 1.0, 2.1],
                            [1.9, 0.3, 2.4]])


print(f"A NICE cross entropy loss is {loss(Y_pred_nice, Y).item()}")
print(f"A POOP cross entropy loss is {loss(Y_pred_poop, Y).item()}")

# and lets also see what item was picked as a prediction here

_, predictions_nice = torch.max(Y_pred_nice, 1)
_, predictions_poop = torch.max(Y_pred_poop, 1)

print(predictions_nice)  # This DOES give us in all cases the correct label

print(predictions_poop)  # here we missed on all 3 instances

#! NEURAL NETWORK APPLICATION


class NeuralNet2(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet2, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)  # Linear Layer
        self.relu = nn.ReLU()  # Activation Funciton
        # Another Linear layer
        self.linear2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):  # here we apply the layers
        out = self.linear1(x)
        out = self.relu(out)
        out = self.linear2(out)
        # no softmax at the end
        return out


model = NeuralNet2(input_size=28*28, hidden_size=5, num_classes=3)

criterion = nn.CrossEntropyLoss()  # apply the softmax
