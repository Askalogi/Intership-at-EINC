import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import snntorch as snn
import torch.nn as nn
import torch.utils.data.dataloader
import torchvision
import norse.torch as norse
import torchvision.transforms as transforms
from torch.utils.data import random_split
from snntorch import surrogate
from snntorch import functional as SF

from class_dataset import CustomSinDataset

horizontal = pd.read_csv("../FULL_PROJECT/custom_dataset/labels.csv")
# Initialize the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# We can see here what device we are actually using
print(device)

#!HYPERPARAMETERS
input_size = 256  # because 16 by 16 size
hidden_size = 128  # hidden size neurons
output_size = 2  # since the number of classes is 0 and 1
batch_size = 1
num_steps = 100  # amount of steps through each epoch
epochs = 30  # the number of epochs where the model is going to be trained on
learning_rate = 0.1  # the learning rate
prcnt_of_train = 0.7  # 70% is going to be trained so input a float here

#!preparing the Data
# first we create the transformation
transform = transforms.Compose(
    [
        transforms.ToTensor(),  # let's only try just the ToTensors for now
        transforms.Normalize((0.5), (0.5)),  # and lets normalize the tensors
    ]
)

# create the train/test dataset
dataset = CustomSinDataset(
    root="./custom_dataset",
    label_csv="./custom_dataset/labels.csv",
    transform=transform,
)

# Split the dataset into train and test with customizable size
train_size = int(prcnt_of_train * len(dataset))
test_size = len(dataset) - train_size

# Save them into actual train/test datasets
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

# And now apply the above in the Dataloader
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True
)
test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=batch_size, shuffle=True
)

#!spike gradient for the non-differentiability of spikes (not norse)
# spike_grad = surrogate.sigmoid()  # let's try with this
# CHECK INSIDE THE CLASS


#! Creating the Class Model (snn)
class SpikingNeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, beta=0.9):
        super().__init__()

        self.spike_grad = surrogate.sigmoid()
        # Input connections - linear transformation before LIF
        self.fc1 = nn.Linear(input_size, hidden_size)

        # Hidden layer LIF
        self.lif1 = norse.LIFCell(p=norse.LIFParameters(alpha=20))

        # Hidden to hidden connections
        self.fc2 = nn.Linear(hidden_size, hidden_size)

        # Second hidden layer LIF
        self.lif2 = norse.LIFCell(p=norse.LIFParameters(alpha=20))

        # Output connections
        self.fc3 = nn.Linear(hidden_size, output_size)

        # Output layer LIF
        self.lif3 = norse.LIFCell(p=norse.LIFParameters(alpha=20))
        # Create a hashmap for storing the weights so we can later plot them
        self.weight_history = {
            "1st_Fully_Connected_layer": [],
            "2nd_Fully_Connected_layer": [],
            "3rd_Fully_Connected_layer": [],
        }

    def forward(self, x, num_steps=num_steps):
        # define the batch size
        batch_size = x.shape[0]

        # Initialize states with dummy tensor
        dummy_hidden = torch.zeros(batch_size, hidden_size, device=device)
        dummy_output = torch.zeros(batch_size, output_size, device=device)

        # Initialize state for each layer
        lif1_state = self.lif1.initial_state(dummy_hidden)
        lif2_state = self.lif2.initial_state(dummy_hidden)
        lif3_state = self.lif3.initial_state(dummy_output)

        # Storing the output spikes
        spike_record_1 = []
        spike_record_2 = []
        spike_record_final = []

        for _ in range(num_steps):
            # First transform input to hidden size
            x1 = self.fc1(x)

            # First LIF layer
            out1, lif1_state = self.lif1(x1, lif1_state)

            spike_record_1.append(out1)
            # a1 = torch.stack(spike_record_1, dim=0).sum(0)
            # print(f"this is the contents of the spike after the 1st liff layer {spike_record_1} ")
            # print(f"This is the sum of spikes over time  for the first layer {a1}")

            # Second transformation
            x2 = self.fc2(out1)

            # Second LIF layer
            out2, lif2_state = self.lif2(x2, lif2_state)

            spike_record_2.append(out1)
            # a2 = torch.stack(spike_record_2, dim=0).sum(0)
            # print(f"this is the contents of the spike after the 2nd liff layer{spike_record_2} ")
            # print(f"This is the sum of spikes over time for the second layer {a2}")

            # Output transformation
            x3 = self.fc3(out2)

            # Final LIF layer
            out3, lif3_state = self.lif3(x3, lif3_state)

            # Store output
            spike_record_final.append(out3)

        # Return the sum of spikes over time
        return torch.stack(spike_record_final, dim=0).sum(0)

    def save_weights(self):
        """Saving the weights for each layer"""
        self.weight_history["1st_Fully_Connected_layer"].append(
            self.fc1.weight.data.clone().cpu()
        )
        self.weight_history["2nd_Fully_Connected_layer"].append(
            self.fc2.weight.data.clone().cpu()
        )
        self.weight_history["3rd_Fully_Connected_layer"].append(
            self.fc3.weight.data.clone().cpu()
        )

    def plot_weights(self):
        """We are defining the method which weights are plotted here"""
        # We initiate a single figure with 3 subplots
        fig, axes = plt.subplots(1, 3, figsize=(20, 10))

        # for each fully connected layer it checks if there is history stored for that specific layer
        for i, layer_name in enumerate(
            [
                "1st_Fully_Connected_layer",
                "2nd_Fully_Connected_layer",
                "3rd_Fully_Connected_layer",
            ]
        ):
            if not self.weight_history[layer_name]:
                continue

            # retrieves the weight stored snapshots in the weights history
            weights = self.weight_history[layer_name]

            # actual plotting of the weights
            for j, w in enumerate(weights):
                if (
                    j % 2 == 0 or j == len(weights) - 1
                ):  # Plot every other epoch of the weights and the final one(if it's even)
                    axes[i].hist(
                        w.flatten().numpy(),
                        alpha=0.5,  # alpha is the opacity
                        label=f"Weight Epochs {j + 1}",
                        bins=20,
                    )

            axes[i].set_title(
                f" Fully Connected Layer : {layer_name} Weight Distribution"
            )
            axes[i].set_xlabel("Weight Value")
            axes[i].set_ylabel("Density")
            axes[i].legend()

        plt.tight_layout()
        plt.show()


#! CREATING THE MODEL
model = SpikingNeuralNetwork(input_size, hidden_size, output_size).to(device)

#! Defining the loss function and the optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=1e-5)


#! Defining the Train Function
def train(model, train_loader, criterion, optimizer, num_steps, epochs):
    # We have the training metrics
    loss_history = []
    acc_history = []

    for epoch in range(epochs):
        total_loss = 0
        correct = 0
        total = 0

        for i, (inputs, labels) in enumerate(train_loader):
            inputs = inputs.view(inputs.shape[0], 256).to(device)
            labels = labels.to(device)
            # Forward pass
            outputs = model(inputs, num_steps)
            loss = criterion(outputs, labels)
            # print(outputs.shape)
            # Backward pass and optimize (backprop)
            optimizer.zero_grad()
            loss.backward()
            # print(model.fc1.weight)
            optimizer.step()

            # We need to track the metric so we can print them later
            total_loss = total_loss + loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total = total + labels.size(0)
            correct = correct + (predicted == labels).sum().item()

        # We are storing the models weights after each epoch
        model.save_weights()

        # Calculate epoch metrics
        avg_loss = total_loss / len(train_loader)
        accuracy = 100 * correct / total

        loss_history.append(avg_loss)
        acc_history.append(accuracy)

        print(
            f"Epoch {epoch + 1} / {epochs}, the Loss is: {avg_loss:.4f}, the Accuracy is: {accuracy:.2f}%"
        )

    # Let's also plot the metrics here
    plt.figure(figsize=(12, 5))

    # 2 subplots one for accuracy and one for loss
    # loss
    plt.subplot(1, 2, 1)
    plt.plot(loss_history)
    plt.title("Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")

    # accuracy
    plt.subplot(1, 2, 2)
    plt.plot(acc_history)
    plt.title("Accuracy of the Training")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")

    plt.tight_layout()
    plt.show()

    return loss_history, acc_history


#! DEFINE THE TEST FUNCTION
def test(model, test_loader, criterion, num_steps):
    model.eval()  # model evaluation

    with torch.no_grad():
        correct = 0
        total = 0
        total_loss = 0

        for inputs, labels in test_loader:
            inputs = inputs.view(-1, input_size).to(device)
            labels = labels.to(device)

            outputs = model(inputs, num_steps)
            loss = criterion(outputs, labels)

            total_loss = total_loss + loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total = total + labels.size(0)
            print(f"the predicted ones are {predicted} ")
            print(f" The labels are {labels} ")
            correct = correct + (predicted == labels).sum().item()

        print(
            f"Test Loss is: {total_loss / len(test_loader):.4f}, Test Accuracy is: {100 * correct / total:.2f}%"
        )

    model.train()
    return 100 * correct / total


#! TRAIN THE MODEL
loss_history, acc_history = train(
    model, train_loader, criterion, optimizer, num_steps, epochs
)

#! TEST THE MODEL
test_accuracy = test(model, test_loader, criterion, num_steps)

#! VISUALIZE THE WEIGHTS
model.plot_weights()

# print(dict(model.named_parameters()))
