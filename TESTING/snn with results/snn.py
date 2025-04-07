import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import snntorch as snn 
import torch.nn as nn
import torch.utils.data.dataloader
import torchvision
import torchvision.transforms as transforms
from snntorch import surrogate
from snntorch import functional as SF


#!TEST SNN FOR MNIST DATASET

# DEVICE
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#? print(device)

# HYPERPARAMETERS
# Hyperparameters
batch_size = 100
input_size = 784
hidden_size = 196
output_size = 10 
membrane_potential_decay_rate = 0.9
num_steps = 20  
epochs = 10
learning_rate = 0.001

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Dataset preparation
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))  # MNIST mean and std normalization
])

train_dataset = torchvision.datasets.MNIST(
    root="../CNN,SNN,SCNN", train=True, transform=transform, download=True)

test_dataset = torchvision.datasets.MNIST(
    root="../CNN,SNN,SCNN", train=False, transform=transform, download=True)

train_loader = torch.utils.data.DataLoader(
    dataset=train_dataset, batch_size=batch_size, shuffle=True)

test_loader = torch.utils.data.DataLoader(
    dataset=test_dataset, batch_size=batch_size, shuffle=False)

# Spike gradient
spike_grad = surrogate.atan()  # Increased slope for better gradient flow

class ImprovedSpikingNeuralNetwork(nn.Module):
    def __init__(self, input_size=784, hidden_size=196, output_size=10, 
                 beta=0.9, spike_grad=surrogate.sigmoid()):
        super().__init__()
        
        # Input layer
        self.l1 = nn.Linear(input_size, hidden_size)
        
        # Initialize weights for better gradient flow
        nn.init.xavier_uniform_(self.l1.weight)
        nn.init.zeros_(self.l1.bias)
        
        # Leaky Integrate-and-Fire (LIF) neuron for hidden layer
        self.lif1 = snn.Leaky(beta=beta, spike_grad=spike_grad, learn_beta=True, threshold=1.0)
        
        # Output layer
        self.l2 = nn.Linear(hidden_size, output_size)
        
        # Initialize weights for output layer
        nn.init.xavier_uniform_(self.l2.weight)
        nn.init.zeros_(self.l2.bias)
        
        # LIF neuron for output layer
        self.lif2 = snn.Leaky(beta=beta, spike_grad=spike_grad, learn_beta=True)

    def forward(self, x):
        # Flatten input
        x = x.view(-1, input_size)
        
        # Initialize hidden states
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        
        # Spike and membrane potential records
        spike_record = []
        mem_record = []

        # Propagate through time steps
        for _ in range(num_steps):
            # Hidden layer
            cur1 = self.l1(x) / np.sqrt(hidden_size)
            spk1, mem1 = self.lif1(cur1, mem1)
            
            # Output layer
            cur2 = self.l2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)

            spike_record.append(spk2)
            mem_record.append(mem2)

        # Stack records
        spike_record = torch.stack(spike_record)
        
        # Sum spikes across time steps for classification
        spike_sum = torch.sum(spike_record, dim=0)
        
        return spike_sum / num_steps

def train(model, train_loader, optimizer, criterion, device, epochs=10):
    model.train()
    
    for epoch in range(epochs):
        total_loss = 0
        for images, labels in train_loader:
            # Move to device
            images = images.to(device)
            labels = labels.to(device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            output = model(images)
            
            # Compute loss
            loss = criterion(output, labels)
            
            # Backward pass with gradient clipping
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            # Update weights
            optimizer.step()
            
            total_loss += loss.item()
        
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {total_loss/len(train_loader):.4f}')

def test(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            output = model(images)
            _, predicted = torch.max(output, 1)
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = 100 * correct / total
    print(f'Test Accuracy: {accuracy:.2f}%')
    return accuracy

# Main training script
def main():
    # Model initialization
    model = ImprovedSpikingNeuralNetwork(
        input_size=input_size, 
        hidden_size=hidden_size, 
        output_size=output_size, 
        beta=membrane_potential_decay_rate, 
        spike_grad=spike_grad
    ).to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=1e-5)

    # Training loop
    train(model, train_loader, optimizer, criterion, device, epochs)

    # Testing
    test(model, test_loader, device)

    # Gradient checking with detailed reporting
    print("\nGradient Statistics:")
    for name, param in model.named_parameters():
        if param.grad is not None:
            print(f"{name} gradient stats:")
            print(f"  Mean: {param.grad.mean().item()}")
            print(f"  Std: {param.grad.std().item()}")
            print(f"  Min: {param.grad.min().item()}")
            print(f"  Max: {param.grad.max().item()}")
        else:
            print(f"{name} has no gradient")

#Checks the gradients MUST not be 0 or NaN

if __name__ == "__main__":
    main()

for name, param in model.named_parameters():
    if param.grad is not None:
        print(f"{name} gradient stats:")
        print(f"  Mean: {param.grad.mean().item()}")


#plotting the loss funciton 
def plot_weights(layer, title="Weight Matrix Heatmap"):
    if isinstance(layer, nn.Linear):
        weights = layer.weight.cpu().detach().numpy()
        
        plt.figure(figsize=(8, 6))
        plt.imshow(weights, cmap="coolwarm", aspect="auto")  # Visualizing the weight matrix
        plt.colorbar()  # Add color scale
        plt.title(title)
        plt.xlabel("Neurons")
        plt.ylabel("Weights")
        plt.show()


plot_weights(model.hidden_size, "Hidden Layer Weights")
plot_weights(model.output_size, "Output Layer Weights")
