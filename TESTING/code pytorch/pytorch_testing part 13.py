import torch
import torch.nn as nn
import torch.utils.data.dataloader
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt


#! FEED FORWARD NEURAL NETWORK FULL EXAMPLE

# device config  first adn foremost
device = torch.device("cuda " if torch.cuda.is_available() else "cpu")

# hyperparameters
input_size = 784  # because 28 epi 28
hidden_size = 100  # free to change stuff
num_classes = 10  # digits apo 0 - 9 sto MNIST data set
num_epochs = 2  # epoxes
batch_size = 100  # megethos tou batch pou feedaroume kathe fora
learning_rate = 0.001  # ruthmos mathisis

# Importing MNIST dataset
train_dataset = torchvision.datasets.MNIST(
    root="./data", train=True, transform=transforms.ToTensor(), download=True)

test_dataset = torchvision.datasets.MNIST(
    root="./data", train=False, transform=transforms.ToTensor())

train_loader = torch.utils.data.DataLoader(
    dataset=train_dataset, batch_size=batch_size, shuffle=True)

test_loader = torch.utils.data.DataLoader(
    dataset=test_dataset, batch_size=batch_size, shuffle=False)

examples = iter(test_loader)

samples, labels = next(examples)
print(samples.shape, labels.shape)

for i in range(100):
    plt.subplot(10, 10, i+1)
    plt.imshow(samples[i][0], cmap="gray")
# plt.show()


class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.linear1(x)
        out = self.relu(out)
        out = self.linear2(out)

        return out


# we create the model that takes the cusotm class we made
model = NeuralNet(input_size, hidden_size, num_classes)

criterion = nn.CrossEntropyLoss()  # We create this vital part
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Training loops
n_total_steps = len(train_loader)

for epoch in range(num_epochs):
    # tha mas dwsei ton deikti kai epeita to data pou einai eikona kai label se morphi tuple
    for i, (images, labels) in enumerate(train_loader):
        # 100, 1, 28, 28
        # 100, 784
        images = images.reshape(-1, 28 * 28).to(device)
        labels = labels.to(device)

        # forward pass
        output = model(images)
        loss = criterion(output, labels)

        # backwar pass
        # EMPTY THE VALUES
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()  # update the parameters

        if (i+1) % 100 == 0:
            print(
                f" This is {epoch+1} /{num_epochs}, step {i+1}/{n_total_steps}, loss= {loss.item():.4f} ")

# test
# wrap this
with torch.no_grad():
    n_correct = 0
    n_samples = 0
    for images, labels in test_loader:
        images = images.reshape(-1, 28*28).to(device)
        labels = labels.to(device)
        outputs = model(images)

        # Value and the index (we want the idnex)
        _, predictions = torch.max(outputs, 1)
        n_samples += labels.shape[0]
        n_correct += (predictions == labels).sum().item()

    acc = 100.0 * n_correct / n_samples
    print(f"acc equals {acc}")
