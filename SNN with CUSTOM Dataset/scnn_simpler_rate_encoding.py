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

# TODO implement custom learning rate approach
# TODO perhaps implement custom LIF parameters

#! INITIALIZE THE DEVICE

# we remove the threshhold
torch.set_printoptions(threshold=float("inf"))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(device)

#! HYPERPARAMETERS
input_size = 256  # 16*16
# hidden_size = ?
num_classes = 2
epochs = 10
prcnt_of_train = 0.7
batch_size = 30
learning_rate = 0.01
num_steps = 300
v_threshold = torch.tensor([0.8])

#! TRANSFORM
transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Grayscale(),
        # transforms.Normalize((0.5,), (0.5,)),
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


#! DEFINING AND CREATING THE MODEL
class SimpleSNN(nn.Module):
    def __init__(self, input_size, output_size, num_steps):
        super(SimpleSNN, self).__init__()

        self.num_steps = num_steps

        #! DEFINE THE LAYERS (SIMPLE THOUGH)
        # Convolutional layer
        self.conv = nn.Conv2d(1, 2, 4, 1)
        # Pooling layer
        self.pool = nn.MaxPool2d(2)
        # Flatten where the new size is 13x13 cause of the kernel 4 convolution
        self.flatten_size = 2 * 6 * 6
        # Linear layer to map the flattened features to output classes
        self.linear = nn.Linear(self.flatten_size, output_size)
        # Single LIF layer for the spikes
        self.lif = norse.LIFCell(p=norse.LIFParameters())

    def forward(self, x):
        batch_size = x.shape[0]  # current batch size cause [B,C,H,W]

        # I also scaled the input tensor because the original tensor had smaller values
        x_encoded = constant_current_lif_encode(
            x * 10, p=norse.LIFParameters(), seq_length=self.num_steps
        )  # now the shpae is [time,b,c,h,w]

        # print(f"this is the x_encoded {x_encoded.shape}")
        counter = 0
        mem_state = None
        mem_record = []
        spk_record = []

        for step in range(self.num_steps):
            xt = x_encoded[step]  # gives the [b,c,h,w]

            conv_out = self.conv(xt)
            pooled = self.pool(conv_out)
            flat = pooled.view(batch_size, -1)
            linear_out = self.linear(flat)
            spk, mem_state = self.lif(linear_out, mem_state)
            if step % 20 == 0:
                counter += 1
                # print(f"prints the encoded input every 20 steps {xt} for step{20*counter}")
                # print(f"This is the spk value {spk_record} for step {counter*20}")
                # print(f"This is the membrance voltage {mem_state.v} for step {counter*20}")
            spk_record.append(spk)
            mem_record.append(mem_state.v)

        # print(f" this is the mem record {len(mem_record)}")
        #! spk_record and mem_record have the same sized elements
        # print(f" this is the mem_state.v {mem_record[20].shape}  and the spk is {spk_record[20].shape}")

        spk_out = torch.stack(spk_record, dim=0).sum(0)
        # print(f"this is the spk_out {spk_out.shape}")
        # print(spk_out.shape)

        # print(f'this is the spkout_max {spk_out.max(1)}')
        # spk_out has [Batch_size, num_classes size]
        # mem_record is a list and has the lkength of time steps
    

        return spk_out, mem_record, spk_record


#! CREATE THE MODEL
model = SimpleSNN(input_size=input_size, output_size=num_classes, num_steps=num_steps)

#! LOSS FUNCTION
criterion = nn.CrossEntropyLoss()

#! OPTIMIZER
optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate)


#! DEFINING THE TRAIN IN THE MODEL
def train(model, train_loader, optimizer, criterion, num_steps, epochs):
    loss_history = []
    acc_history = []

    for epoch in range(epochs):
        epoch_loss = 0
        epoch_acc = 0
        num_batches = 0

        for i, (inputs, labels) in enumerate(train_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)

            # if i == 0 and epoch % 2 == 0:
            #     print(f"Input shape: {inputs.shape}, Labels: {labels}")
            #     print(f"Input min: {inputs.min()}, max: {inputs.max()}")
            model.train()
            spk_out, mem_record, _ = model(inputs)

            loss = torch.zeros((1), device=device)

            loss = criterion(spk_out, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            preds = spk_out.argmax(dim=1)
            acc = (preds == labels).float().mean() * 100

            epoch_loss += loss.item()
            epoch_acc += acc.item()
            num_batches += 1

            # print(f"batch = {i + 1}:")
            # print(f"inputs shape: {inputs.shape}")
            # print(f"spikek output: {spk_out}")
            # print(f"preds: {preds}")
            # print(f"labels: {labels}")
            # print(f"batch loss: {loss.item():.4f}, \nbatch accuracy: {acc.item():.2f}%")

        avg_epoch_loss = epoch_loss / num_batches
        avg_epoch_acc = epoch_acc / num_batches

        loss_history.append(avg_epoch_loss)
        acc_history.append(avg_epoch_acc)

        print(
            f"Epoch {epoch + 1}/{epochs} - Loss: {(avg_epoch_loss):.4f} - Accuracy: {avg_epoch_acc:.4f} %"
        )

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(loss_history)
    plt.title(" Training Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")

    plt.subplot(1, 2, 2)
    plt.plot(acc_history)
    plt.title("Training Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")

    plt.tight_layout()
    plt.show()


#! DEFINE THE TESTING
def test(model, test_loader, num_steps):
    correct = 0
    loss = 0
    accuracy = 0
    total = 0

    # for plotting
    imgs_to_plot = []
    true_labels = []
    pred_labels = []

    # Disables gradient calculation
    with torch.no_grad():
        model.eval()

        for data, labels in test_loader:
            data = data.to(device)
            labels = labels.to(device)

            test_spk, test_mem, _ = model(data)

            _, predicted = test_spk.max(1)  # we obtain the predicted spike values

            total += labels.size(0)
            correct = correct + (predicted == labels).sum().item()

            loss += (
                criterion(test_spk, labels).item()
            )  # update the loss based on the last value ofthe spiking membrance voltage

            # Save 2 images from each batch to plot and see the prediction
            for i in range(min(2, data.size(0))):
                img = data[i].squeeze().numpy()
                imgs_to_plot.append(img)  # we remove the 1 greyscale channel
                true_labels.append(labels[i].item())
                pred_labels.append(predicted[i].item())

    avg_loss = loss / len(test_loader)
    accuracy = 100 * (correct / total)

    print(f"Test loss is : {(avg_loss):.4f} \n Accurace is : {(accuracy):.2f} %")

    # we are going to plot the images with the true labels and the predicted ones
    num_images = len(imgs_to_plot)
    columns = 5
    rows = round((num_images + columns) // columns)

    plt.figure(figsize=(15, rows * 3))

    for idx, img in enumerate(imgs_to_plot):
        plt.subplot(rows, columns, idx + 1)
        plt.imshow(img, cmap="cool")
        plt.title(
            f"true label : {true_labels[idx]}\n predicted label :{pred_labels[idx]}"
        )
        plt.tight_layout()
    plt.show()


#! TRAIN THE MODEL
train(model, train_loader, optimizer, criterion, num_steps, epochs)

#!TEST THE MODEL
test(model, test_loader, num_steps)


#! SAVE/STORE THE MODEL
torch.save(model.state_dict(), "model.pth")

#! LOAD THE MODEL
# model.load_state_dict(torch.load("model.pth"))


# SAMPLE DATA

sample_data, sample_label = test_dataset[136]
sample_data.shape  # [C, H, W]
if sample_label == 1:
    sample_labelstr = "vertical lines"
else:
    sample_labelstr = "horizontal lines"

plt.title(f"The image's label is {sample_label} in binary or {sample_labelstr} ")
plt.imshow(sample_data[0], cmap="bone")

# sample_data.shape # [1,16,16] so channel and hwieght and width
# sample_data = sample_data.unsqueeze(0) # we add another dimension
sample_data.shape  # should be [1,1,16,16]


#! DEFINING SOME PLOTS
def membrance_spikes_plot(model, dataset, sample_i, neuron_i: None):
    """
    We are visualizing the membrance potential over time
    
    Args:
        model->: takes the model that we have trained 
        dataset->: takes the dataset from which we are going to extract sample
        sample_i->: takes the sample index whcih has the sample picture and label inside
        neuron_i->: takes the neuron index
    """
    data, labels = dataset[sample_i]
    # data is [1,16,16]

    data = data.unsqueeze(0) #[1,1,16,16]

    model.eval()
    with torch.no_grad():
        _, mem_record, spk_record = model(data)

        mem_values =[]
        for mem in mem_record :
            mem_values.append(mem.detach().numpy())
        
        mem_potentials = np.array(mem_values) #shape should be [num_steps, output neuron]

        # convert spiekes to np arrays 
        spk_values = []
        for spk in spk_record :
            spk_values.append(spk.detach().numpy())

        spikes = np.array(spk_values) #shape shoudlbe the same as above

        mem_potentials = mem_potentials.squeeze(axis=1) #removes teh batch dimension
        spikes = spikes.squeeze(axis=1) # // 

        #Creating a figure with 3 plots
        figure = plt.figure(figsize=(15,10))

        if labels == 0 :
            labelstr = "horizontal lines"
        else: 
            labelstr = "vertical lines"

        #first we show the image that was picked
        ax1 = figure.add_subplot(3,1,1)
        ax1.imshow(data[0].squeeze().numpy(), cmap="cool")
        ax1.set_title(f"Input image is class {labels} so it has {labelstr}")
        ax1.axis("off") #turns off axis since this is an image

        #Membrance voltage plot
        ax2 = figure.add_subplot(3,1,2)
        time_steps =np.arange(mem_potentials.shape[0]) #num_steps
        for i in range(mem_potentials.shape[1]) :
            ax2.plot(time_steps,mem_potentials[:,i], label=f"Neuron {i}")

        ax2.set_xlabel("Time steps")
        ax2.set_ylabel("Membrance Voltage ")
        ax2.set_title("Membrance Voltage Traces")
        ax2.legend()
        ax2.grid(True)


        #SPike raster plot
        ax3 = figure.add_subplot(3,1,3)

        #For each neuron we need time setps wheere spike occured \
        for n_i in range(spikes.shape[1]):
            spike_times = np.where(spikes[:,n_i] > 0)[0]

            ax3.scatter(spike_times,np.ones_like(spike_times)*n_i
                        , marker="|",s=100, color=f"C{n_i}", label=f"Neuron {n_i}")
        
        ax3.set_xlabel("Time Steps")
        ax3.set_ylabel("Neuron indix")
        ax3.set_yticks([0,1])
        ax3.set_yticklabels(["Neuron 0", "Neuron 1"])
        ax3.set_title('Spike Raster Plot')
        ax3.set_xlim(0, spikes.shape[0])
        ax3.grid(True, axis='x')

        plt.tight_layout()
        plt.show()

    return print("Spikes shape:", spikes.shape) , print("Total spikes:", np.sum(spikes))


membrance_spikes_plot(model,test_dataset,110,None)

#! display the kernels/filters

