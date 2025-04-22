import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, utils
from PIL import Image
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from pathlib import Path
import glob as glob


class CustomSinDataset(Dataset):
    def __init__(self, root: str, transform: Optional[Callable] = None):
        self.root = Path(root)
        self.transform = transform

        # Load all image paths
        all_images = list(self.root.glob("*.png"))

        # Extract the index and label from filenames (assuming format: "index_label.png")
        self.images_with_labels = []
        for img_path in all_images:
            # Parse the filename to get index and label
            parts = img_path.stem.split("_")
            if len(parts) == 2:
                index = int(parts[0])
                label = int(parts[1])
                self.images_with_labels.append((img_path, label, index))

        # Sort by index to ensure consistent ordering
        self.images_with_labels.sort(key=lambda x: x[2])

    def __len__(self) -> int:
        return len(self.images_with_labels)

    def __getitem__(self, index: int):
        img_path, label, _ = self.images_with_labels[index]

        # Load and process the image
        image = Image.open(img_path).convert("L")  # GRAYSCALE

        if self.transform:
            image = self.transform(image)

        return image, label


#! TESTING
# # creating the tranform function that is going to be taken inside the CustomDataset
# transfrom = transforms.Compose([
#     transforms.ToTensor(), # convert to tensor
#     transforms.Normalize(mean=[0.5], std=[0.2])

# ])
# #Creating the dataset
# sin_dataset = CustomSinDataset(root="./custom_dataset", transform=transfrom)
# sin_dataset

# #Splitting the dataset into training and testing
# train_size = int(0.8 * len(sin_dataset))
# test_size = len(sin_dataset) - train_size
# train_size
# test_size

# #here we randomly split the data with the sizes that are givn inside the list that are defined above

# train_data , test_data = random_split(sin_dataset, [train_size, test_size])

# #Now we use the loaders :
# train_loader = DataLoader(train_data,batch_size= 8,shuffle=True)
# test_loader = DataLoader(test_data, batch_size=2, shuffle=True)

# for image, label in train_loader:
#     print(image.shape, label.shape)  # Example output: torch.Size([8,3, 16, 16])  batch size the color channels and then sizex and sizey
#     break

# transform = transforms.Compose(
#     [
#         transforms.ToTensor(), # let's only try just the ToTensors for now
#         transforms.Grayscale(), # lets also make it gray scale
#         transforms.Normalize((0,), (1,)),  # and lets normalize the tensors
#     ]
# )

# # create the train/test dataset
# dataset = CustomSinDataset(
#     root="./custom_dataset",
#     transform=transform,
# )

# # Split the dataset into train and test with customizable size
# train_size = int(0.7 * len(dataset))
# test_size = len(dataset) - train_size

# # Save them into actual train/test datasets
# train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

# sample_data, sample_label = train_dataset[0]
# plt.imshow(sample_data[0], cmap="grey")
# print(f"Testing the dataset's size sample {sample_data.size()}")
# print(f"this is the label of the image below remember \n 0 = horizontal \n 1 = vertical\n label -> {sample_label}")
