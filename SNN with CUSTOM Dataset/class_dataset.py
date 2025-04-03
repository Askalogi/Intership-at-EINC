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
    """Custom Sin distribution dataset
    "Reminder label 0 is horizontal and 1 is vertical lines

    
    Arguments:
        root: -> string that has the directory of the dataset 
        label_csv: -> string that has the labels saved as a csv already hot encoded
        transform: -> accepts transform.compose so we can make it into tensors and normalize them
    """
    def __init__(self,
                 root: str,
                 label_csv : str,
                 transform : Optional[Callable] = None,
                 
                 ):
        self.root = Path(root)
        self.label_csv = label_csv
        self.transform = transform 

        #Loading the image paths with the glob method searching for all .png instances
        self.image_paths = list(self.root.glob("*.png"))

        #Loading the labels path fro the label csv file
        self.image_labels = pd.read_csv(self.label_csv)

        #making sure images and labels match regarding their population
        assert len(self.image_paths) == len(self.image_labels), "WRONG MISMATCH BETWEEN IMAGES AND LABELS"

    def __len__(self) -> int:
        "Return the total ammount of samples inside"
        return len(self.image_paths)
    
    def __getitem__(self, index :int):
        "Here we will be taking an image and label pair"

        img_path = self.image_paths[index]
        label = self.image_labels.iloc[index , 0]

        #Here we can acces the image that is stated above 
        image = Image.open(img_path).convert("L") #GRAY SCALE so one channel

        #We also apply trasformation if they are provided and defined 
        if self.transform:
            image = self.transform(image)

            return image , label #returns a tuple 
        

#! TESTING
# creating the tranform function that is going to be taken inside the CustomDataset
transfrom = transforms.Compose([
    transforms.ToTensor(), # convert to tensor
    transforms.Normalize(mean=[0.5], std=[0.2])

])
#Creating the dataset
sin_dataset = CustomSinDataset(root="./custom_dataset", label_csv="./custom_dataset/labels.csv", transform=transfrom)
sin_dataset

#Splitting the dataset into training and testing 
train_size = int(0.8 * len(sin_dataset))
test_size = len(sin_dataset) - train_size
train_size
test_size

#here we randomly split the data with the sizes that are givn inside the list that are defined above

train_data , test_data = random_split(sin_dataset, [train_size, test_size])

#Now we use the loaders :
train_loader = DataLoader(train_data,batch_size= 8,shuffle=True)
test_loader = DataLoader(test_data, batch_size=2, shuffle=True)

for image, label in train_loader:
    print(image.shape, label.shape)  # Example output: torch.Size([8,3, 16, 16])  batch size the color channels and then sizex and sizey
    break 

