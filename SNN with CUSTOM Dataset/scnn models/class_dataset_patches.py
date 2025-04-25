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


class CustomSinPatchesDataset(Dataset):
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
