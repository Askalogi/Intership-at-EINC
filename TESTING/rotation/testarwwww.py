import numpy as np
from typing import Union
import random
from IPython.display import display
from PIL import Image
import PIL
import ipywidgets as widgets
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline
from scipy.ndimage import gaussian_filter, zoom, median_filter
import pandas as pd
import torch
from torchvision.transforms import functional as TF
from torchvision.transforms import Pad, Resize
import os
import glob


def sin_distribution(
    size: int, numbrer_of_peaks: int, orientation: str, noise_level: float
):
    """
    Creating a funtion that takes the input of x and y axis that eventually the Image is going to have
    The end result should be a distribution in grayscale that follows the sin's wave form

    Parameters:
    size --> this is the size of the x and y axis or rather the # of columns and rows
    number_of_peaks --> this is the number of peaks or the # of black lines
    orientation --> chooses between horizontal and vertical lines
    noise_level --> the noise intenisty higher = more noise

    """
    bing = np.linspace(0, 2 * numbrer_of_peaks * np.pi, size)

    if orientation == "vertical":
        # we are applying the sin function into the starting evenly spaced array that we can control its peaks
        # after we add 1 so we dont get any negative numbers
        # then we tile this new aray repeating it with the size parameter and also reshaping it
        # into a size by size array
        final = np.tile(np.sin(bing) + 1, size).reshape(size, size)

    elif orientation == "horizontal":
        # here while the first 2 lines stay the same
        # we are using indexing on the bing array so we can make this into
        bing = np.sin(bing) + 1
        final = np.tile(bing[:, np.newaxis], (1, size))
    else:
        raise ValueError(
            "This Orientations either doesnt exist or isnt implemented quite yet"
        )

    # we apply the guassian filter the bigger the sigma the more the blur/noise we have
    final = gaussian_filter(final, sigma=0.3)

    # we also marametrize the noise semi randomly
    noise = np.random.normal(0.5, noise_level, final.shape)
    # and then clip the result
    # this is aslo customizable regarding the intensity of the pixels
    final = np.clip(final + noise, 0, 10)

    # greyscale this to 8 bits
    final = (final / 10.0 * 255.0).astype(np.uint8)

    return Image.fromarray(final)
    # return final
    # return torch.from_numpy(final)
    # return Image.fromarray(final)


a1 = sin_distribution(1000, 3, "horizontal", 0.3)

a1.show()

num_steps = int(
    input(
        "Give me the number of steps that you want to use :",
    )
)
num_steps

rotation = input(
    "Choose between clock or c_clock",
)

if rotation == "clock":
    for i in range(num_steps):
        a1 = TF.rotate(a1, angle=-5,fill=127 )
        display(a1)
        # a1.save(f"../rotation/img_{i}.png")
elif rotation == "c_clock":
    for i in range(num_steps):
        a1 = TF.rotate(a1, angle=5)
        display(a1)
        # a1.save(f"../rotation/img{i}.png")
else:
    raise ValueError("BING BONG NOT A ROTATION ! :(  ")

resized_image = Resize()