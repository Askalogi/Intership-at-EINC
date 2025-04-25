import numpy as np
from typing import Union
import random
from PIL import Image
import ipywidgets as widgets
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline
from scipy.ndimage import gaussian_filter, zoom, median_filter
import pandas as pd
import os
import glob


def sin_patches(
    full_size: int,
    patch_size: int,
    numbrer_of_peaks: int,
    orientation: str,
    noise_level: float,
):
    """
    Creating a funtion that takes the input of x and y axis that eventually the Image is going to have
    The end result should be a distribution in grayscale that follows the sin's wave form

    Parameters:
    full_size --> this is the size of the x and y axis or rather the # of columns and rows
    patch_size --> thi is the size of the patch inside having the specific patter we will want to label
    number_of_peaks --> this is the number of peaks or the # of black lines
    orientation --> chooses between horizontal and vertical lines
    noise_level --> the noise intenisty higher = more noise
    """

    bing = np.linspace(0, 2 * numbrer_of_peaks * np.pi, patch_size)

    if orientation == "vertical":
        # we are applying the sin function into the starting evenly spaced array that we can control its peaks
        # after we add 1 so we dont get any negative numbers
        # then we tile this new aray repeating it with the size parameter and also reshaping it
        # into a size by size array
        test = (
            np.tile(np.cos(bing) + 1, patch_size).reshape(patch_size, patch_size) / 2
        ) * 255

    elif orientation == "horizontal":
        # here while the first 2 lines stay the same
        # we are using indexing on the bing array so we can make this into
        bing = np.cos(bing) + 1
        test = (np.tile(bing[:, np.newaxis], (1, patch_size)) / 2) * 255

    else:
        raise ValueError(
            "This Orientations either doesnt exist or isnt implemented quite yet"
        )

    # we have created a 5x5 image and we want to place this os substitude the values from the full on noisy
    # big image with this one BUT it has to be placed randomly and also fit

    noise_array = np.random.randint(0, 255, size=(full_size, full_size))
    # filtered_array = gaussian_filter(noise_array, sigma=noise_level)
    # this 16-5 = 11 means we can only use the first 11 columns and the first 11 rows
    # for the beginning of the first index
    # noise_array = gaussian_filter(noise_array,sigma=noise_level)
    avaliable_idx = full_size - patch_size

    # noise_array[1:patch_size,3:patch_size] = test
    # chooses 2 random starting indices
    x_i = random.randint(0, avaliable_idx)
    y_i = random.randint(0, avaliable_idx)

    noise_array[x_i : x_i + patch_size, y_i : y_i + patch_size] = test

    final = noise_array
    # add filter
    # # final = filtered_array
    final = gaussian_filter(noise_array, sigma=noise_level)
    # final = median_filter(noise_array, size=2)
    # YOU HAVE TO NORMALIZE THIS
    final = np.clip(final, 0, 255).astype(np.uint8)
    # noise = np.random.normal(0, noise_level * 255, final.shape)
    # final = final + noise

    return Image.fromarray(final)
    print(final)
    # return final


testing_hor = sin_patches(
    full_size=16,
    patch_size=5,
    numbrer_of_peaks=5,
    orientation="horizontal",
    noise_level=0.6,
)
testing_ver = sin_patches(
    full_size=16,
    patch_size=5,
    numbrer_of_peaks=5,
    orientation="vertical",
    noise_level=0.7,
)


plt.figure(figsize=(10, 10))
plt.imshow(testing_hor, cmap="cool", interpolation="nearest")
plt.colorbar()
plt.title("Testing Noisy image with anotheri mage inside of it")
plt.show()


plt.figure(figsize=(10, 10))
plt.imshow(testing_ver, cmap="cool", interpolation="nearest")
plt.colorbar()
plt.title("Testing Noisy image with anotheri mage inside of it")
plt.show()


def create_dataset_patches(num_samples: int, f_size: int, p_size: int):
    # first we create the dataset path where the images are going to be stored
    dataset_path = "../custom_dataset_patches"

    # make sure the directory is there
    os.makedirs(dataset_path, exist_ok=True)

    # empty the last contents of the path so we can reuse this endlessly
    for file in glob.glob(os.path.join(dataset_path, "*.png")):
        os.remove(file)

    noise_lvl = 0.6
    # creating the main loop for randomized images
    for i in range(num_samples):
        num_peaks = random.choice(
            [1, 2, 3, 5]
        )  # random amount of peaks except for 4 cause its buggy
        orientation = random.choice(
            ["horizontal", "vertical"]
        )  # random choice between orientations
        noise_lvl = random.randrange(40, 70)/100 #random noise

        img = sin_patches(
            full_size=f_size,
            patch_size=p_size,
            numbrer_of_peaks=num_peaks,
            orientation=orientation,
            noise_level=noise_lvl,
        )
        # here we choose the label 0 for horizontal and 1 for vertical
        label = 0 if orientation == "horizontal" else 1

        # create the file path and the names that these images will have
        file_path = os.path.join(dataset_path, f"{i + 1}_{label}.png")

        img.save(file_path)


create_dataset_patches(1000, 16, 5)
