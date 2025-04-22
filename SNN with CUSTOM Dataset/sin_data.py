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

def sin_distribution(size: int, numbrer_of_peaks: int, orientation: str, noise_level: float):
    """
    Creating a funtion that takes the input of x and y axis that eventually the Image is going to have
    The end result should be a distribution in grayscale that follows the sin's wave form

    Parameters:
    size --> this is the size of the x and y axis or rather the # of columns and rows
    number_of_peaks --> this is the number of peaks or the # of black lines
    orientation --> chooses between horizontal and vertical lines 
    noise_level --> the noise intenisty higher = more noise

    """
    bing = np.linspace(0, 2 * numbrer_of_peaks*np.pi, size)

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
            "This Orientations either doesnt exist or isnt implemented quite yet")

    # we apply the guassian filter the bigger the sigma the more the blur/noise we have
    final = gaussian_filter(final, sigma=0.3)

    # we also marametrize the noise semi randomly
    noise = np.random.normal(0.5, noise_level, final.shape)
    # and then clip the result
    # this is aslo customizable regarding the intensity of the pixels
    final = np.clip(final + noise, 0, 10)

    # greyscale this to 8 bits
    final = (final/10.0 * 255.0).astype(np.uint8)

    return Image.fromarray(final)
    # return Image.fromarray(final)


#ΤODO We need to export and save the dataset 
#TODO also we need to have the dataset be dynamic so we can change the ammount of pictures we are saving
#TODO while also having them be equal number of horizontal and vertical line images


def create_dataset(num_samples: int, size = 16):
    #index the path to the custom dataset
    dataset_path = "./custom_dataset"

    #makes sure the directory exists 
    os.makedirs(dataset_path, exist_ok=True)
    #empty the last contents of the path so we can reuse this endlessly
    for file in glob.glob(os.path.join(dataset_path, "*.png")):
        os.remove(file)

    # create the main loop where we get random peaks, orientation and noise
    for i in range(num_samples):
        num_peaks = random.randint(1,5) #random amount of peaks
        orientation = random.choice(["horizontal", "vertical"]) #random choice between orientations
        noise_lvl = random.randrange(1, 4)/10 #random noise 
        
        #use the function defined above
        img = sin_distribution(size=size, numbrer_of_peaks=num_peaks, orientation=orientation, noise_level=noise_lvl)

        #label the pictures with 0,1 (hot one encoding)
        label = 0 if orientation == "horizontal" else 1 

        #create the file path and the names that these images will have 
        file_path = os.path.join(dataset_path, f"{i+1}_{label}.png")

        #save them there
        img.save(file_path)
        
#!TESTING THE CREATION OF THE DATASET

create_dataset(1000, 16)





