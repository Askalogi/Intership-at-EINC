import numpy as np
from typing import Union
import random
from PIL import Image
import ipywidgets as widgets
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline
# function that creates the images
# 0 is black and 255 is white

# TODO -> Implement Distribution to make the lines fuller! Change the display method to matplotlib
# TODO -> Implement variables that control the distance of the full lines so pretty much control the distribution


#!NOTES: Implement a distribution that indicates how many pixels are colored
#! the bigger the value at the distribution the fuller the line is going to be
#! the weaker the less black pixel we are going to have and plotting with matplotlibe

# not sos-> Something about interportation in the nd array for the blur

# Again
def create_image(size_x: int, size_y: int, orientation: str):
    """
    Takes dimensionality, orientetion, line distribution as in both the density and width of the line
    and also the space between line , also distribution controls whether or not the line are fully filled with the pixels.

    Parameters:
    size_x -> refers to the value of the x axis (columns)
    size_y -> refers to the value of the y axis (rows)
    oientetion - > refers to the orientation of the line that are gonna be drawn
    """
    # Creates a size_x by size_y grid filled with ones and
    # it gets multiplied by 255 to indicate that it is white
    # due to the 0 - 255 values of the 8 bit data type

    # distribution ->
    # We need a threshhold so that when it exceeds that value or satisfies it then we have fully black lines!
    #

    img = np.ones((size_y, size_x), dtype=np.uint8) * 255

    if orientation == "horizontal":
        step_y = random.randint(size_y // 10, size_y//3)
        # we need to ranomly pick the lines that get turned into black
        # sample takes from size_y list a step_y amount of values from the list
        random_y_lines = random.sample(range(size_y), step_y)
        for y in random_y_lines:
            img[y, :] = 0  # Sets the entire row to black

    elif orientation == "vertical":
        step_x = random.randint(size_x//10, size_x//3)
        # same thing as above
        random_x_lines = random.sample(range(size_x), step_x)
        for x in random_x_lines:
            img[:, x] = 0  # Sets the entire column to black

    elif orientation == "random":
        # Practically we are giving a random ammount of pixels to be turned black
        density = 0.3
        num_pixels = int(size_x * size_y * density)
        for i in range(num_pixels):
            # In both cases we make sure we substract 1
            x = random.randint(0, size_x - 1)
            y = random.randint(0, size_y - 1)
            img[y, x] = 0

    # Converting the numpy array to an image (maybe we change that)
    return Image.fromarray(img)


bing = create_image(1000, 1000, "horizontal")

bing.show()

x = np.random.rand(2, 2)
x
