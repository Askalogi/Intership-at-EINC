import numpy as np
import random
from PIL import Image
from scipy.interpolate import make_interp_spline, interp1d
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter, zoom, median_filter


def parametric_2d_noise(size_x: int, size_y: int, orientation: str, blur: int, black_lines_position: list = None):
    """
    Generating 2d noise and also some black lines while also making them blurry

    Parameters :
    size_x -> takes the size of the x axis
    size_y -> takes the size of the y axis
    orientation -> choose the orientation of the fully black lines with a string input
    black_lines_position -> takes a list input to indicate where we want those lines


    """
    # initializign
    noise_array = np.random.randint(0, 255, size=(
        size_y, size_x))  # remember its y * x and also the lowest and highest value is stated

    if black_lines_position is None:
        black_lines_position = random.randint(0, size_x - 1)
    else:
        black_lines_position = black_lines_position

    if orientation == "vertical":
        for x in black_lines_position:
            if x < size_x:
                noise_array[:, x] = 0
            else:
                raise ValueError(
                    "The input of the black line is outside of the grid :( "
                )

    elif orientation == "horizontal":
        for y in black_lines_position:
            if y < size_y:
                noise_array[y, :] = 0
            else:
                raise ValueError(
                    "the input of the black line is outside of the grid :( "
                )

    elif orientation == "diagonal":
        for i in black_lines_position:  # i =2
            # diagonal from top-left to bottom-right
            for j in range(size_x):
                if i + j < size_x and j < size_y:
                    noise_array[j, i + j] = 0
                else:
                    break
            # diagonal from top-left to bottom right
            for j in range(size_y):
                if i + j < size_y and j < size_x:
                    noise_array[i + j, j] = 0
                else:
                    break

    else:
        raise ValueError("Not viable orientation please try again :)")

    # noise_array = gaussian_filter(noise_array, sigma=blur)

    return noise_array


test1_horizontal = parametric_2d_noise(
    100, 100, "horizontal", 2, [0, 5, 7, 9, 12, 15, 20, 67, 80, 90, 99, 56])
test2_vertical = parametric_2d_noise(20, 20, "vertical", 2, [1, 5, 9, 12, 15])
test3_diagonal = parametric_2d_noise(
    16, 16, "diagonal", 2, [2, 5, 7, 9, 10, 13])
# test4_error = parametric_2d_noise(20, 10, "bingbong", [1, 2])

plt.figure(figsize=(10, 10))
plt.imshow(test2_vertical, cmap="gray", interpolation="nearest")
plt.colorbar()
plt.title("16x16 vertical black lines at given places and median filter")
plt.show()


plt.figure(figsize=(10, 10))
plt.imshow(test1_horizontal, cmap='gray', interpolation="nearest")
plt.colorbar()
plt.title("16x16 horizontal black lines at given places and median filter")
plt.show()


plt.figure(figsize=(10, 10))
plt.imshow(test3_diagonal, cmap="gray", interpolation="nearest")
plt.colorbar()
plt.title("16x16 diagonal black lines at given places and median filter")
plt.show()
