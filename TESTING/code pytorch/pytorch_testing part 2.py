import numpy as np
from typing import Union
import random
from PIL import Image
import matplotlib.pyplot as plt
import torch

#!TODO TENSOR BASICS
#! TENSORS
x = torch.empty(3)  # 1D  empty 3 element tensor/vector
x
x = torch.empty(2, 3)  # 2D empty
x
x = torch.empty(3, 1, 5)  # 3D empty
x

x = torch.rand(2, 2)  # 2x2 random value from 0 to 1 tensor
x

x = torch.ones(2, 2, dtype=torch.float)  # 2x2 filed with ones tensor
x.dtype  # float 32 by default
x.size()  # funciton that shows the size [2,2]

x = torch.tensor([2, 3, 0.1])  # custom made tensor making
x

#! OPERATIONS
x = torch.rand(2, 2)
y = torch.rand(2, 2)
x
y
z = x + y  # Element addition
z = torch.add(x, y)  # Same as above
z

y.add_(x)  # In place operation of addition
# ? TIP --> any _ in the torchg module does an inplace operation (modifies the variable that is on)
y

z = x-y  # Element substraction
z = torch.sub(x, y)  # Same as above


z = x*y  # Element multiplication
z = torch.mul(x, y)  # Same as above


z = x/y  # Element division
z = torch.div(x, y)  # Same as abve

#! SLICING OPERATIONS
x = torch.rand(5, 3) * 10
x
x[:, 0]  # We get teh first column
x[1, :]  # Second row all the columns
x[2, 2]  # 3rd row 3rd column
# ? In all of the above we get a tensor if we want the value inside we must

x[2, 2].item()  # We get the tensors value

#! RESHAPING
x = torch.rand(4, 4)
x
# Takes the values from the 4x4 tensor above and makes it 1dimensional putting them line by line
y = x.view(16)
y
# its a 2 by 8 pytorch automatically fills in the other diemnsion through (-1)
z = x.view(-1, 8)
z

#! From NumPy to PyTorch and vise versa
a = torch.ones(5)
a
b = a.numpy()  # to numpy
a.add_(1)  # Both of the values will change cause they both point to the same memory location
a
b  # This is a float 32

a = np.ones(5)
b = torch.from_numpy(a)  # To pytorch
b  # but this is a float64

if torch.cuda.is_available():
    device = torch.device("cuda")
    # cuda chekcs if we havea gpu and if so then the tensor will be made and be located at the GPU
    x = torch.ones(5, device=device)
    # ? NUMPY CAN ONLY HANDLE CPU TENSORS NOT GPU


x = torch.ones(5, requires_grad=True)  # By Default this is false
# True means that this will have to calculate the gradiance
x
