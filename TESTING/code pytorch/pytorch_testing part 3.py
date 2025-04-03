import numpy as np
import torch

#! GRADIENT CALCULATOR WITH AUTOGRAD FROM PYTORCH
np
x = torch.randn(3, requires_grad=True)  # We must pass that arguement to TRUE
x

y = x + 2
y  # every single operation will create a node with inputs and outputs
# Pay attention to the grad_fn when this is printed
# Later we wil do backpropogation

z = y*y*2  # this will have a mulbackwards grad_fn saved
z
z = z.mean()  # this will ahve a meanbackwards grad_fn saved

# if we dont have a scalar then we must put a vector arguewment insiode the backward function
z.backward()  # ? dz/dx and it must be a scalar(single value)
print(x.grad)

#! PREVENT PYTORCH GRADIENT HISTORY (no tracking)
# 1)
# it doesnt have the requires grad attribute when printed
x.requires_grad_(False)
x
# 2)
y = x.detach()  # Creates a new tensor copy of x but doenst require the gradiance
y
# 3)
with torch.no_grad():
    y = x + 2
    print(y)  # Will also not have the gradience funciton

#! TRAINING EXAMPLE
weights = torch.ones(4, requires_grad=True)

for epoch in range(3):
    model_output = (weights*3).sum()

    model_output.backward()

    print(weights.grad)

    # we must empty the gradients
    weights.grad.zero_()  # we must do that


#! OPTIMIZERS
weights = torch.ones(4, requires_grad=True)

optimizer = torch.optim.SGD(weights, lr=0.01)
optimizer.step()
optimizer.zero_grad()
