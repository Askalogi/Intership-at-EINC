import numpy as np
import torch
import torch.nn as nn

#! Using Loss and optimizer classes with pytorch
#! and chaning the forward to a model from pytorch
# ?Pipeline guide -->
# ? 1) Design model (input, output size, forward pass)
# ? 2) Construct loss and optimizer
# ? 3) Training loop : compute prediction
# ? - forward pass : gradients
# ? - update weights


# f = w * x

# f = 2  *  x
X = torch.tensor([[1], [2], [3], [4]], dtype=torch.float32)
Y = torch.tensor([[2], [4], [6], [8]], dtype=torch.float32)


X_test = torch.tensor([5], dtype=torch.float32)

n_samples, n_feat = X.shape
print(n_samples, n_feat)
# model prediction

input_size = n_feat
output_size = n_feat

# Custom model


class LinearRegressionModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LinearRegressionModel, self).__init__()
        # define layers
        self.lin = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.lin(x)


model = LinearRegressionModel(input_size, output_size)

print(f"Prediction Before training f(5) = {model(X_test).item():.3f}")

# Training
learning_rate = 0.02
n_iters = 100

loss = nn.MSELoss()

optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

for epoch in range(n_iters):
    # prediction = forward pass
    y_pred = model(X)

    # loss
    l = loss(Y, y_pred)

    # gradients = backwardpass
    l.backward()  # will calculate the gradient of the less in regard to w
    # ? ^ dl/dw

    # step of the optimizer
    optimizer.step()

    # zero gradients
    optimizer.zero_grad()

    if epoch % 10 == 0:
        [w, b] = model.parameters()
        print(f"epoch {epoch + 1}: w = {w[0][0].item():.3f}, loss = {l:.8f}")

print(f"Prediction After training f(5) = {model(X_test).item():.3f}")
