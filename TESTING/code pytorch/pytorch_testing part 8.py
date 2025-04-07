# ?Pipeline guide -->
# ? 1) Design model (input, output size, forward pass)
# ? 2) Construct loss and optimizer
# ? 3) Training loop : compute prediction
# ? - forward pass : gradients
# ? - update weights

import torch
import torch.nn as nn
import numpy as np
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

#! Logistic Regression
# 0) data loading and preprocessing
bc = datasets.load_breast_cancer()

X, y = bc.data, bc.target
n_samples, n_features = X.shape
# We check our data structure 569 samples and 30 features (a lot !)
print(n_samples, n_features)

# data splitting
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Scale the features
# 0 mean and unit variance scaling data ideal for logistic regression
sc = StandardScaler()

X_train = sc.fit_transform(X_train)  # fit and transform
X_test = sc.transform(X_test)  # only transform

X_train = torch.from_numpy(X_train.astype(np.float32))
X_test = torch.from_numpy(X_test.astype(np.float32))
y_train = torch.from_numpy(y_train.astype(np.float32))
y_test = torch.from_numpy(y_test.astype(np.float32))

# .shpape[0] refers to the size of the FIRST dimension which is the rows
y_train = y_train.view(y_train.shape[0], 1)
y_test = y_test.view(y_test.shape[0], 1)

# 1) model
# f = wx + b, sigmoid funciton at the end


class LogisticRegression(nn.Module):
    def __init__(self, n_input_features):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(n_input_features, 1)

    def forward(self, x):
        y_pred = torch.sigmoid(self.linear(x))
        return y_pred


model = LogisticRegression(n_features)

# 2) loss and optimizer
learning_rate = 0.01

criterion = nn.BCELoss()  # binary cross entropy loss
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# 3)training loop
num_epoch = 100

for epoch in range(num_epoch):
    # forward pass and loss
    y_pred = model(X_train)
    loss = criterion(y_pred, y_train)

    # backwards pass
    loss.backward()

    # updates
    optimizer.step()  # pytorch does this automatically

    # zero gradients
    if (epoch+1) % 10 == 0:
        print(f"epoch {epoch+1}, loss = {loss.item():.4f}")

with torch.no_grad():  # we are doing the evaluation
    y_predicted = model(X_test)
    y_predicted_cls = y_predicted.round()
    # it summs all the predictions that are correct from the eq function (equals)
    acc = y_predicted_cls.eq(y_test).sum() / float(y_test.shape[0])
    print(f"Accuracy is equal to {acc:.4f}")
