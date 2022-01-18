from tkinter.tix import Y_REGION
from turtle import forward
import torch
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

bc = datasets.load_breast_cancer()
X, Y = bc.data, bc.target

n_samples, n_features = X.shape

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=1234)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

X_train = torch.from_numpy(X_train.astype(np.float32))
X_test = torch.from_numpy(X_test.astype(np.float32))
Y_train = torch.from_numpy(Y_train.astype(np.float32))
Y_test = torch.from_numpy(Y_test.astype(np.float32))

Y_train = Y_train.view(Y_train.shape[0], 1)
Y_test = Y_test.view(Y_test.shape[0], 1)

class LogisticRegression(nn.Module):
    def __init__(self, n_input) -> None:
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(n_input, 1)

    def forward(self, data):
        y_pred = torch.sigmoid(self.linear(data))
        return y_pred

model = LogisticRegression(n_features)

learning_rate = 0.02
criterion = nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

num_iteration = 150
for epoch in range(num_iteration):
    y_pred = model.forward(X_train)
    loss = criterion(y_pred, Y_train)

    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    
    if epoch % 10 == 0:
        print(f"epoch: {epoch+1} loss = {loss.item():.4f}")

with torch.no_grad():
    y_predicted = model(X_test)
    y_predicted_cls = y_predicted.round()
    acc = y_predicted_cls.eq(Y_test).sum() / float(Y_test.shape[0])
    print(f"accuracy = {acc:.4f}")