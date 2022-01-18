import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

from sklearn import datasets

X_np, Y_np = datasets.make_regression(100, 1, noise=20, random_state=1)
X = torch.from_numpy(X_np.astype(np.float32))
Y = torch.from_numpy(Y_np.astype(np.float32))
Y = Y.view(Y.shape[0], 1)

n_samples, n_features = X.shape

input_size = n_features
output_size = 1

model = nn.Linear(input_size, output_size)

learning_rate = 0.01
criterion = nn.MSELoss()
optimazer = torch.optim.SGD(model.parameters(), lr=learning_rate)

num_epochs = 100
for epoch in range(num_epochs):
    y_pred = model(X)

    loss = criterion(y_pred, Y)

    loss.backward()

    optimazer.step()

    optimazer.zero_grad()

    if epoch % 10 == 0:
        print(f'epoch {epoch+1}, loss = {loss.item():.8f}')

predicted = model(X).detach().numpy()
plt.plot(X_np, Y_np, 'ro')
plt.plot(X_np, predicted, 'b')
plt.show()