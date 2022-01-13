from statistics import mode
import torch
import torch.nn as nn

# f = w * x

# f = 2 * x
X = torch.tensor([[1], [2], [3], [4]], dtype=torch.float32)
Y = torch.tensor([[2], [4], [6], [8]], dtype=torch.float32)

x_test = torch.tensor([5], dtype=torch.float32)

n_samples, n_features = X.shape

input_size = n_features
output_size = n_features

model = nn.Linear(input_size, output_size)

print(f'Prediction before f(5) = {model(x_test).item():.3f}')

# Training
learning_rate = 0.05
iters = 150

loss = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

for epoch in range(iters):
    # prediction
    y_pred = model(X)
    # loss
    l = loss(Y, y_pred)
    # gradient
    l.backward()

    optimizer.step()
    
    optimizer.zero_grad()

    if epoch % 10 == 0:
        [w, b] = model.parameters()
        print(f'epoch {epoch+1}: w = {w[0][0].item():.3f}, loss = {l.item():.8f}')

print(f'Prediction after f(5) = {model(x_test).item():.3f}')