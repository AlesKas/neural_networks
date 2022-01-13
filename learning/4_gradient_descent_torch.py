import torch

# f = w * x

# f = 2 * x
X = torch.tensor([1,2,3,4], dtype=torch.float32)
Y = torch.tensor([2,4,6,8], dtype=torch.float32)

W = torch.tensor([0.0], dtype=torch.float32, requires_grad=True)

def forward(x):
    return W * x

def loss(y, y_pred):
    return ((y_pred - y) ** 2).mean()


print(f'Prediction before f(5) = {forward(5).item():.3f}')

# Training
learning_rate = 0.01
iters = 100

for epoch in range(iters):
    # prediction
    y_pred = forward(X)
    # loss
    l = loss(Y, y_pred)
    # gradient
    l.backward()
    # update weights
    with torch.no_grad():
        W  -= learning_rate * W.grad
    
    W.grad.zero_()

    if epoch % 10 == 0:
        print(f'epoch {epoch+1}: w = {W.item():.3f}, loss = {l.item():.8f}')

print(f'Prediction after f(5) = {forward(5).item():.3f}')