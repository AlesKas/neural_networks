import numpy as np

# f = w * x

# f = 2 * x
X = np.array([1,2,3,4], dtype=np.float32)
Y = np.array([2,4,6,8], dtype=np.float32)

W = 0

def forward(x):
    return W * x

def loss(y, y_pred):
    return ((y_pred - y) ** 2).mean()

def gradient(x, y, y_pred):
    return np.dot(2*x, y_pred-y).mean()

print(f'Prediction before f(5) = {forward(5):.3f}')

# Training
learning_rate = 0.01
iters = 12

for epoch in range(iters):
    # prediction
    y_pred = forward(X)
    # loss
    l = loss(Y, y_pred)
    # gradient
    dw = gradient(X,Y,y_pred)
    # update weights
    W  -= learning_rate * dw

    if epoch % 1 == 0:
        print(f'epoch {epoch+1}: w = {W:.3f}, loss = {l:.8f}')

print(f'Prediction after f(5) = {forward(5):.3f}')