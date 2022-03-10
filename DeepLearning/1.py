import torch

t1 = torch.tensor(4.)
print(t1)
print(t1.shape)

t2 = torch.tensor([1., 2, 3, 4])
print(t2)
print(t2.shape)

t3 = torch.tensor([
    [1,2],
    [3,4],
    [5,6]
])
print(t3)
print(t3.shape)

t4 = torch.tensor([
    [[11,12,13],
    [13,14,15]],
    [[0,1,2],
    [3,4,5]]
])
print(t4)
print(t4.shape)

x = torch.tensor(3.)
w = torch.tensor(4., requires_grad=True)
b = torch.tensor(5., requires_grad=True)

y = w * x + b
print(y)
y.backward()

print(f"dy/dx = {x.grad}")
print(f"dy/dw = {w.grad}")
print(f"dy/db = {b.grad}")

import numpy as np
x1 = np.array([[1, 2], [3 ,4]])
print(x1)
y1 = torch.from_numpy(x1)
print(y1)
z = y1.numpy()
print(z)