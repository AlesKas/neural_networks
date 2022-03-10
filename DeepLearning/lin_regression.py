from statistics import mode
import torch
import numpy as np
from urllib3 import Retry

# Input (teplota, srazky, vlhkost)
inputs = np.array([[73, 67, 43], 
                   [91, 88, 64], 
                   [87, 134, 58], 
                   [102, 43, 37], 
                   [69, 96, 70]], dtype='float32')

# Output (jabka, pomerance)
targets = np.array([[56, 70], 
                    [81, 101], 
                    [119, 133], 
                    [22, 37], 
                    [103, 119]], dtype='float32')
                    
inputs = torch.from_numpy(inputs)
targets = torch.from_numpy(targets)

w = torch.rand(2, 3, requires_grad=True)
b = torch.rand(2, requires_grad=True)
# print(w)
# print(b)

def model(x):
    return x @ w.t() + b

preds = model(inputs)
# print(preds)
# print(targets)

def MSE(pred, targ):
    diff = pred - targ
    return torch.sum(diff * diff) / diff.numel()

learning_rate = 1e-4

for i in range(150):
    preds = model(inputs)
    loss = MSE(preds, targets)
    loss.backward()
    with torch.no_grad():
        w -= w.grad * learning_rate
        b -= b.grad * learning_rate
        w.grad.zero_()
        b.grad.zero_()
    if (i + 1) % 10 == 0:
        preds = model(inputs)
        loss = MSE(preds, targets)
        print(loss)

print(preds)
print(targets)