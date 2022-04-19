from pickletools import optimize
import torch
import random

import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

from torchvision.datasets import MNIST
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

if torch.cuda.is_available():  
  dev = "cuda:0" 
else:  
  dev = "cpu"

BATCH_SIZE = 100

dataset = MNIST(root='data/', download=True, train=True, transform=transforms.ToTensor())

def split_dataset(num, val_pct):
    size = int(num*val_pct)
    idx = np.random.permutation(num)
    return idx[size:], idx[:size]

train, validation = split_dataset(len(dataset), 0.2)

train_sampler = SubsetRandomSampler(train)
train_dl = DataLoader(dataset, BATCH_SIZE, sampler=train_sampler)

validation_sampler = SubsetRandomSampler(validation)
validation_dl = DataLoader(dataset, BATCH_SIZE, sampler=validation_sampler)

# input_size = 28 * 28
# hidden_size = 64
# num_classes = 10

class MnistNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes) -> None:
        super().__init__()
        self.linear = nn.Linear(input_size, hidden_size)
        self.hidden_linear = nn.Linear(hidden_size, num_classes)

    def forward(self, xb : torch.Tensor):
        xb = xb.view(xb.size(0), -1)
        out = F.relu(self.linear(xb))
        return self.hidden_linear(out)

model = MnistNN(28*28, 64,10).to(device=dev)
loss_fun = F.cross_entropy
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

def loss_batch(model, loss_fun, xb, yb, opt=None, metric=None):
    preds = model(xb)
    loss = loss_fun(preds, yb)

    if opt is not None:
        loss.backward()
        opt.step()
        opt.zero_grad()

    metric_result = None
    if metric is not None:
        metric_result = metric(preds, yb)

    return loss.item(), len(xb), metric_result

def evaluate(model, loss_fun, valid_dl, metric=None):
    with torch.no_grad():
        results = [loss_batch(model, loss_fun, xb, yb, metric=metric) for xb, yb in valid_dl]
        losses, nums, metric = zip(*results)

        total = np.sum(nums)

        avg_loss = np.sum(np.multiply(losses, nums)) / total
        avg_metric = None
        if metric is not None:
            avg_metric = np.sum(np.multiply(metric, nums)) / total
        
    return avg_loss, total, avg_metric

def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.sum(preds == labels).item() / len(preds)

def fit(epochs, model, loss_fun, opt, train_dl, valid_dl, metric=None):
    for epoch in range(epochs):
        for xb, yb in train_dl:
            loss, _, _ = loss_batch(model, loss_fun, xb, yb, opt, metric)
        
        result = evaluate(model, loss_fun, valid_dl, metric)
        val_loss, total, val_metric = result

        if metric is None:
            print(f'Epoch {epoch+1}, loss: {val_loss:.4f}')
        else:
            print(f'Epoch {epoch+1}, loss: {val_loss:.4f}, {metric.__name__}: {val_metric:.4f}')

fit(10, model, loss_fun, optimizer, train_dl, validation_dl, accuracy)