import torch
import random

import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torchvision.transforms as transforms

from torchvision.datasets import MNIST
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

BATCH_SIZE = 100

dataset = MNIST(root='data/', download=True, train=True, transform=transforms.ToTensor())

# image_tensor, label = dataset[0]
# print(image_tensor.shape, label)

# print(image_tensor[:, 10:15, 10:15])
# print(torch.max(image_tensor), torch.min(image_tensor))

# plt.imshow(image_tensor[0, 10:15, 10:15], cmap='gray')
# plt.show()

# Split dataset into training data and validation data
def split_indices(n, val_pct):
    # Determine size of validation set
    n_val = int(val_pct*n)

    idxs = np.random.permutation(n)

    return idxs[n_val:], idxs[:n_val]

train_indexes, validation_indexes = split_indices(len(dataset), 0.2)
# print(len(train_data), len(validation_data))
# print(f"Sample validation data: ", validation_data[:20])

train_sampler = SubsetRandomSampler(train_indexes)
train_loader = DataLoader(dataset, BATCH_SIZE, sampler=train_sampler)

val_sampler = SubsetRandomSampler(validation_indexes)
val_loader = DataLoader(dataset, BATCH_SIZE, sampler=val_sampler)

input_size = 28 * 28
num_classes = 10


# print(model.weight.shape)
# print(model.weight)
# print(model.bias.shape)
# print(model.bias)

class MnistModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(input_size, num_classes)

    def forward(self, xb):
        xb = xb.reshape(-1, 784)
        out = self.linear(xb)
        return out

# print(model.linear.weight.shape, model.linear.bias.shape)
# print(list(model.parameters()))
# print(model.linear.weight.shape, model.linear.bias.shape)
# print(list(model.parameters()))

# probs = F.softmax(outputs, dim=1)
# print(f"Sample probs: \n{probs[:2].data}")
# print(f"Sum: {torch.sum(probs[0]).item()}")
# max_probs, preds = torch.max(probs, dim=1)
# print(preds)
# print(labels)

loss_fun = F.cross_entropy

# def accuracy(labels1, labels2):
#     return torch.sum(labels1 == labels2).item() / len(labels1)
# Accuracy = e^-loss

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

model = MnistModel()
learning_rate = 0.001
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

fit(10, model, loss_fun, optimizer, train_loader, val_loader, accuracy)

val_loss, total, val_acc = evaluate(model, loss_fun, val_loader, metric=accuracy)
print(f"Loss: {val_loss:.4f}, accuracy: {val_acc:.4f}")

test_dataset = MNIST(root='data/', train=False, transform=transforms.ToTensor())

def predict_image(image, model):
    xb = image.unsqueeze(0)
    yb = model(xb)
    _, preds = torch.max(yb, dim=1)
    return preds[0].item()

# for i in range(3):
#     img, label = random.choice(test_dataset)
#     print(f"Label: {label}, predicted: {predict_image(img, model)}")

test_loader = DataLoader(test_dataset, batch_size=200)
test_loss, total, test_acc = evaluate(model, loss_fun, test_loader, metric=accuracy)
print(f"Loss: {test_loss:.4f}, accuracy: {test_acc:.4f}")

import os

torch.save(model.state_dict(), os.path.join(os.getcwd(), "model/model.t7"))