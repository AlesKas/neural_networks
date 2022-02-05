import math
from pyexpat import features
import torch
import torchvision
import numpy as np

from torch.utils.data import Dataset, DataLoader 

class  WineData(Dataset):
    def __init__(self) -> None:
        xy = np.loadtxt('learning/data/wine.csv', delimiter=',', dtype=np.float32, skiprows=1)
        self.x = torch.from_numpy(xy[:, 1:])
        self.y = torch.from_numpy(xy[:, [0]])
        self.n_samples = xy.shape[0]

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return self.n_samples

dataset = WineData()
dataLoader = DataLoader(dataset=dataset, batch_size=4, shuffle=True)


num_epochs = 2
total_samples = len(dataset)
n_iter = math.ceil(total_samples / 4)

for epoch in range(num_epochs):
    for i, (inputs, labels) in enumerate(dataLoader):
        if (i+1) % 5 == 0:
            print(f"epoch {epoch+1}/{num_epochs}, step {i+1}/{n_iter}, inputs {inputs.shape}")
