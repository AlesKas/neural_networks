import numpy
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(device)

print(torch.zeros(2,2))