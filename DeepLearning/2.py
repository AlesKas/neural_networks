import torch
import torchvision.transforms as transforms

from torchvision.datasets import MNIST
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from logistic_regression import MnistModel, evaluate, accuracy, loss_fun

model = MnistModel()
model.load_state_dict(torch.load('model/model.t7'))

test_dataset = MNIST(root='data/', train=False, transform=transforms.ToTensor())

test_loader = DataLoader(test_dataset, batch_size=500)
test_loss, total, test_acc = evaluate(model, loss_fun, test_loader, metric=accuracy)
print(f"Loss: {test_loss:.4f}, accuracy: {test_acc:.4f}")