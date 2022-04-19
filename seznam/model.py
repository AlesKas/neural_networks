import os
from statistics import mode
from sklearn.metrics import auc
import torch

import numpy as np
import pandas as pd
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch.distributions as D

from torch.utils.data import TensorDataset, DataLoader

# Use GPU if possible
if torch.cuda.is_available():  
  dev = "cuda:0" 
else:  
  dev = "cpu"  

CONVERT_OPTIONS = {'A':1, 'B':2, 'C':3}

TRAIN_FILE = os.path.join(os.path.dirname(os.path.realpath(__file__)), "data", "features_train.csv")
BIDS_TEST_FILE = os.path.join(os.path.dirname(os.path.realpath(__file__)), "data", "bids_test.csv")
FEATURES_TEST_FILE = os.path.join(os.path.dirname(os.path.realpath(__file__)), "data", "features_test.csv")

# Load features, which affects probab. of click
inputs = pd.read_csv(TRAIN_FILE, usecols=['feature_1', 'feature_2'])
targets = pd.read_csv(TRAIN_FILE, usecols=['prob'])
#inputs['bidder_id'] = inputs['bidder_id'].replace(CONVERT_OPTIONS)

# Convers data to tensors and move them to GPU if possible
inputs = torch.tensor(inputs.values, dtype=torch.float32).to(device=dev)
targets = torch.tensor(targets.values, dtype=torch.float32).to(device=dev)

# Create dataloader
train_ds = TensorDataset(inputs, targets)
batch_size = 5
train_dl = DataLoader(train_ds, batch_size)

class LinearModel(nn.Module):
    def __init__(self, x, y) -> None:
        super().__init__()
        self.linear = nn.Linear(x, y)
        self.linear2 = nn.Linear(y, y)
        self.linear3 = nn.Linear(y, y)
        self.linear4 = nn.Linear(y, y)
        self.linear5 = nn.Linear(y, y)

    # Aplly sigmoid application function to prediction
    def forward(self, xb):
        out = torch.sigmoid(self.linear(xb))
        out = self.linear2(out)
        out = self.linear3(out)
        out = self.linear4(out)
        out = self.linear5(out)
        return out

# Set up model, optimizer and loss function
learning_rate = 1e-5
model = LinearModel(inputs.shape[1], targets.shape[1]).to(device=dev)
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
loss_fun = F.mse_loss

for epoch in range(100):
    for xb, yb in train_dl:
        pred = model(xb)
        loss = loss_fun(pred, yb)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1}, Loss: {loss:.4f}")



# Model predicts value with decenet accuracy relatively fast
# I used standard linear regression with mean square error loss function
# and stochastic gradient descent optimizer 

EARNINGS = 0

bids_test = pd.read_csv(BIDS_TEST_FILE)
features_test = pd.read_csv(FEATURES_TEST_FILE)
for auction_id in range(len(bids_test)):
    bids_row = bids_test.loc[bids_test['auction_id'] == auction_id]
    # Tmp dict to store probab. of click per bidder
    tmp_dict = {}
    # Tmp list to store bids
    tmp_list = []
    for bidder in ['A', 'B', 'C']:
        bid = None if pd.isna(bids_row[bidder].item()) else bids_row[bidder].item()
        tmp_list.append((bidder, bid))
        # Select corresponding rows
        features_row = features_test.loc[(features_test['auction_id'] == auction_id) & (features_test['bidder_id'] == bidder)]
        if len(features_row) == 0:
            continue
        feat_1 = features_row['feature_1'].item()
        feat_2 = features_row['feature_2'].item()
        # Compute probab. of click
        nn_input = torch.tensor([feat_1, feat_2], dtype=torch.float32).to(device=dev)
        with torch.no_grad():
            pred = model(nn_input)
        tmp_dict[bidder] = {
            "pred" : pred.item(),
            "click" : features_row['click'].item()
        }
    for key, val in tmp_dict.items():
        if val['click'] == 1.0:
            tmp_list.sort(key=lambda x: x[1], reverse=True)
            prob_first = tmp_dict[tmp_list[0][0]]['pred']
            prob_second = tmp_dict[tmp_list[1][0]]['pred']
            bid_second = tmp_list[1][1]
            price = (prob_second / prob_first) * bid_second
            EARNINGS += price

print(f"Total Earnings: {EARNINGS:.4f}")