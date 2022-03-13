from msilib.schema import Feature
import os
from re import L, match

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import auc

from auction import DATA_FRAME

FILE = os.path.join(os.path.dirname(os.path.realpath(__file__)), "data", "features_train.csv")

# Load csv into dataframe
data = pd.read_csv(FILE)
df = pd.DataFrame(data)
df['time'] = pd.to_datetime(df['time'])

FEATURE_1_A = []
FEATURE_2_A = []

FEATURE_1_B = []
FEATURE_2_B = []

FEATURE_1_C = []
FEATURE_2_C = []

for index, auction in df.iterrows():
    if auction['click'] == 1:
        if auction['bidder_id'] == 'A':
            FEATURE_1_A.append(auction['feature_1'])
            FEATURE_2_A.append(auction['feature_2'])
        elif auction['bidder_id'] == 'B':
            FEATURE_1_B.append(auction['feature_1'])
            FEATURE_2_B.append(auction['feature_2'])
        elif auction['bidder_id'] == 'C':
            FEATURE_1_C.append(auction['feature_1'])
            FEATURE_2_C.append(auction['feature_2'])


# Subtask 1
fig = plt.figure(figsize=(9,9))
ax1 = fig.add_subplot(111)

ax1.scatter(FEATURE_1_A, FEATURE_2_A, c='r')
ax1.scatter(FEATURE_1_B, FEATURE_2_B, c='g')
ax1.scatter(FEATURE_1_C, FEATURE_2_C, c='b')
plt.show()
# Judging by the plots of these features by each bidder, there is none correlation between these features and bidder

# Based on the plot I see no correlation between these to features
# but there are some cluster visible

# So in my opinion one of these factors could be location od advertisment on busy sites,
# the other could be price paid for the advert

# Subtask 2
output = []

for index, auction in DATA_FRAME.iterrows():
    res = df.loc[df['auction_id'] == auction['auction_id']]
    for bidder in ['A', 'B', 'C']:
        if pd.isna(auction[bidder]):
            output.append((auction['auction_id'], bidder, None))
            continue
        prob = res.loc[res['bidder_id'] == bidder]['prob'].values
        online_bid = (auction[bidder] * prob)[0]
        output.append((auction['auction_id'], bidder, online_bid))
        
with open('online_bid.txt', 'w') as file:
    for bid in output:
        file.write(f"Auction: {bid[0]}, bidded: {bid[1]}, online bid: {bid[2]}\n")

# Subtask 3
with open('online_bid_winner.txt', 'w') as file:
    for out in zip(*[iter(output)]*3):
        ordered = sorted(list(out), key=lambda x:float('-inf') if x[2] is None else x[2], reverse=True)
        winner = ordered[0]
        file.write(f"Winner of auction {winner[0]} is {winner[1]} with online bid: {winner[2]}\n")

# Subtask 4

