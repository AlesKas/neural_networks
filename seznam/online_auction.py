import os

import pandas as pd
import matplotlib.pyplot as plt

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

for _, auction in df.iterrows():
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

for _, auction in DATA_FRAME.iterrows():
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
winrate_A = 0
winrate_B = 0
winrate_C = 0
count = len(output) / 3
with open('online_bid_winner.txt', 'w') as file:
    for out in zip(*[iter(output)]*3):
        ordered = sorted(list(out), key=lambda x:float('-inf') if x[2] is None else x[2], reverse=True)
        winner = ordered[0]
        if winner[1] == 'A':
            winrate_A += 1
        if winner[1] == 'B':
            winrate_B += 1
        if winner[1] == 'C':
            winrate_C += 1
        file.write(f"Winner of auction {winner[0]} is {winner[1]} with online bid: {winner[2]}\n")

print(f"Winrate of bidder A: {(winrate_A / count) * 100}")
print(f"Winrate of bidder B: {(winrate_B / count) * 100}")
print(f"Winrate of bidder C: {(winrate_C / count) * 100}")
# Probability of click on the advert changed winrate significantly

# Subtask 4
# Subtask 4 is only formula, no tas was assigned here

# Subtask 5
final_price = 0
# Auxiliary list of tulpes, format: [(auction_id, bidder_id, price, prob_of_click, clicked)]
tmp_list = []
for _, auction in DATA_FRAME.iterrows():
    features_row = df.loc[df['auction_id'] == auction['auction_id']]
    for bidder in ['A', 'B', 'C']:
        single_row = features_row.loc[features_row['bidder_id'] == bidder]
        if len(single_row) == 0:
            tmp_list.append((auction['auction_id'], bidder, None, None, None))
            continue
        click_value = None
        if len(single_row['click'] != 0):
            click_value = single_row['click'].values[0]
        tmp_list.append((auction['auction_id'], bidder, auction[bidder], single_row['prob'].values[0], click_value))

for out in zip(*[iter(tmp_list)]*3):
    out = list(out)
    if out[0][4] == 1.0 or out[1][4] == 1.0 or out[2][4] == 1.0:
        out.sort(key=lambda x:(float('-inf') if x[4] is None else x[4], float('-inf') if x[2] is None else x[2]), reverse=True)
        probab_first = out[0][3]
        probab_second = out[1][3]
        price = out[1][2]
        if price is None:
            continue
        final_price += (probab_second / probab_first) * price

print(f"Total earnings of auction: {final_price}")