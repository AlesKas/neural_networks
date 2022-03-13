import os

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from scipy import stats
from matplotlib.dates import DateFormatter

FILE = os.path.join(os.path.dirname(os.path.realpath(__file__)), "data", "bids_train.csv")

# Number of wins per player
WINS = {
    'A' : 0,
    'B' : 0,
    'C' : 0
}

# List of prices paid by auction, format (auction_id, price_paid, winner, time)
PRICES = []

# Load csv into dataframe
data = pd.read_csv(FILE)
DATA_FRAME = pd.DataFrame(data)
DATA_FRAME['time'] = pd.to_datetime(DATA_FRAME['time'])

NUM_OF_AUCTIONS = len(DATA_FRAME)

# Iterate rows of csv
for _, auction in DATA_FRAME.iterrows():
    # List of tuplec containig player and price bided
    auction_data = [('A', auction['A']), ('B', auction['B']), ('C', auction['C'])]
    # Sort by price from largest to lowest
    auction_data.sort(key= lambda x: float('-inf') if pd.isna(x[1]) else x[1], reverse=True)
    # Winner is player who bidded the mosts
    winner = auction_data[0][0]
    # Price paid is second largest bid
    price = auction_data[1][1]
    # If there is not any second highest bid, use highest bid
    if pd.isna(price):
        WINS[winner] += 1
        PRICES.append((auction['auction_id'], auction_data[0][1], winner, auction['time']))
        continue
    PRICES.append((auction['auction_id'], price, winner, auction['time']))
    WINS[winner] += 1

# For re-usability of loading the data, I have decided to create this function
def Compute():
    # Compute winrate of of each player, rounded to 2 decimals
    # Subtask 1
    A_WINRATE = round((WINS['A'] / 1000) * 100, 2)
    B_WINRATE = round((WINS['B'] / 1000) * 100, 2)
    C_WINRATE = round((WINS['C'] / 1000) * 100, 2)
    print(f"Player A: {A_WINRATE}%\nPlayer B: {B_WINRATE}%\nPlayer C: {C_WINRATE}%")

    # Subtask 2
    with open('auction.txt', 'w') as file:
        for auction_id, price, winner, _ in PRICES:
            file.write(f"Auction {auction_id}, won by: {winner}, price paid: {price}\n")

    # Subtask 3
    print(f"Total prices paid: {sum(price for _, price, _, _ in PRICES)}")

    # Subtask 4
    only_prices = [price for _, price, _, _ in PRICES]
    # plt.hist(only_prices, bins=100)
    # plt.show()
    # Judging by the plot, it is generalized extreme value distribution, let's perform kolmogorov-smirnov test
    # Form hypothesis, that data comes from this distribution
    dist = getattr(stats, "genextreme")
    param = dist.fit(only_prices)
    d, p = stats.kstest(only_prices, "genextreme", args=param)
    if p < 0.05:
        print("Reject h0, values does not come from this distribution")
    else:
        print("Accept h1, values comes from this distribution.")

    # Subtask 5
    PRICES.sort(key=lambda x: x[1], reverse=True)
    bidders = {x[2] for x in PRICES}
    sums = [(i, sum(x[1] for x in PRICES if x[2] == i)) for i in bidders]
    print(sums)

    sums = [(i, sum(x[1] for x in PRICES if x[2] == i and x[1] > 50)) for i in bidders]
    print(sums)
    # We can see, that bidder C has the biggest sums of bids,
    # althou bidder B is willing to buy more expensive items

    # Subtask 6
    PRICES.sort(key=lambda x: x[3])
    price_evolution = [x[1] for x in PRICES]
    time_evolution = [x[3] for x in PRICES]
    df = pd.DataFrame(data=price_evolution, index=time_evolution, columns=['price'])
    fig, ax = plt.subplots(figsize=(15,5))
    ax.plot(df)
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=1))
    ax.xaxis.set_major_formatter(DateFormatter('%H:%M'))

    plt.show()

if __name__ == '__main__':
    Compute()