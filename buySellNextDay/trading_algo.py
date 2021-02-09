import numpy as np
from keras.models import load_model
from load_data import csv_to_dataset, history_points
import sys

model = load_model(f'model/{sys.argv[1]}_model.h5')

ohlcv_histories, technical_indicators, next_day_open_values, unscaled_y, y_normaliser = csv_to_dataset(f'data/{sys.argv[1]}.csv')

test_split = 0.9
n = int(ohlcv_histories.shape[0] * test_split)

ohlcv_train = ohlcv_histories[:n]
tech_ind_train = technical_indicators[:n]
y_train = next_day_open_values[:n]

ohlcv_test = ohlcv_histories[n:]
tech_ind_test = technical_indicators[n:]
y_test = next_day_open_values[n:]

unscaled_y_test = unscaled_y[n:]

y_test_predicted = model.predict([ohlcv_test, tech_ind_test])
y_test_predicted = y_normaliser.inverse_transform(y_test_predicted)

buys = []
sells = []
buy_threshold = .01
sell_threshold = .02

start = 0
end = -1
# x will be the index representing a specific day
x = -1
bought=False;

# predict the price for the next day at each day in the data, and decide to buy or sell stock based on 
# whetehr the predicted value clears a given threshol 
for ohlcv, ind in zip(ohlcv_test[start: end], tech_ind_test[start: end]):
    normalised_price_today = ohlcv[-1][0]
    normalised_price_today = np.array([[normalised_price_today]])
    # revert normaliztion on price
    price_today = y_normaliser.inverse_transform(normalised_price_today)
    ohlcv = np.expand_dims(ohlcv, axis=0)
    ind = np.expand_dims(ind, axis=0)
    # predict price for tomorrow and turn into 0D array
    predicted_price_tomorrow = np.squeeze(y_normaliser.inverse_transform(model.predict([ohlcv, ind])))
    delta = (predicted_price_tomorrow - price_today)/price_today
    # always sell the day after you buy
    if (not bought):
        if delta > buy_threshold:
            buys.append((x, price_today[0][0]))
            bought=True;
    else:
        sells.append((x, price_today[0][0]))
        bought=False;
    
        
    x += 1

print(f"buys: {len(buys)}")
print(f"sells: {len(sells)}")
print(f'Number of days calculated over: {ohlcv_test.shape[0]}')


def compute_earnings(buys_, sells_):
    # keep everything in terms of $10 worth of stock to help with performance comparisons
    purchase_amt = 10
    stock = 0
    balance = 0
    # assume that you have some budget on how much you can spend
    budget=-100
    while len(buys_) > 0 and len(sells_) > 0:
        if buys_[0][0] < sells_[0][0]:
            # time to buy $10 worth of stock if there's room in the budget 
            if (balance > budget):
                balance -= purchase_amt
                stock += purchase_amt / buys_[0][1]
            buys_.pop(0)
        else:
            # time to sell all of our stock
            balance += stock * sells_[0][1]
            stock = 0
            sells_.pop(0)
    print(f"earnings: ${balance}")


# we create new lists so we dont modify the original
compute_earnings([b for b in buys], [s for s in sells])

import matplotlib.pyplot as plt

plt.gcf().set_size_inches(11, 7, forward=True)

real = plt.plot(unscaled_y_test[start:end], label='real')
pred = plt.plot(y_test_predicted[start:end], label='predicted')
if len(buys) > 0:
    plt.scatter(list(list(zip(*buys))[0]), list(list(zip(*buys))[1]), c='#00ff00', s=50)
if len(sells) > 0:
    plt.scatter(list(list(zip(*sells))[0]), list(list(zip(*sells))[1]), c='#ff0000', s=50)

# real = plt.plot(unscaled_y[start:end], label='real')
# pred = plt.plot(y_predicted[start:end], label='predicted')

plt.legend(['Real', 'Predicted', 'Buy', 'Sell'])

plt.show()
