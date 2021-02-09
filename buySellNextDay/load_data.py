# the prediction algorithm here is based on a tutorial by Yacoub Ahmed
import pandas as pd
from sklearn import preprocessing
import numpy as np

# determine how many days of data are used in predictions (recommended 50)
history_points = 50

def csv_to_dataset(csv_path):
    print("Path to data is: %s"%csv_path)
    data = pd.read_csv(csv_path)
    # don't need the date, just need the order of the data
    data=data.drop('timestamp', axis=1)
    # remove the opening day, usually an IPO is anomalous 
    data=data.drop(0,axis=0)
    # normalize the data 
    data_normaliser = preprocessing.MinMaxScaler()
    data_normalised = data_normaliser.fit_transform(data)
    # each value in ohlcv_histories is a 50 point numpy array containing open, high, low, close, and volume values for a stock going from oldest to newest
    # this is the x values for the neural network
    ohlcv_histories_normalised = np.array([data_normalised[i  : i + history_points].copy() for i in range(len(data_normalised) - history_points)])
    # we are predicting the opening price for the next day
    # these are our y values
    # note: [i+history_points] is the index immediately after the range [i  : i + history_points]
    # CHANGE THIS VALUE IF YOU WANT TO PREDICT FARTHER INTO THE FUTURE
    next_day_open_values_normalised = np.array([data_normalised[:,0][i + history_points].copy() for i in range(len(data_normalised) - history_points)])
    next_day_open_values_normalised = np.expand_dims(next_day_open_values_normalised, -1)

    next_day_open_values = np.array([data.iloc[:, 0][i + history_points].copy() for i in range(len(data) - history_points)])
    # expand dimensions to 
    next_day_open_values = np.expand_dims(next_day_open_values, -1)

    # y_nomralizer is to reverse normalization when results are produced 
    y_normaliser = preprocessing.MinMaxScaler()
    y_normaliser.fit(next_day_open_values)

    # use technical indicators like sma value to augment data
    def calc_ema(values, time_period):
        # https://www.investopedia.com/ask/answers/122314/what-exponential-moving-average-ema-formula-and-how-ema-calculated.asp
        sma = np.mean(values[:, 3])
        ema_values = [sma]
        k = 2 / (1 + time_period)
        for i in range(len(his) - time_period, len(his)):
            close = his[i][3]
            ema_values.append(close * k + ema_values[-1] * (1 - k))
        return ema_values[-1]

    technical_indicators = []
    for his in ohlcv_histories_normalised:
        # note since we are using his[3] we are taking the SMA of the closing price
        sma = np.mean(his[:, 3])
        macd = calc_ema(his, 12) - calc_ema(his, 26)
        # technical_indicators.append(np.array([sma]))
        technical_indicators.append(np.array([sma,macd,]))

    technical_indicators = np.array(technical_indicators)

    tech_ind_scaler = preprocessing.MinMaxScaler()
    technical_indicators_normalised = tech_ind_scaler.fit_transform(technical_indicators)


    # check that the number of x points matched the number of y points
    assert ohlcv_histories_normalised.shape[0] == next_day_open_values_normalised.shape[0] == technical_indicators_normalised.shape[0]
    return ohlcv_histories_normalised, technical_indicators_normalised, next_day_open_values_normalised, next_day_open_values, y_normaliser

