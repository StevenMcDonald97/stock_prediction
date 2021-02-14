# the prediction algorithm here is based on a tutorial by Yacoub Ahmed
import pandas as pd
from sklearn import preprocessing
import numpy as np

# determine how many days of data are used in predictions (recommended 50)
history_points = 50

def csv_to_dataset(csv_path):
    # print("Path to data is: %s"%csv_path)
    data = pd.read_csv(csv_path)
    # don't need the date, just need the order of the data
    data=data.drop('timestamp', axis=1)
    # remove the opening day, usually an IPO is anomalous 
    data=data.drop(0,axis=0)
    # normalize the data 
    data_normaliser = preprocessing.MinMaxScaler()
    data_normalised = data_normaliser.fit_transform(data)
    # each value in ohlcv_histories is an array of 50 numpy arrays each containing open, high, low, close, and volume values for a stock going from oldest to newest
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
    # calculate on balance volume for current day based off of last time_period number of days
    def calc_obv(values, time_period):
        # https://www.investopedia.com/terms/o/onbalancevolume.asp
        obv = 0
        prev_close = values[0][3]
        for i in range(len(values) - time_period, len(values)):
            vol = values[i][4]
            close=values[i][3]
            if close>prev_close:
                obv=obv+vol
            elif close<prev_close:
                obv=obv-vol
            else:
                obv=obv
            prev_close=close
        return obv


    # Relative Strength Index using last time_period number of days
    def calc_rsi(values, time_period):
        # https://www.ig.com/us/trading-strategies/a-trader-s-guide-to-the-relative-strength-index--rsi--190320
        # RSI = 100 – 100 / (1 + RS)
        rs_gain=0
        rs_loss=0
        for i in range(len(values) - time_period, len(values)):
            change=values[i][0]
            if (change>=0):
                rs_gain += change
            else:
                rs_loss -= change
        
        if(rs_loss == 0):
            return 100

        rs = rs_gain/rs_loss
        return 100-100/(1+rs)

    technical_indicators = []
    for his in ohlcv_histories_normalised:
        # note since we are using his[3] we are taking the SMA of the closing price
        sma = np.mean(his[:, 3])
        macd = calc_ema(his, 12) - calc_ema(his, 26)
        obv=calc_obv(his,50)
        rsi=calc_rsi(his,14)
        # technical_indicators.append(np.array([sma]))
        # technical_indicators.append(np.array([macd]))
        # technical_indicators.append(np.array([obv]))
        # technical_indicators.append(np.array([rsi])) # JUST RSI WORKS BEST FOR ADBE
        # technical_indicators.append(np.array([sma,rsi,]))
        technical_indicators.append(np.array([sma,macd,obv,rsi]))

    technical_indicators = np.array(technical_indicators)

    tech_ind_scaler = preprocessing.MinMaxScaler()
    technical_indicators_normalised = tech_ind_scaler.fit_transform(technical_indicators)


    # check that the number of x points matched the number of y points
    assert ohlcv_histories_normalised.shape[0] == next_day_open_values_normalised.shape[0] == technical_indicators_normalised.shape[0]
    return ohlcv_histories_normalised, technical_indicators_normalised, next_day_open_values_normalised, next_day_open_values, y_normaliser

