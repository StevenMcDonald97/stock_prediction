# This file will train a LSTM classifier on a given stock
# provided the stock's data is already download from Alpha Vantage 
# in the data folder
# To usa call: python model.py <stock_symbol> 
# ex. python model.py AMZN

import sys
import keras
from keras.models import Model
from keras.layers import Dense, Dropout, LSTM, Input, Activation, concatenate
from keras import optimizers
import numpy as np
np.random.seed(4)
import tensorflow as tf
tf.random.set_seed(4)
from load_data import csv_to_dataset, history_points

ohlcv_histories, technical_indicators, percent_change_values = csv_to_dataset('data/%s.csv' % sys.argv[1])

test_split = 0.9
n = int(ohlcv_histories.shape[0] * test_split)

ohlcv_train = ohlcv_histories[:n]
tech_ind_train = technical_indicators[:n]
y_train = percent_change_values[:n]

ohlcv_test = ohlcv_histories[n:]
tech_ind_test = technical_indicators[n:]
y_test = percent_change_values[n:]

percent_change_values_test = percent_change_values[n:]

# model architecture 
# create model
lstm_input = Input(shape=(history_points, 5), name='lstm_input')
dense_input = Input(shape=(technical_indicators.shape[1],), name='tech_input')

# create one branch for processing timeseries
x = LSTM(50, name='lstm_0')(lstm_input)
x = Dropout(0.2, name='lstm_dropout_0')(x)
lstm_branch = Model(inputs=lstm_input,outputs=x)

# create another branch for technical indicators
y=Dense(20, name='tech_dense_0')(dense_input)
y=Activation("relu", name='tech_relu_0')(y)
y=Dropout(0.2,name='tech_dropout_0')(y)
technical_indicators_branch = Model(inputs=dense_input, outputs=y)

# combine the output of the two branches
combined = concatenate([lstm_branch.output, technical_indicators_branch.output], name='concatenate')
# finish processing merged branches
z = Dense(64, activation="sigmoid", name='dense_pooling')(combined)
z = Dense(1, activation="linear", name='dense_out')(z)

# train model with an ADAM optimizer
print(lstm_branch.input.shape)
print(technical_indicators_branch.input.shape)
model = Model(inputs=[lstm_branch.input, technical_indicators_branch.input], outputs=z)
adam = optimizers.Adam(lr=0.0005)
model.compile(optimizer=adam, loss='mse')
model.fit(x=[ohlcv_train, tech_ind_train], y=y_train, batch_size=32, epochs=50, shuffle=True, validation_split=0.1)
model.save(f'model/{sys.argv[1]}_model.h5')

# evaluate model
print(ohlcv_test.shape)
print(tech_ind_test.shape)
y_test_predicted = model.predict([ohlcv_test, tech_ind_test])
# y_test_predicted = y_normaliser.inverse_transform(y_test_predicted)
# y_predicted = model.predict([ohlcv_histories, technical_indicators])
# y_predicted = y_normaliser.inverse_transform(y_predicted)

assert percent_change_values_test.shape == y_test_predicted.shape

# calculate the mean squared error
real_mse = np.mean(np.square(percent_change_values_test - y_test_predicted))
# scaled_mse = real_mse / (np.max(percent_change_values_test) - np.min(percent_change_values_test)) * 100
print(f'Mean Squared Error for {sys.argv[1]} is: {real_mse}')

import matplotlib.pyplot as plt

plt.gcf().set_size_inches(11, 7, forward=True)

start = 0
end = -1

real = plt.plot(percent_change_values_test[start:end], label='real')
pred = plt.plot(y_test_predicted[start:end], label='predicted')

# real = plt.plot(unscaled_y[start:end], label='real')
# pred = plt.plot(y_predicted[start:end], label='predicted')

plt.legend(['Real', 'Predicted'])
plt.savefig(f'graphs/{sys.argv[1]}.pdf')
plt.show()

# from datetime import datetime
# model.save(f'model/{sys.argv[1]}_model.h5')
