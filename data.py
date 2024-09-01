import numpy as np
import pandas as pd
import math as math


df = pd.read_csv('yahoo_stock.csv')
df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)


data = np.array(df)
data_rows, data_columns = data.shape

input_data = data[:,1:]
input_data = input_data.T
idata_rows, idata_columns = input_data.shape
for i in range(idata_rows):
    input_data[i] = input_data[i] / np.max(input_data[i])

x = np.array(input_data[0:-1, :])
y = np.array(input_data[-1, :])


train_range = 1248

y_train = np.array(y[0:(train_range)])
x_train = np.array(x[:, 0:(train_range)])

y_test = np.array(y[(train_range):])
x_test = np.array(x[:, (train_range):])