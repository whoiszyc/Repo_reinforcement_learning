# Regression Example With Boston Dataset: Baseline

import numpy as np
import pandas
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from scipy.io import loadmat


# load training set
trainSet = loadmat('data_load_forecast/trainSet.mat')
# # split into input (X) and output (Y) variables
trainX = trainSet['trainX']
trainY = trainSet['trainY']


# load testing set
## OPT 1: still using training set
# testSet = loadmat('data_load_forecast/trainSet.mat')
# testX = trainSet['trainX']
# testY = trainSet['trainY']
## OPT 2: using testing set
testSet = loadmat('data_load_forecast/testSet.mat')
testX = testSet['testX']
testY = testSet['testY']


# # find row indices that have nan
indices_nan_train = np.isnan(trainX).any(axis=1)
indices_nan_test = np.isnan(testX).any(axis=1)
# # delete the corresponding rows in both X and Y
trainX = trainX[~indices_nan_train]
trainY = trainY[~indices_nan_train]
testX = testX[~indices_nan_test]
testY = testY[~indices_nan_test]

# scale the power data to per unit
PowerBase = 1
trainX[:, 5:8] = trainX[:, 5:8]/PowerBase
trainY = trainY/PowerBase
testX[:, 5:8] = testX[:, 5:8]/PowerBase
testY = testY/PowerBase

# create model
model = Sequential()
model.add(Dense(5, input_dim=8, activation='relu')) # activation= 'sigmoid'
model.add(Dense(5, activation='relu'))
model.add(Dense(5, activation='relu'))
model.add(Dense(1, activation='relu'))

# Compile model
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mape'])

# Fit the model
model.fit(trainX, trainY, nb_epoch=20, batch_size=100) #

# evaluate the model
scores = model.evaluate(testX, testY)
print("%s: %.2f" % (model.metrics_names[1], scores[1]))

# perform prediction
hatY = model.predict(testX)
error = testY - hatY

# plot
plt.figure()
plt.plot(testY[0+24*300:23+24*300])
plt.plot(hatY[0+24*300:23+24*300])
plt.title('Load Forecasting')
plt.show()

# plot
plt.figure()
plt.plot(error)
plt.title('Load Forecasting Error')
plt.show()

# plot
plt.figure()
plt.plot(testY[0+24*300:23+24*300])
plt.title('Load Forecasting')
plt.show()

