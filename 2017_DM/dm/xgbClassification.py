#! /usr/bin/python
import numpy as np
import xgboost as xgb

# label need to be 0 to num_class -1
# if col 33 is '?' let it be 1 else 0, col 34 substract 1
# data = np.loadtxt('data/201504_train.csv', delimiter=',')
# sz = data.shape
#
# train = data[:int(sz[0] * 0.7), :]  # take row 1-256 as training set
# test = data[int(sz[0] * 0.7):, :]  # take row 257-366 as testing set

# train_X = train[:, 0:33]
# train_Y = train[:, 34]
#
# test_X = test[:, 0:33]
# test_Y = test[:, 34]

import pandas as pd

# dataname = "201504"
# data_file = "data/" + dataname + "_train.csv"
dataname = "201504"
data_file = "data/" + dataname + "_data.csv"
data = pd.read_csv(data_file)
X = np.array(data.drop('label', axis=1))
y = data.label.values  # np.array
train = data[:int(len(data) * 1 * 0.9)]
test = data[int(len(data) * 1 * 0.9):]
train_Y = train.label.values  # np.array
train_X = np.array(train.drop('label', axis=1))
test_Y = test.label.values
test_X = np.array((test.drop('label', axis=1)))

xg_train = xgb.DMatrix(train_X, label=train_Y)
xg_test = xgb.DMatrix(test_X, label=test_Y)
# setup parameters for xgboost
param = {}
param['booster'] = "gbtree"
# use softmax multi-class classification
param['objective'] = 'multi:softmax'
# param['objective'] = 'reg:logistic'
# scale weight of positive examples

param['eta'] = 0.1
param['max_depth'] = 6
param['silent'] = 1
param['nthread'] = 4
param['num_class'] = 6

watchlist = [(xg_train, 'train'), (xg_test, 'test')]
num_round = 5
bst = xgb.train(param, xg_train, num_round, watchlist);
# get prediction
pred = bst.predict(xg_test);

print ('predicting, classification error=%f' % (
sum(int(pred[i]) != test_Y[i] for i in range(len(test_Y))) / float(len(test_Y))))

# do the same thing again, but output probabilities
param['objective'] = 'multi:softprob'
bst = xgb.train(param, xg_train, num_round, watchlist);
# Note: this convention has been changed since xgboost-unity
# get prediction, this is in 1D array, need reshape to (ndata, nclass)
yprob = bst.predict(xg_test).reshape(test_Y.shape[0], 6)
ylabel = np.argmax(yprob, axis=1)  # return the index of the biggest pro

print ('predicting, classification error=%f' % (
sum(int(ylabel[i]) != test_Y[i] for i in range(len(test_Y))) / float(len(test_Y))))
