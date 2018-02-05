#coding=utf-8
from sklearn import metrics
from sklearn import cross_validation
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
import numpy as np
import pandas as pd
from numpy import random

dataname = "20150401"
data_file = "data/" + dataname + "_train.csv"
data = pd.read_csv(data_file)
# fun = lambda x: calcLabel(x)
# data.label = list(map(fun, data.label))
# print data.head(10)  # * 0.5
data = data[:int(len(data) * 1)]
y = data.label.values  # np.array
X = np.array(data.drop('label', axis=1))
#
# X=np.arange(15).reshape(5,3)
# y=np.arange(5)
# print X
# print y
Y_1 = np.arange(5)
random.shuffle(Y_1)
Y_2 = np.arange(5)
random.shuffle(Y_2)
Y = np.c_[Y_1,Y_2]

def multiclassSVM():
    X_train, test_x, y_train, test_y = cross_validation.train_test_split(X, y, test_size=0.2,random_state=0)
    model = OneVsRestClassifier(SVC())
    model.fit(X_train, y_train)
    # predicted = model.predict(test_x)
    # print predicted
    predict = model.predict(test_x)
    precision = metrics.precision_score(test_y, predict, average='weighted')
    recall = metrics.recall_score(test_y, predict, average='weighted')
    print('precision: %.2f%%, recall: %.2f%%' % (100 * precision, 100 * recall))
    accuracy = metrics.accuracy_score(test_y, predict)
    print('accuracy: %.2f%%' % (100 * accuracy))

    # return model
def multilabelSVM():
    Y_enc = MultiLabelBinarizer().fit_transform(Y)
    X_train, X_test, Y_train, Y_test = cross_validation.train_test_split(X, Y_enc, test_size=0.2, random_state=0)
    model = OneVsRestClassifier(SVC())
    model.fit(X_train, Y_train)
    predicted = model.predict(X_test)
    # print predicted
    # return model
if __name__ == '__main__':
    multiclassSVM()


