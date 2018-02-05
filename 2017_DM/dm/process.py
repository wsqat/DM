# coding=gbk

import time
from sklearn import metrics
import pickle as pickle
import pandas as pd
import numpy as np

from sklearn import cross_validation
# from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
# 交叉验证
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold



# Multinomial Naive Bayes Classifier
def naive_bayes_classifier(train_x, train_y):
    from sklearn.naive_bayes import MultinomialNB
    model = MultinomialNB(alpha=0.01)
    model.fit(train_x, train_y)
    return model


# KNN Classifier
def knn_classifier(train_x, train_y):
    from sklearn.neighbors import KNeighborsClassifier
    model = KNeighborsClassifier()
    model.fit(train_x, train_y)
    return model


# Logistic Regression Classifier
def logistic_regression_classifier(train_x, train_y):
    from sklearn.linear_model import LogisticRegression
    model = LogisticRegression(penalty='l2')
    model.fit(train_x, train_y)
    return model


# Random Forest Classifier
def random_forest_classifier(train_x, train_y):
    from sklearn.ensemble import RandomForestClassifier
    model = RandomForestClassifier(n_estimators=8)
    model.fit(train_x, train_y)
    return model


# Decision Tree Classifier
def decision_tree_classifier(train_x, train_y):
    from sklearn import tree
    model = tree.DecisionTreeClassifier()
    model.fit(train_x, train_y)
    return model


# GBDT(Gradient Boosting Decision Tree) Classifier
def gradient_boosting_classifier(train_x, train_y):
    from sklearn.ensemble import GradientBoostingClassifier
    model = GradientBoostingClassifier(n_estimators=200)
    model.fit(train_x, train_y)
    return model


# SVM Classifier 二分类
def svm_classifier(train_x, train_y):
    from sklearn.svm import SVC
    model = SVC(kernel='rbf', probability=True)
    model.fit(train_x, train_y)
    return model

# SVM Classifier 多分类
def mul_svm_classifier(train_x, train_y):
    # X = np.array([[-1, -1], [-2, -1], [1, 1], [2, 1]])
    # y = np.array([1, 1, 2, 2])
    from sklearn.svm import SVC
    clf = SVC(decision_function_shape='ovr')
    clf.fit(train_x, train_y)
    # clf.fit(X, y)
    # dec = clf.decision_function([[1]])
    # print dec.shape[1]  # 4 classes: 4*3/2 = 6
    # print clf.predict([[1,4,0,1,1,0,1427836800]])
    # print clf.predict([[-0.8, -1]])
    return clf

# SVM Classifier using cross validation
def svm_cross_validation(train_x, train_y):
    from sklearn.grid_search import GridSearchCV
    from sklearn.svm import SVC
    model = SVC(kernel='rbf', probability=True)
    param_grid = {'C': [1e-3, 1e-2, 1e-1, 1, 10, 100, 1000], 'gamma': [0.001, 0.0001]}
    grid_search = GridSearchCV(model, param_grid, n_jobs=1, verbose=1)
    grid_search.fit(train_x, train_y)
    best_parameters = grid_search.best_estimator_.get_params()
    for para, val in list(best_parameters.items()):
        print(para, val)
    model = SVC(kernel='rbf', C=best_parameters['C'], gamma=best_parameters['gamma'], probability=True)
    model.fit(train_x, train_y)
    return model

def calcLabel(label):
    if label>2:
        return 0
    else:
        return 1


def read_data_for_mul(data_file):

    data = pd.read_csv(data_file)
    X = np.array(data.drop('label', axis=1))
    y = data.label.values  # np.array
    train = data[:int(len(data) * 0.5*1 * 0.9)]
    test = data[int(len(data) * 0.5*1 * 0.1):]
    train_y = train.label.values # np.array
    train_x = np.array(train.drop('label', axis=1))
    test_y = test.label.values
    test_x = np.array((test.drop('label', axis=1)))
    return train_x, train_y, test_x, test_y

    # kf = KFold(n_splits=10)
    # # skf = StratifiedKFold(n_splits=10)
    # for train_index, test_index in kf.split(X):
    # # for train_index, test_index in skf.split(X,y):
    #     print("TRAIN:", train_index, "TEST:", test_index)
    #     X_train, X_test = X[train_index], X[test_index]
    #     y_train, y_test = y[train_index], y[test_index]
    # return X_train, y_train, X_test, y_test



def multiclassSVM(train_x, train_y):
    from sklearn.svm import SVC
    model = OneVsRestClassifier(SVC)
    model.fit(train_x, train_y)
    # predicted = model.predict(X_test)
    # print clf.predict([[1,4,0,1,1,0,1427836800]])
    print model



if __name__ == '__main__':
    dataname = "201504"
    # data_file = "H:\\Research\\data\\trainCG.csv"
    data_file = "data/" + dataname + "_dataset.csv"
    thresh = 0.5
    model_save_file = None
    model_save = {}

    # test_classifiers = ['SVMS']
    # GBDT梯度提升决策树
    # test_classifiers = ['NB', 'KNN', 'LR', 'RF', 'DT', 'GBDT']
    test_classifiers = ["SVM"]
    # test_classifiers = ['SVM']
    classifiers = {'NB': naive_bayes_classifier,
                   'KNN': knn_classifier,
                   'LR': logistic_regression_classifier,
                   'RF': random_forest_classifier,
                   'DT': decision_tree_classifier,
                   'SVM': svm_classifier,
                   'SVMS':mul_svm_classifier,
                   'SVMCV': svm_cross_validation,
                   'GBDT': gradient_boosting_classifier
                   }

    print('reading training and testing data...')
    # train_x, train_y, test_x, test_y = read_data(data_file)
    train_x, train_y, test_x, test_y = read_data_for_mul(data_file)

    for classifier in test_classifiers:
        print('******************* %s ********************' % classifier)
        start_time = time.time()
        model = classifiers[classifier](train_x, train_y)
        # 1, 1, 0, 1, 1, 0, 1428268140, 2
        # 1, 23, 0, 1, 1, 0, 1428308640, 4
        # 1, 73, 0, 1, 1, 0, 1428295320, 5
        print "test begin:"
        print model.predict([[1, 4, 0, 1, 1, 0, 1427836800]])
        print model.predict([[1, 23, 0, 1, 1, 0, 1428308640]])
        print model.predict([[1, 73, 0, 1, 1, 0, 1428295320]])
        print "test end:"
        print('training took %fs!' % (time.time() - start_time))
        predict = model.predict(test_x)
        if model_save_file != None:
            model_save[classifier] = model
        # target_names = target_names =['1','2','3','4','5','6']
        # print model.predict([[1, 4, 0, 1, 1, 0, 1427836800]])
        # precision = metrics.precision_score(test_y, predict,average='weighted')
        precision = metrics.precision_score(test_y, predict,average='weighted')
        recall = metrics.recall_score(test_y, predict,average='weighted')
        print('precision: %.2f%%, recall: %.2f%%' % (100 * precision, 100 * recall))
        accuracy = metrics.accuracy_score(test_y, predict)
        print('accuracy: %.2f%%' % (100 * accuracy))

    if model_save_file != None:
        pickle.dump(model_save, open(model_save_file, 'wb'))