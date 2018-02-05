# coding=gbk

import time
from sklearn import metrics
import pickle as pickle
import pandas as pd


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
    # model = RandomForestClassifier(n_estimators=8)
    model = RandomForestClassifier(min_samples_leaf=1,n_estimators=100)
    model.fit(train_x, train_y)
    print model.feature_importances_    # 显示每一个特征的重要性指标，越大说明越重要
    return model


# Decision Tree Classifier
def decision_tree_classifier(train_x, train_y):
    from sklearn import tree
    model = tree.DecisionTreeClassifier()
    model.fit(train_x, train_y)
    print model.feature_importances_  # 显示每一个特征的重要性指标，越大说明越重要
    return model


# GBDT(Gradient Boosting Decision Tree) Classifier
def gradient_boosting_classifier(train_x, train_y):
    from sklearn.ensemble import GradientBoostingClassifier
    model = GradientBoostingClassifier(n_estimators=200)
    model.fit(train_x, train_y)
    print model.feature_importances_  # 显示每一个特征的重要性指标，越大说明越重要
    return model


# SVM Classifier
def svm_classifier(train_x, train_y):
    from sklearn.svm import SVC
    model = SVC(kernel='rbf', probability=True)
    model.fit(train_x, train_y)
    return model


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

def read_data(data_file):
    data = pd.read_csv(data_file)
    # fun = lambda x: calcLabel(x)
    # data.label = list(map(fun, data.label))
    # print data.head(10) #* 0.5
    train = data[:int(len(data) * 0.9)]
    test = data[int(len(data) * 0.9):]
    train_y = train.label
    train_x = train.drop('label', axis=1)
    test_y = test.label
    test_x = test.drop('label', axis=1)

    # 交叉验证
    # from sklearn.model_selection import KFold
    # from sklearn.model_selection import StratifiedKFold
    # import numpy as np
    # X = np.array(data.drop('label', axis=1))
    # y = data.label.values  # np.array
    # # kf = KFold(n_splits=10)
    # skf = StratifiedKFold(n_splits=10)
    # # for train_index, test_index in kf.split(X):
    # for train_index, test_index in skf.split(X,y):
    #     print("TRAIN:", train_index, "TEST:", test_index)
    #     X_train, X_test = X[train_index], X[test_index]
    #     y_train, y_test = y[train_index], y[test_index]
    # return X_train, y_train, X_test, y_test

    return train_x, train_y, test_x, test_y

if __name__ == '__main__':
    dataname = "201504"
    # data_file = "H:\\Research\\data\\trainCG.csv"
    data_file = "data/" + dataname + "_data.csv"
    thresh = 0.5
    model_save_file = None
    model_save = {}

    test_classifiers = ['NB', 'KNN', 'LR', 'RF', 'DT', 'SVM', 'SVMCV', 'GBDT']
    # GBDT梯度提升决策树
    # test_classifiers = ['NB', 'KNN', 'LR', 'RF', 'DT', 'GBDT']
    # test_classifiers = ['RF']
    classifiers = {'NB': naive_bayes_classifier,
                   'KNN': knn_classifier,
                   'LR': logistic_regression_classifier,
                   'RF': random_forest_classifier,
                   'DT': decision_tree_classifier,
                   'SVM': svm_classifier,
                   'SVMCV': svm_cross_validation,
                   'GBDT': gradient_boosting_classifier
                   }

    print('reading training and testing data...')
    train_x, train_y, test_x, test_y = read_data(data_file)

    for classifier in test_classifiers:
        print('******************* %s ********************' % classifier)
        start_time = time.time()
        model = classifiers[classifier](train_x, train_y)
        # print model.feature_importances_  # 显示每一个特征的重要性指标，越大说明越重要
        print('training took %fs!' % (time.time() - start_time))
        predict = model.predict(test_x)
        if model_save_file != None:
            model_save[classifier] = model
        precision = metrics.precision_score(test_y, predict, average='weighted')
        recall = metrics.recall_score(test_y, predict, average='weighted')
        print('precision: %.2f%%, recall: %.2f%%' % (100 * precision, 100 * recall))
        accuracy = metrics.accuracy_score(test_y, predict)
        print('accuracy: %.2f%%' % (100 * accuracy))

    # 打印特征工程
    import matplotlib.pyplot as plt
    import numpy as np
    # 使用feature_importances_对boston数据的特征进行排序
    from sklearn.ensemble import RandomForestRegressor
    data = pd.read_csv("data/20150407_train.bak.csv")
    train = data[:int(len(data) * 1)]
    train_y = train.label
    train_x = train.drop('label', axis=1)
    feature_names = np.array(train)
    x, y = train_x, train_y
    # feature_name = ['line','count','isBusytime','isWorkday','isIn','isOK','hour','min','label']
    # feature_name = ['min','line','count','hour','isIn','isBusytime','isOK','isWorkday']
    # [5 2 3 4 0 6 7 1]
    feature_name = ['isWorkday', 'isOK', 'isBusytime', 'isIn', 'hour', 'count', 'line', 'min']
    rf = RandomForestRegressor(n_estimators=100, random_state=101).fit(x, y)
    importance = np.mean([tree.feature_importances_ for tree in rf.estimators_], axis=0)
    std = np.std([tree.feature_importances_ for tree in rf.estimators_], axis=0)
    print importance
    indices = np.argsort(importance)
    print indices
    print importance[indices]
    print feature_name
    range_ = range(len(importance))
    print range_

    plt.figure()
    plt.title("random forset importance")
    plt.barh(range_, importance[indices], color='r', xerr=std[indices], alpha=0.4, align='center')
    plt.yticks(range(len(importance)), feature_name[0:8])
    plt.ylim([-1, len(importance)])
    plt.xlim([0.0, 0.65])
    plt.show()

    # 优化调参
    # sample_leaf_options = list(range(1, 500, 3))
    # n_estimators_options = list(range(1, 1000, 5))
    # sample_leaf_options = list(range(1, 200, 10))
    # n_estimators_options = list(range(10, 400, 10))
    # results = []
    # # 优化随机森林
    # from sklearn.ensemble import RandomForestClassifier
    # for leaf_size in sample_leaf_options:
    #     for n_estimators_size in n_estimators_options:
    #         alg = RandomForestClassifier(min_samples_leaf=leaf_size, n_estimators=n_estimators_size, random_state=50)
    #         alg.fit(train_x, train_y)
    #         predict = alg.predict(test_x)
    #         # 用一个三元组，分别记录当前的 min_samples_leaf，n_estimators， 和在测试数据集上的精度
    #         results.append((leaf_size, n_estimators_size, (test_y == predict).mean()))
    #         # 真实结果和预测结果进行比较，计算准确率
    #         print("leaf_size: "+str(leaf_size)+" n_estimators_size: "+str(n_estimators_size)+" precision: "+str((test_y == predict).mean()))
    #

    # if model_save_file != None:
    #     pickle.dump(model_save, open(model_save_file, 'wb'))