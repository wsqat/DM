# -*- coding: utf-8 -*-

from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
# from scoring import cost_based_scoring as cbs  # 自己编写的scoring
import pickle

# with open('../data/training_df.pkl', 'rb') as f:  # load数据集
#     df = pickle.load(f)
# with open(r'../data/selected_feat_names.pkl', 'rb') as f:  # 特征和标签的key
#     selected_feat_names = pickle.load(f)
# print("data loaded")

# y = df["attack_type"].values  # 标签，y值
# X = df[selected_feat_names].values  # 所有特征值

import main as m

data_file = "data/201504_data.csv"
print('reading training and testing data...')
train_x, train_y, test_x, test_y = m.read_data(data_file)

rfc = RandomForestClassifier(n_jobs=-1)  # 随机森林分类器

parameters = {
    'min_samples_leaf' : range(1, 3),
    'n_estimators' : [10,20,30,40,50,60,70,80,90,100],
    'criterion': ("gini", "entropy")
}

# scorer = cbs.scorer(show=True)

if __name__ == '__main__':
    gscv = GridSearchCV(rfc, parameters,
                        scoring="accuracy",
                        cv=3,
                        verbose=2,
                        refit=False,
                        n_jobs=1,
                        return_train_score=False)
    gscv.fit(train_x, train_y)
    print(gscv.cv_results_)
    # print(gscv.best_params_, gscv.best_score_)
    print("Best: %f using %s" % (gscv.best_score_, gscv.best_params_))
    print("grid search finished")

# ({'n_estimators': 10, 'criterion': 'gini', 'min_samples_leaf': 1}, 1.0)
