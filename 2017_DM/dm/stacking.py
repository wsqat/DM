from sklearn import datasets
import pandas as pd

def read_data(data_file):
    data = pd.read_csv(data_file)
    # data = data[int(len(data) * 0.):]
    y = data.label
    X = data.drop('label', axis=1)
    return X,y

dataname = "201504"
data_file = "data/" + dataname + "_data.csv"

# iris = datasets.load_iris()
# X, y = iris.data[:, 1:3], iris.target
X,y = read_data(data_file) # 514184

from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from mlxtend.classifier import StackingClassifier
import numpy as np

clf1 = KNeighborsClassifier(n_neighbors=5) #1:0.43 5:0.50 100: 0.55
clf2 = RandomForestClassifier(min_samples_leaf=1, n_estimators=200, criterion='gini')  # (random_state=1): 0.44  100:0.53
clf3 = GradientBoostingClassifier(n_estimators=200) #0: 0.47
clf4 = DecisionTreeClassifier() # 0.43
# clf5 = GaussianNB()

lr = LogisticRegression()
sclf = StackingClassifier(classifiers=[clf1, clf2, clf3, clf4],
                          meta_classifier=lr)
sclf1 = StackingClassifier(classifiers=[clf1, clf2, clf3, clf4],
                          use_probas=True,
                          average_probas=False,
                          meta_classifier=lr)
print('5-fold cross validation:\n')

for clf, label in zip([clf1, clf2, clf3, clf4, sclf, sclf1],
                      ['KNN',
                       'Random Forest',
                       'GBDT',
                       'DT',
                       # 'Naive Bayes',
                       'StackingClassifier',
                       'StackingClassifier2']):
    scores = model_selection.cross_val_score(clf, X, y,
                                             cv=5, scoring='accuracy')
    print("Accuracy: %0.2f (+/- %0.2f) [%s]"
          % (scores.mean(), scores.std(), label))