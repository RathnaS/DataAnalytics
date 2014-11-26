__author__ = 'Rathna'
import numpy
from sklearn import datasets
iris = datasets.load_iris()

from sklearn.naive_bayes import GaussianNB
nb = GaussianNB()

num_folds = 10
from sklearn import cross_validation
cv = cross_validation.StratifiedKFold(iris.target, n_folds=num_folds, shuffle=True)

print(iris.target)

from sklearn.metrics import precision_recall_fscore_support

list_precision_fold = [[0, 0, 0]]
list_recall_fold = [[0, 0, 0]]
list_f1_fold = [[0, 0, 0]]
accuracy = 0.0;

for train_index, test_index in cv:
    X_train, X_test = iris.data[train_index], iris.data[test_index]
    y_train, y_test = iris.target[train_index], iris.target[test_index]
    y_pred = nb.fit(X_train,y_train).predict(X_test)
    accuracy = accuracy + nb.score(X_test, y_test)
    p, r, f, s = precision_recall_fscore_support(y_test, y_pred)
    list_precision_fold = numpy.vstack([list_precision_fold, p])
    list_recall_fold = numpy.vstack([list_recall_fold, r])
    list_f1_fold = numpy.vstack([list_f1_fold, f])


avg_precision_scores = [sum([s[j] for s in list_precision_fold])/(len(list_precision_fold)-1) for j in range(len(list_precision_fold[0]))]
avg_recall_scores = [sum([s[j] for s in list_recall_fold])/(len(list_recall_fold)-1) for j in range(len(list_recall_fold[0]))]
avg_f1_scores = [sum([s[j] for s in list_f1_fold])/(len(list_f1_fold)-1) for j in range(len(list_f1_fold[0]))]

print(avg_precision_scores)
print(avg_recall_scores)
print(avg_f1_scores)
print(accuracy/num_folds)

