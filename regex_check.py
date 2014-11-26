__author__ = 'Rathna'
import numpy as np
import nltk
from nltk.corpus import stopwords
import string

from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer(input=u'filename',analyzer=u'word',max_df=0.95,max_features=500)

sample_files = ['sherlock1.txt', 'jane1.txt', 'sherlock2.txt', 'jane2.txt']
X_data = vectorizer.fit_transform(sample_files)
y_labels = [0,1,0,1]
print(vectorizer.get_feature_names())
print(len(vectorizer.get_feature_names()))
print(X_data.toarray())

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
ch2 = SelectKBest(score_func=chi2,k=100)
ch2.fit_transform(X_data,y_labels)

top_ranked_features = sorted(enumerate(ch2.scores_),key=lambda x:x[1], reverse=True)[:100]
top_ranked_features_indices = map(list,zip(*top_ranked_features))[0]
print((np.array(vectorizer.get_feature_names())[top_ranked_features_indices]).tolist())
sel_kfeatures = (np.array(vectorizer.get_feature_names())[top_ranked_features_indices]).tolist()

param_l = 25;
import nltk.data
sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')
data_set = []
y_target = []
target = -1
for file in sample_files:
    text = open(file,'r')
    if "sherlock" in file:
        target = 0
    elif "jane" in file:
        target = 1
    content = text.read().decode()
    sent_list = sent_detector.tokenize(content)
    print(sent_list)
    i = 1
    temp_str = ""
    for x in range(0,len(sent_list)):
        if i%param_l == 0:
            temp_str = temp_str + sent_list[x]
            data_set.append(temp_str)
            y_target.append(target)
            temp_str = ""
            i+=1
        elif x == len(sent_list)-1:
            temp_str = temp_str + sent_list[x]
            data_set.append(temp_str)
            y_target.append(target)
        else:
            temp_str = temp_str + sent_list[x]
            i+=1

print(data_set)
print(y_target)

y_target = np.asarray(y_target)

mod_vectorizer = CountVectorizer(input=u'content',analyzer=u'word',vocabulary=sel_kfeatures)
X_dataset = mod_vectorizer.fit_transform(data_set)

# print(len(X_dataset.toarray()[0]))
print(X_dataset.toarray().shape[0])
print(len(y_target))

prior = [0.5,0.5]
from sklearn.naive_bayes import MultinomialNB
mnb = MultinomialNB(alpha=0.0,fit_prior=False)

mnb.fit(X_dataset, y_target)



num_folds = 5
from sklearn import cross_validation
cv = cross_validation.StratifiedKFold(y_target, n_folds=num_folds, shuffle=True)

from sklearn.metrics import precision_recall_fscore_support

list_precision_fold = [[0, 0]]
list_recall_fold = [[0, 0]]
list_f1_fold = [[0, 0]]
accuracy = 0.0
misclasserror = 0.0

for train_index, test_index in cv:
    X_train, X_test = X_dataset[train_index], X_dataset[test_index]
    y_train, y_test = y_target[train_index], y_target[test_index]
    y_pred = mnb.fit(X_train,y_train).predict(X_test)
    accuracy = accuracy + mnb.score(X_test, y_test)
    misclasserror += float((y_test != y_pred).sum())/len(y_pred)
    p, r, f, s = precision_recall_fscore_support(y_test, y_pred)
    list_precision_fold = np.vstack([list_precision_fold, p])
    list_recall_fold = np.vstack([list_recall_fold, r])
    list_f1_fold = np.vstack([list_f1_fold, f])


avg_precision_scores = [sum([s[j] for s in list_precision_fold])/(len(list_precision_fold)-1) for j in range(len(list_precision_fold[0]))]
avg_recall_scores = [sum([s[j] for s in list_recall_fold])/(len(list_recall_fold)-1) for j in range(len(list_recall_fold[0]))]
avg_f1_scores = [sum([s[j] for s in list_f1_fold])/(len(list_f1_fold)-1) for j in range(len(list_f1_fold[0]))]

print(avg_precision_scores)
print(avg_recall_scores)
print(avg_f1_scores)
print(accuracy/num_folds)
print(misclasserror/num_folds)


