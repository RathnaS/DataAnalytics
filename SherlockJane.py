__author__ = 'Rathna'
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import string

pstemmer = PorterStemmer()

def my_tokenizer(text):
    l_text = text.lower()
    np_lcontent = l_text.translate(string.punctuation)
    tokens = nltk.word_tokenize(np_lcontent)
    tokens_sw_removed = [w for w in tokens if not w in stopwords.words('english')]
    stems = []
    for token in tokens_sw_removed:
        stems.append(pstemmer.stem(token))
    return tokens_sw_removed

file_content_dict = {}
data_files = ['sherlock1.txt', 'jane1.txt', 'sherlock2.txt', 'jane2.txt']

from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer(input=u'filename', analyzer=u'word', max_df=0.95, tokenizer=my_tokenizer, max_features=100)
X_data = vectorizer.fit_transform(data_files)
y_labels = [0,1,0,1]

# print(vectorizer.get_feature_names())
# print(X_data.toarray())

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
ch2 = SelectKBest(score_func=chi2, k=50)
ch2.fit_transform(X_data,y_labels)
top_ranked_features = sorted(enumerate(ch2.scores_),key=lambda x:x[1], reverse=True)[:50]
top_ranked_features_indices = map(list, zip(*top_ranked_features))[0]
sel_kfeatures = (np.array(vectorizer.get_feature_names())[top_ranked_features_indices]).tolist()

# print((np.array(vectorizer.get_feature_names())[top_ranked_features_indices]).tolist())

param_l = 10;
import nltk.data
sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')
data_set = []
y_target = []
target = -1
for file in data_files:
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

# print(data_set)
# print(y_target)

mod_vectorizer = CountVectorizer(input=u'content',analyzer=u'word',vocabulary=sel_kfeatures)
X_dataset = mod_vectorizer.fit_transform(data_set)

# print(len(X_dataset.toarray()[0]))
# print(X_dataset.toarray().shape[0])
# print(len(y_target))
