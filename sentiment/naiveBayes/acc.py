import time
import pickle
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from utils import load_reviews, data_suffle

# stopwordPath = './data/stopword.txt'
# userDictPath = './data/user_dict.txt'
csvFilePath = '../../corpus/100k/allTrimed.csv'
modelPath = './data/bayes.model'

# 载入自定义字典
# jieba.load_userdict(userDictPath)

time_start = time.time()

labels, reviews = load_reviews(csvFilePath)
labels, reviews = data_suffle(labels, reviews)

# 1/4 分割数据集
n = len(labels) // 5
labels_train, reviews_train = labels[n:], reviews[n:]
labels_test, reviews_test = labels[:n], reviews[:n]

print(f'Load Corpus Cost {time.time() - time_start:.4f} Sec')
print(reviews[:5], type(reviews), type(reviews[0]))

time_start = time.time()

vectorizer = CountVectorizer()
vec_train = vectorizer.fit_transform([np.str_(review) for review in reviews_train])
clf = MultinomialNB().fit(vec_train, labels_train)

print(f'Train Model Cost {time.time() - time_start:.4f} Sec')

vec_test = vectorizer.transform([np.str_(review) for review in reviews_test])
print(vec_test.shape, vec_train.shape)

pred = clf.predict(vec_test)

from sklearn import metrics
print(metrics.classification_report(labels_test, pred))
print("准确率:", metrics.accuracy_score(labels_test, pred))