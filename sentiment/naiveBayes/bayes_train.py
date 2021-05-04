import time
import pickle
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from utils import load_corpus, data_suffle

stopwordPath = './data/stopword.txt'
userDictPath = './data/user_dict.txt'
csvFilePath = '../../corpus/100k/all.csv'
modelPath = './data/bayes.model'

# 载入自定义字典
# jieba.load_userdict(userDictPath)

time_start = time.time()

labels, reviews = load_corpus(csvFilePath, stopwordPath)
labels, reviews = data_suffle(labels, reviews)
# 将reviews的格式转为[str]，为CountVectorizer使用
reviews = [' '.join(review) for review in reviews]

# 1/4 分割数据集
n = len(labels) // 5
labels_train, reviews_train = labels[n:], reviews[n:]
labels_test, reviews_test = labels[:n], reviews[:n]

print(f'Load Corpus Cost {time.time() - time_start:.4f} Sec')
time_start = time.time()

# 加载bayes分类器
# 统计法向量化
vectorizer = CountVectorizer(max_df=0.8, min_df=3)
tfidftransformer = TfidfTransformer()
# 先转换成词频矩阵，再计算TFIDF值
tfidf = tfidftransformer.fit_transform(vectorizer.fit_transform(reviews_train))
# 朴素贝叶斯中的多项式分类器
clf = MultinomialNB().fit(tfidf, labels_train)

print(f'Train Model Cost {time.time() - time_start:.4f} Sec')

with open(modelPath, 'wb') as f:
    pickle.dump({
        "clf": clf,
        "vectorizer": vectorizer,
        "tfidftransformer": tfidftransformer,
    }, f)

print("训练完成")
