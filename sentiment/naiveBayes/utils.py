# -*- coding: utf-8 -*-
import random
import re

import jieba
import pandas as pd


def trim(text):
    """
    带有语料清洗功能的分词函数, 包含数据预处理, 可以根据自己的需求重载
    使用re保证了一些本来可能会分开的表情图标不分开
    return: [str]
    """
    text = re.sub("\{%.+?%\}", " ", text)           # 去除 {%xxx%} (地理定位, 微博话题等)
    # text = re.sub("@.+?( |$)", " ", text)           # 去除 @xxx (用户名)
    text = re.sub("@.+?( |:)", " ", text)           # 去除 @xxx (用户名)
    text = re.sub("【.+?】", " ", text)              # 去除 【xx】 (里面的内容通常都不是用户自己写的)
    text = re.sub("[a-zA-Z0-9]", " ", text)         # 去除字母和数字
    icons = re.findall("\[.+?\]", text)             # 提取出所有表情图标
    text = re.sub("\[.+?\]", "IconMark", text)      # 将文本中的图标替换为`IconMark`

    tokens = []
    # for k, w in enumerate(jieba.lcut(text)):
    jieba.load_userdict('./data/user_dict.txt')
    for w in jieba.cut(text):
        w = w.strip()
        if "IconMark" in w:                         # 将IconMark替换为原图标
            for i in range(w.count("IconMark")):
                tokens.append(icons.pop(0))
        elif w and w != '\u200b' and w.isalpha():   # 只保留有效文本
            tokens.append(w)
    return tokens


def load_corpus(csvFilePath, stopwordPath):
    """
    加载语料库，并进行分词，数据清洗，去除停用词
    """
    # 数据读取
    df = pd.read_csv(csvFilePath)
    stopword = load_stopword(stopwordPath)
    labels, reviews = df['label'].to_list(), df['review'].to_list()
    trimedReviews = []
    for review in reviews:
        # 数据清洗
        trimedReview = trim(review)
        # 去除停用词
        finalReview = []
        for word in trimedReview:
            if word not in stopword:
                finalReview.append(word)
        trimedReviews.append(finalReview)
    return labels, trimedReviews

def load_reviews(csvFilePath):
    df = pd.read_csv(csvFilePath)
    return df['label'], df['review']

def load_stopword(filePath):
    """
    加载停用词
    """
    with open(filePath, encoding='UTF-8') as words:
        stopword = [word.strip() for word in words]
    return stopword


def data_suffle(labels, reviews):
    """
    打乱数据
    """
    join = list(zip(labels, reviews))
    random.shuffle(join)
    labels, reviews = zip(*join)
    return list(labels), list(reviews)

def pre_trim(csvFilePath, stopwordPath):
    """
    预处理csv文本，并持久化
    """
    df = pd.read_csv(csvFilePath)
    _, reviews = load_corpus(csvFilePath, stopwordPath)
    for index in range(len(reviews)):
        reviews[index] = ' '.join(reviews[index])
    df['review'] = reviews
    df.to_csv(csvFilePath[:-4] + 'Trimed.csv', index=False)

if __name__ == '__main__':
    # csvFilePath = '../../corpus/100k/all.csv'
    # stopwordPath = './data/stopword.txt'
    # pre_trim(csvFilePath, stopwordPath)

    csvFilePath = '../../corpus/100k/sample200Trimed.csv'
    _, reviews = load_reviews(csvFilePath)
    print(reviews)