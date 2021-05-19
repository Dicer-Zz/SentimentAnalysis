# -*- coding: utf-8 -*-
import random
import pandas as pd

def load_reviews(csvFilePath):
    df = pd.read_csv(csvFilePath)
    return df['label'].astype('str'), df['review'].astype('str')

def data_suffle(labels, reviews):
    """
    打乱数据
    """
    join = list(zip(labels, reviews))
    random.shuffle(join)
    labels, reviews = zip(*join)
    return list(labels), list(reviews)
