import pandas as pd
from snownlp import SnowNLP as nlp

path = '../corpus/100k/'

negFile = path + 'neg.csv'
posFile = path + 'pos.csv'

sampleSize = 100
negDf = pd.read_csv(negFile).sample(sampleSize)
posDf = pd.read_csv(posFile).sample(sampleSize)

negCnt = 0
for index, row in negDf.iterrows():
    content = row['review']
    s = nlp(content)
    if s.sentiments < 0.5:
        negCnt += 1
    else:
        # 输出与snownlp分析相异的微博
        print("sent: %.4f, content: %s" % (s.sentiments, content))

posCnt = 0
for index, row in posDf.iterrows():
    content = row['review']
    s = nlp(content)
    if s.sentiments > 0.5:
        posCnt += 1
    else:
        print("sent: %.4f, content: %s" % (s.sentiments, content))

print("neg acc: ", negCnt/sampleSize)
print("pos acc: ", posCnt/sampleSize)

# 通过分析结果可以发现snownlp的情感分析有时候有很大的问题，对否定词没有较好的结果。
