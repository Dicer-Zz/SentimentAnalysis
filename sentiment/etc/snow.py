import pandas as pd
from snownlp import SnowNLP as nlp

filePath = '../corpus/moods/喜悦.csv'

df = pd.read_csv(filePath).sample(1000)
count = 0
for index, row in df.iterrows():
    content = row['review']
    s = nlp(content)
    # print(s.sentiments)
    if s.sentiments < 0.7:
        print(s.sentiments, content)
        count += 1
print(count/df.shape[0])
# 利用snownlp交叉印证数据的质量，喜悦的准确度只有大约70%