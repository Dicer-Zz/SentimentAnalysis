import pandas as pd
import snownlp as nlp

path = '../corpus/moods'
filePath = '../corpus/moods/总表.csv'

csv = pd.read_csv(filePath)
moods = {0: '喜悦', 1: '愤怒', 2: '厌恶', 3: '低落'}

print('微博数目（总表）：%d' % csv.shape[0])

for label, mood in moods.items():
    print('微博数目（%s）：%d' % (mood, csv[csv.label == label].shape[0]))

# 将不同情绪的内容分割
for label, mood in moods.items():
    fileName = path + mood + '.csv'
    # 选择对应情绪的行
    fileDF = csv[csv['label'] == label]
    # 不使用index
    fileDF.to_csv(fileName, index=False)
