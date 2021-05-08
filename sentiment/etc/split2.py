import pandas as pd

path = '../corpus/100k/'
filePath = path + 'all.csv'

df = pd.read_csv(filePath)
negDf = df[df['label'] == 0]
posDf = df[df['label'] == 1]

negDf.to_csv(path + 'neg.csv', index=False)
posDf.to_csv(path + 'pos.csv', index=False)