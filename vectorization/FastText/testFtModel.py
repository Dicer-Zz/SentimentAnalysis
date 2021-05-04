from gensim.models import FastText

modelPath = 'ft.model'
model = FastText.load(modelPath)

print(model.wv['我'])
print(model.wv.most_similar('我'))
print(model.wv.most_similar('开心'))
print(model.wv.most_similar('[二哈]'))
print(model.wv.most_similar('离谱'))
print(model.wv.most_similar('卧槽'))
print(model.wv.most_similar('口吐芬芳'))
print(model.wv.most_similar('耗子尾汁'))