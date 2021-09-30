from gensim.models import FastText

modelPath = './data/ft.model'
model = FastText.load(modelPath)

# print(model.wv['我'])
# print(model.wv.most_similar('我'))
# print(model.wv.most_similar('开心'))
# print(model.wv.most_similar('[二哈]'))
# print(model.wv.most_similar('离谱'))
# print(model.wv.most_similar('卧槽'))
# print(model.wv.most_similar('口吐芬芳'))
# print(model.wv.most_similar('耗子尾汁'))
# print(model.wv.most_similar('欢天喜地'))

wv = model.wv
words = ['焦作', '热', '河南', '理工大学', '没有', '空调', '这', '合理', '吗', '住在', '六楼', '跟', '桑拿房', '一样', '半夜', '热到', '睡不着', '好不容易', '睡着', '一会儿', '醒来', '一身', '汗', '太难', '各位', '校友', '都', '咋', '挺', '过来', '我要', '热', '哭', '了']
vecs = [wv[word] for word in words]
print(vecs)