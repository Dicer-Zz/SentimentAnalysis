from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from utils import load_review

modelPath = './d2v.model'
model = Doc2Vec.load(modelPath)

# 词语推断
word_vector = model.wv
print(word_vector['我'])
print(word_vector.most_similar('我'))
print(word_vector.most_similar('开心'))
# print(word_vector.most_similar('[二哈]'))
print(word_vector.most_similar('离谱'))
print(word_vector.most_similar('卧槽'))
# print(word_vector.most_similar('口吐芬芳'))
# print(word_vector.most_similar('耗子尾汁'))

# 句子推断 效果感人
doc_vector = model.dv

filePath = '../../corpus/100k/all.csv'
review = load_review(filePath)

documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(review)]

vector = model.infer_vector(['梦想','有','多大','舞台','就','有','多大','[鼓掌]'])
sims = doc_vector.most_similar([vector], topn=10)
sim_doc = [documents[sim[0]] for sim in sims]
print(sim_doc)