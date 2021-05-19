from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from utils import load_reviews

modelPath = './data/d2v.model'
model = Doc2Vec.load(modelPath)

# 词语推断
word_vector = model.wv
# print(word_vector['我'])
# print(word_vector.most_similar('我'))
print(word_vector.most_similar('开心'))
# print(word_vector.most_similar('[二哈]'))
print(word_vector.most_similar('离谱'))
print(word_vector.most_similar('卧槽'))
# print(word_vector.most_similar('口吐芬芳'))
# print(word_vector.most_similar('耗子尾汁'))

# 句子推断 效果感人
doc_vector = model.dv

filePath = '../../corpus/100k/allTrimed.csv'
_, reviews = load_reviews(filePath)
reviews = [str(review).split() for review in reviews]

documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(reviews)]

vector = model.infer_vector(['梦想','有','多大','舞台','就','有','多大','[鼓掌]'])
print(vector)
sims = doc_vector.most_similar([vector], topn=10)
sim_doc = [documents[sim[0]] for sim in sims]
print(sim_doc)