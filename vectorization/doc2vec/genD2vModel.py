from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from utils import load_reviews
import time

start = time.time()

filePath = '../../corpus/100k/allTrimed.csv'
_, reviews = load_reviews(filePath)
reviews = [review.split() for review in reviews]

cost = time.time() - start
print(f'Loading reviews cost: {cost:.4f} Sec')
start = time.time()

documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(reviews)]
# print(document)
model = Doc2Vec(documents, vector_size=20, window=2, min_count=5, epochs=10)
model.save_word2vec_format('./data/d2v.txt')
model.save('./data/d2v.model')

cost = time.time() - start
print(f'Training model cost: {cost:.4f} Sec')