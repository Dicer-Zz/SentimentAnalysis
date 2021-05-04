from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from utils import load_review
import time

time_start = time.time()

filePath = '../../corpus/100k/all.csv'
review = load_review(filePath)

documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(review)]
# print(document)
model = Doc2Vec(documents, vector_size=20, window=2, min_count=5, epochs=10)
model.save_word2vec_format('d2v.txt')
model.save('d2v.model')

# all.csv 78Sec vector_size=20, min_count=5, epochs=10, workers=8
# all.csv 75Sec vector_size=20, min_count=5, epochs=10, workers=4
# all.csv 69Sec vector_size=20, min_count=5, epochs=10, workers=2
# all.csv 75Sec vector_size=20, min_count=5, epochs=10, workers=1
print(f'Train Model Cost {time.time() - time_start:.4f} Sec')