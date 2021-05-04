from gensim.models import FastText
from utils import trim, load_review
import time

time_start = time.time()

filePath = '../../corpus/100k/all.csv'

review = load_review(filePath)
# model = FastText(vector_size=10, window=2, min_count=3)
# model.build_vocab(sentences=review)
# model.train(sentences=review, total_examples=len(review), epochs=10)

model = FastText(review, vector_size=20, window=2, min_count=5, min_n=2, max_n=4, word_ngrams=1, epochs=10, workers=8)

model.wv.save_word2vec_format('ftModel.txt', binary=False)
model.save('ft.model')

# 36 sec
print(f'Train Model Coast {time.time()-time_start:.4f} Sec')
