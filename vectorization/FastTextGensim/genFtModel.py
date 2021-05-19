from gensim.models import FastText
from utils import load_reviews
import time

time_start = time.time()

filePath = '../../corpus/100k/allTrimed.csv'
_, reviews = load_reviews(filePath)
reviews = [str(review).split() for review in reviews]

print(f'Loading reviews Coast {time.time()-time_start:.4f} Sec')
time_start = time.time()

model = FastText(reviews, vector_size=100, window=2, min_count=5, min_n=2, max_n=4, word_ngrams=1, epochs=10, workers=8)

model.wv.save_word2vec_format('./data/ftModel.txt', binary=False)
model.save('./data/ft.model')

print(f'Train Model Coast {time.time()-time_start:.4f} Sec')
