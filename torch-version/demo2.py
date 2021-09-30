from gensim.models import FastText
import time
import numpy as np
import pandas as pd


vector_size = 200
# train fasttext model
time_start = time.time()

filePath = '../corpus/6moods/train/usual_trainTrimed.csv'
df = pd.read_csv(filePath)
labels, reviews = df['label'].astype('str'), df['review'].astype('str')
reviews = [str(review).split() for review in reviews]

print(f'Loading reviews Coast {time.time()-time_start:.4f} Sec')
time_start = time.time()

ft = FastText(reviews, vector_size=vector_size, epochs=20, window=2, min_count=5, min_n=2, max_n=4, word_ngrams=1, workers=8)

print(f'Train Model Coast {time.time()-time_start:.4f} Sec')

# convert label from str to int
# mood dict
m2i = {
    'sad':0,
    'angry': 1,
    'fear': 2,
    'neutral': 3,
    'surprise': 4,
    'happy': 5,
}
i2m = {k:i for k, i in enumerate(m2i)}
labels = [m2i[label] for label in labels]

# word2vec and padding
vectors = [[ft.wv[word] for word in review] for review in reviews]
max_len = max([len(vector) for vector in vectors])
zeros = [0 for i in range(vector_size)]
for i in range(len(vectors)):
    while len(vectors[i]) < max_len:
        vectors[i].insert(0, zeros)

# reformat data
def to_categorical(labels):
    num_type = 6
    res = [list(range(num_type))[label] for label in labels]
    return res

vectors = np.array(vectors)
labels = to_categorical(labels)
labels = np.array(labels, dtype=int)
vectors.shape, labels.shape