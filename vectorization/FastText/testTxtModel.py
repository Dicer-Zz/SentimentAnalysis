import gensim
import time
from gensim.models.keyedvectors import KeyedVectors

# Load Model
time_start = time.time()
model = KeyedVectors.load_word2vec_format('./ftModel.txt', binary=False)
time_end = time.time()
print('Load model time cost: %.4fs' % (time_end - time_start))

# Test Model

print(model.similarity('猫', '狗'))
print(model.similarity('我', '你'))
print(model.similarity('爸爸', '妈妈'))

print(model.similar_by_word('猫'))
print(model.similar_by_word('狗'))
print(model.similar_by_word('开心'))
print(model.similar_by_word('开森'))
print(model.similar_by_word('难过'))
print(model.similar_by_word('呜呜'))