import re
import jieba
import time
from gensim.models import word2vec

def segment():
    start = time.time()
    content = open('./data/红楼梦.txt', 'r').read()
    trimed = re.sub(r'[^\u4e00-\u9fa5]', ' ', content)
    jieba.load_userdict('./data/dict.txt')
    result = ' '.join(jieba.cut(trimed))
    file = open('./data/cut_result.txt', 'w+')
    file.write(' '.join(result.split()))
    cost = time.time() - start
    print(f"Segment cost: {cost:.4f}")

def train_model():
    start = time.time()
    sentencePath = './data/cut_result.txt'
    modelPath = './data/hlm_model'
    sentence = word2vec.LineSentence(sentencePath)
    model = word2vec.Word2Vec(sentence, vector_size=200, window=3)
    model.save(modelPath)
    cost = time.time() - start
    print(f'Word2Vec model training cost: {cost:.4f}')

if __name__ == '__main__':
    # segment()
    train_model()
