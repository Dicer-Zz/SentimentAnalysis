from bayes_analyzer import SentimentAnalyzer
from utils import trim

modelPath = './data/bayes.model'
stopwordPath = './data/stopword.txt'
userDictPath = './data/user_dict.txt'

SA = SentimentAnalyzer(modelPath, stopwordPath, userDictPath)
reviews = ['  我是个恋爱脑，我谈恋爱总想把一切甜蜜的小细节都在vb和我的朋友分享，给别人也多一份快乐。老马是一个不爱发动态的人，但他会发一些仅我可见的文字，原来这个表面上看起来粗枝大叶的男生心底里也有细腻的一面[哈哈]']

SA.analyze(reviews)