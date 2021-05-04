from bayes_analyzer import SentimentAnalyzer

modelPath = './data/bayes.model'
stopwordPath = './data/stopword.txt'
userDictPath = './data/user_dict.txt'

SA = SentimentAnalyzer(modelPath, stopwordPath, userDictPath)
reviews = ['哈哈']

SA.analyze(reviews)