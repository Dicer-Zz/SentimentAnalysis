import pickle
import jieba
import re

class SentimentAnalyzer():
    def __init__(self, modelPath, stopwordPath, userDictPath):
        self.clt = None
        self.vectorizer = None
        self.tfidftransformer = None
        self.modelPath = modelPath
        self.stopwordPath = stopwordPath
        self.userDictPath = userDictPath
        self.stopWords = []
        self.tokenizer = jieba.Tokenizer()
        self.initalize()

    # 加载模型
    def initalize(self):
        with open(self.stopwordPath, encoding='UTF-8') as words:
            self.stopWords = [word.strip() for word in words.readlines()]

        with open(self.modelPath, 'rb') as f:
            model = pickle.load(f)
            self.clf = model['clf']
            self.vectorizer = model['vectorizer']
            self.tfidftransformer = model['tfidftransformer']

        if self.userDictPath:
            self.tokenizer.load_userdict(self.userDictPath)

    def trim(self, text):
        """
        带有语料清洗功能的分词函数, 包含数据预处理, 可以根据自己的需求重载
        使用re保证了一些本来可能会分开的表情图标不分开
        return: [str]
        """
        text = re.sub("\{%.+?%\}", " ", text)           # 去除 {%xxx%} (地理定位, 微博话题等)
        text = re.sub("@.+?( |:)", " ", text)           # 去除 @xxx (用户名)
        text = re.sub("【.+?】", " ", text)              # 去除 【xx】 (里面的内容通常都不是用户自己写的)
        text = re.sub("[a-zA-Z0-9]", " ", text)         # 去除字母和数字
        icons = re.findall("\[.+?\]", text)             # 提取出所有表情图标
        text = re.sub("\[.+?\]", "IconMark", text)      # 将文本中的图标替换为`IconMark`

        tokens = []
        # for k, w in enumerate(jieba.lcut(text)):
        for w in self.tokenizer.cut(text):
            w = w.strip()
            if "IconMark" in w:                         # 将IconMark替换为原图标
                for i in range(w.count("IconMark")):
                    tokens.append(icons.pop(0))
            elif w and w != '\u200b' and w.isalpha():   # 只保留有效文本
                if w not in self.stopWords:
                    tokens.append(w)

        return ' '.join(tokens)
    
    def predictScore(self, reviews):
        # 清洗 并 分词
        reviews = [self.trim(review) for review in reviews]
        print(reviews)
        tfidf = self.tfidftransformer.transform(self.vectorizer.transform(reviews))
        pred = self.clf.predict_proba(tfidf)
        return pred

    def analyze(self, reviews):
        preds = self.predictScore(reviews)

        for pred in preds:
            print(f'正向: {pred[1]:.4f}, 负向: {pred[0]:.4f}')