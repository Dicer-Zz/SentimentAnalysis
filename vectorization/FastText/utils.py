import re
import jieba
import pandas as pd

def trim(text):
    """
    带有语料清洗功能的分词函数, 包含数据预处理, 可以根据自己的需求重载
    使用re保证了一些本来可能会分开的表情图标不分开
    return: [str]
    """
    
    # 去除 {%xxx%} (地理定位, 微博话题等)
    text = re.sub("\{%.+?%\}", " ", text)

    # 去除 @xxx (用户名)         
    text = re.sub("@.+?( |:)", " ", text)

    # 去除 【xx】 (里面的内容通常都不是用户自己写的) 
    text = re.sub("【.+?】", " ", text)


    icons = re.findall("\[.+?\]", text)             # 提取出所有表情图标
    text = re.sub("\[.+?\]", "IconMark", text)      # 将文本中的图标替换为`IconMark`

    tokens = []
    # for k, w in enumerate(jieba.lcut(text)):
    for w in jieba.cut(text):
        w = w.strip()
        if "IconMark" in w:                         # 将IconMark替换为原图标
            for i in range(w.count("IconMark")):
                tokens.append(icons.pop(0))
        elif w and w != '\u200b' and w.isalpha():   # 只保留有效文本
                tokens.append(w)
    return tokens

def load_corpus(path):
    """
    加载语料库
    """
    data = []
    with open(path, "r", encoding="utf8") as f:
        for line in f:
            [_, sentiment, content] = line.split(",", 2)
            content = trim(content)             # 分词
            data.append((content, int(sentiment)))
    return data

def load_review(filePath):
    """
    加载评论
    return: [[str]]
    """
    S = pd.read_csv(filePath)['review']
    review = []
    for k, v in S.iteritems():
        review.append(trim(v))
    return review

def test():
    filePath = '../../corpus/100k/sample200.csv'
    # review = pd.read_csv(filePath)['review']
    # for index, value in review[:20].iteritems():
    #     print(index, value)
    #     print(trim(value))
    print(load_review(filePath))

if __name__ == '__main__':
    test()