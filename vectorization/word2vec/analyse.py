from gensim.models import word2vec

def analyse():
    model = word2vec.Word2Vec.load('./data/hlm_model')
    print('Nearest 宝玉:',model.wv.most_similar(['宝玉']))
    print('Nearest 黛玉:',model.wv.most_similar(['黛玉']))
    print('Nearest 宝钗:',model.wv.most_similar(['宝钗']))
    print('Nearest 晴雯:',model.wv.most_similar(['晴雯']))
    print('Nearest 袭人:',model.wv.most_similar(['袭人']))
    print('Nearest 贾母:',model.wv.most_similar(['贾母']))

    print(model.wv.doesnt_match(u"贾宝玉 薛宝钗 林黛玉 史湘云".split()))
    print(model.wv.doesnt_match(u"黛玉 元春 探春 迎春 惜春".split()))
    print(model.wv.doesnt_match(u"贾琏 贾政 贾赦 贾敬".split()))

    print(model.wv.similarity('贾宝玉','林黛玉'))
    print(model.wv.similarity('林黛玉','薛宝钗'))
    print(model.wv.similarity('晴雯', '袭人'))
    
    print(model.wv.similarity('林黛玉','薛宝钗'))
    print(model.wv.similarity('林黛玉','宝钗'))
    print(model.wv.similarity('黛玉','薛宝钗'))
    print(model.wv.similarity('黛玉','宝钗'))

    who = model.wv['宝玉'] - model.wv['宝钗'] + model.wv['黛玉']
    print(model.wv.most_similar(positive=[who]))
    who = model.wv['宝玉'] - model.wv['黛玉'] + model.wv['宝钗']
    print(model.wv.most_similar(positive=[who]))

    who = model.wv['宝玉'] + model.wv['丫鬟']
    print(model.wv.most_similar(positive=[who]))

    who = model.wv['黛玉'] + model.wv['丫鬟']
    print(model.wv.most_similar(positive=[who]))

    who = model.wv['宝玉'] - model.wv['袭人'] + model.wv['黛玉']
    print(model.wv.most_similar(positive=[who]))
    
if __name__ == '__main__':
    analyse()