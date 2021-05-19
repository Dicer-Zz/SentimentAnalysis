from gensim.models import word2vec
from sklearn.decomposition import PCA
from matplotlib import pyplot

pyplot.rcParams['font.sans-serif'] = ['Arial Unicode MS']

model = word2vec.Word2Vec.load('./data/hlm_model')

allNames = [x.strip('\n') for x in open('./data/name.txt', 'r').readlines()]
X, names = [], []
for name in allNames:
    try:
        X.append(model.wv[name])
        names.append(name)
    except:
        print("missed name: ", name)

pca = PCA(n_components=2)
result = pca.fit_transform(X)

pyplot.scatter(result[:, 0], result[:, 1])
for i, name in enumerate(names):
	pyplot.annotate(name, xy=(result[i, 0], result[i, 1]))
pyplot.show()
# pyplot.savefig('./data/relation.png', transparent=True)
