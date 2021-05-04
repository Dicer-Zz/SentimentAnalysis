import jieba

jieba.load_userdict('./data/user_dict.txt')

reviews = ['不开心', '好开心']

for review in reviews:
    print(jieba.lcut(review))