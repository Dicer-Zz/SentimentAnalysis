from snownlp import SnowNLP as nlp

s1 = nlp(u'hahaha')
s2 = nlp(u'哈哈哈')
s3 = nlp(u'哈哈哈哈哈哈')
s4 = nlp(u'哈哈哈哈哈哈哈哈哈')

print(s1.sentiments, s2.sentiments, s3.sentiments, s4.sentiments)

s1 = nlp(u'开心')
s2 = nlp(u'快乐')
s3 = nlp(u'不开心')
s4 = nlp(u'不快乐')

print(s1.sentiments, s2.sentiments, s3.sentiments, s4.sentiments)

s1 = nlp(u'有趣')
s2 = nlp(u'无聊')
s3 = nlp(u'傻逼')
s4 = nlp(u'天使')

print(s1.sentiments, s2.sentiments, s3.sentiments, s4.sentiments)

s1 = nlp(u'不知所云')
s2 = nlp(u'耗子尾汁')
s3 = nlp(u'不明觉厉')
s4 = nlp(u'阴阳怪气')

print(s1.sentiments, s2.sentiments, s3.sentiments, s4.sentiments)

s1 = nlp(u'开心')
s2 = nlp(u'不开心')
s3 = nlp(u'很不开心')
s4 = nlp(u'超级不开心')

print(s1.sentiments, s2.sentiments, s3.sentiments, s4.sentiments)