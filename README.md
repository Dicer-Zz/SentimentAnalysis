# 面向微博短文本的情感分析

本文通过一些自然语言处理的技术，搭建了一个基于LSTM的微博短文本的情感分析系统。

数据来自[SMP2020微博情感分析竞赛](https://github.com/smp2020ewect/smp2020ewect.github.io)

主要包含文本向量化和建模两个部分。

## 文本向量化

文本向量化使用了gensim实现的word2vec和fasttext两种方式，并且使用word2vec分析了红楼梦中的人物关系，详细见[博客](https://blog.dicer.fun)

## 模型搭建

分别搭建了一个基于朴素贝叶斯的简单二分类系统，和一个基于LSTM的六分类系统。
