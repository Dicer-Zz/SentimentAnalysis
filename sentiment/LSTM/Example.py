# -*- coding: utf-8 -*-
# 要添加一个新单元，输入 '# %%'
# 要添加一个新的标记单元，输入 '# %% [markdown]'
# %% [markdown]
# ## 感情分析モデル
# <div style="font-size:1.2em; color:green; margin:20px">
#     テキストの内容がネガティブ・ポジティブ・ニュートラルか判別します。<br>
#     入力：英文テキスト<br>
#     出力：感情分類
# <div>

# %%
"""
必要なライブラリをインストール
 tesorflow 2.3.9
 numpy 1.20.1
 keras 2.4.3
"""
import matplotlib.pyplot as plt
import numpy as np
from keras import layers
import os
import tensorflow as tf
from keras.preprocessing import text, sequence
import csv
import random
import pandas as pd


# %%
"""
データセット取得 
 kaggle Shanshank Yadav氏のPre-processed Twitter tweetsを使用させていただく。
 ラベルはNegative(0) Positive(2) Neutral(1)の３分類
 https://www.kaggle.com/shashank1558/preprocessed-twitter-tweets

"""
x_datas = []
y_datas = []

# オリジナルのデータセットは分類ごとのCSVに分割されている
# 必要に応じてパスを変更する
neg_datas = pd.read_csv('./processedNegative.csv', engine="python")
for elem in neg_datas.columns:
    x_datas.append(elem.replace(".",""))
    y_datas.append(0)

neu_datas = pd.read_csv('./processedNeutral.csv', engine="python")
for elem in neu_datas.columns:
    x_datas.append(elem.replace(".",""))
    y_datas.append(1)

pos_datas = pd.read_csv('./processedPositive.csv', engine="python")
for elem in pos_datas.columns:
    x_datas.append(elem.replace(".",""))
    y_datas.append(2)
    
# データは結合されていることを確認
print(x_datas[0:10000:1000])
print(y_datas[0:10000:1000])


# %%
"""
データセットを訓練データとテストデータに分割（7:3）
"""
from sklearn.model_selection import train_test_split

# 訓練データ：テストデータを7:3に分割
X_train, X_test, Y_train, Y_test = train_test_split(
    x_datas, y_datas, 
    test_size=0.3, random_state=0
)


# %%
"""
テキストをトークン化する
String to INT
"""

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

# テキストの最大長を取得
# 今回はLSTMの入力を固定長にするため
max_len = max(map(len, x_datas))

# トークン化
tokenizer = Tokenizer()
tokenizer.fit_on_texts(X_train)
x_train = tokenizer.texts_to_sequences(X_train)
x_test = tokenizer.texts_to_sequences(X_test)

for text, vector in zip(X_train[0:3], x_train[0:3]):
    print(text)
    print(vector)
    
# 固定長ゼロパディング
x_train = pad_sequences(x_train, maxlen=max_len)
x_test = pad_sequences(x_test, maxlen=max_len)

print(x_train[0])
print(type(x_train))


# %%
"""
正解データをone-hotベクトル化
"""
from keras.utils import np_utils

y_train = np.array(Y_train, dtype=int)
y_train = np_utils.to_categorical(y_train) 
y_test = np.array(Y_test, dtype=int)
y_test = np_utils.to_categorical(y_test) 


# %%
"""
学習モデルの構築
"""

from keras.models import Sequential
from keras.layers import LSTM, Dense, Embedding, Dropout
from keras.optimizers import RMSprop

vocabulary_size = len(tokenizer.word_index) + 1  

model = Sequential()

model.add(Embedding(input_dim=vocabulary_size, output_dim=32))   # Embbedding層ではトークン化されたテキストをベクトル化する
model.add(LSTM(16, return_sequences=False, dropout=0.5))  #　すぐ過学習しがちなのでドロップアウトを入れる
model.add(Dense(3, activation='softmax'))   # ３値分類のためsoftmax関数を使用 

model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])   # RMSpropが学習しやすそう

model.summary()


# %%

# 学習
history = model.fit(
    x_train, y_train, batch_size=32, epochs=10,
    validation_data=(x_test, y_test)
)


# %%



# %%
from matplotlib import pyplot as plt

# 精度のplot
plt.plot(history.history['accuracy'], marker='.', label='acc')
plt.plot(history.history['val_accuracy'], marker='.', label='val_acc')
plt.title('model accuracy')
plt.grid()
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.legend(loc='best')
plt.show()

# 損失のplot
plt.plot(history.history['loss'], marker='.', label='loss')
plt.plot(history.history['val_loss'], marker='.', label='val_loss')
plt.title('model loss')
plt.grid()
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend(loc='best')
plt.show()


# %%
"""
不正解データを表示
P：予測値
A:正解値
"""
pre = model.predict(x_test)
print("P A text")
for i,v in enumerate(pre):
    pre_ans = v.argmax()
    ans = y_test[i].argmax()
    dat = X_test[i]
    if ans == pre_ans: continue
    print(pre_ans, ans, dat)


# %%
# 検証
# input_textに190文字以下の英文を入力する

input_text = ["It's good sunny today! "]
input_text = tokenizer.texts_to_sequences(input_text)
input_text = pad_sequences(input_text, maxlen=max_len)
pre = model.predict(input_text)
pre_ans = pre.argmax()
print(pre_ans)


# %%



# %%



# %%



# %%



# %%



# %%



# %%



# %%



# %%



# %%



# %%



# %%



# %%



# %%



# %%



# %%



# %%



# %%



# %%



# %%



# %%



# %%



# %%



# %%



# %%



# %%



